"""Inference engine."""

from pathlib import Path
from typing import Any, Optional

import torch

from beacon.config.schema import RuntimeConfig
from beacon.logging import get_logger
from beacon.models.config import ModelConfig
from beacon.models.transformer import TransformerDecoder
from beacon.registry.manager import ModelRegistryManager
from beacon.runtime.cache import KVCache
from beacon.storage.manager import StorageManager
from beacon.tokenizer.manager import TokenizerManager
from beacon.utils.device import get_device

logger = get_logger(__name__)


class InferenceEngine:
    """High-performance inference engine."""

    def __init__(
        self,
        config: RuntimeConfig,
        storage: StorageManager,
        registry: ModelRegistryManager,
        tokenizer_manager: TokenizerManager,
    ):
        """Initialize inference engine.

        Args:
            config: Runtime configuration
            storage: Storage manager
            registry: Model registry
            tokenizer_manager: Tokenizer manager
        """
        self.config = config
        self.storage = storage
        self.registry = registry
        self.tokenizer_manager = tokenizer_manager
        self.device = get_device()
        self.loaded_models: dict[str, TransformerDecoder] = {}
        self.kv_caches: dict[str, KVCache] = {}
        logger.info(f"Initialized inference engine on device: {self.device}")

    def load_model(self, model_id: str, version: str = "latest") -> TransformerDecoder:
        """Load model.

        Args:
            model_id: Model ID
            version: Model version

        Returns:
            TransformerDecoder: Loaded model
        """
        # Check cache
        cache_key = f"{model_id}:{version}"
        if cache_key in self.loaded_models:
            logger.debug(f"Using cached model: {cache_key}")
            return self.loaded_models[cache_key]

        # Get model metadata
        metadata = self.registry.get_model(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")

        # Create model
        model_config = ModelConfig(
            vocab_size=metadata.metadata.get("vocab_size", 50257),
            hidden_size=metadata.metadata.get("hidden_size", 768),
            num_attention_heads=metadata.metadata.get("num_attention_heads", 12),
            num_hidden_layers=metadata.metadata.get("num_hidden_layers", 12),
            max_position_embeddings=metadata.context_length,
        )
        model = TransformerDecoder(model_config)

        # Load checkpoint
        try:
            checkpoint_path = metadata.checkpoint_path
            checkpoint_data = self.storage.read(f"{checkpoint_path}/checkpoint.pt")
            checkpoint = torch.load(
                checkpoint_data,
                map_location=self.device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")

        model = model.to(self.device)
        model.eval()

        # Cache
        self.loaded_models[cache_key] = model

        logger.info(f"Loaded model: {model_id} v{version}")
        return model

    def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.9,
        **kwargs: Any,
    ) -> str:
        """Generate text.

        Args:
            model_id: Model ID
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            **kwargs: Additional arguments

        Returns:
            str: Generated text
        """
        # Load model
        model = self.load_model(model_id)

        # Tokenize
        tokenizer = self.tokenizer_manager.load_tokenizer("gpt2")
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids]).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # Decode
        output_text = tokenizer.decode(output_ids[0].tolist())
        return output_text

    def clear_cache(self) -> None:
        """Clear model cache."""
        self.loaded_models.clear()
        self.kv_caches.clear()
        logger.info("Cleared inference engine cache")
