"""Tokenizer management."""

from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizer

from beacon.logging import get_logger
from beacon.storage.manager import StorageManager

logger = get_logger(__name__)


class TokenizerManager:
    """Manages tokenizers for models."""

    def __init__(self, storage: StorageManager, cache_dir: str = "./tokenizers") -> None:
        """Initialize tokenizer manager.

        Args:
            storage: Storage manager
            cache_dir: Cache directory for tokenizers
        """
        self.storage = storage
        self.cache_dir = cache_dir
        self.tokenizers: dict[str, PreTrainedTokenizer] = {}
        self.storage.makedirs(cache_dir)
        logger.info(f"Initialized tokenizer manager at {cache_dir}")

    def load_tokenizer(self, tokenizer_name: str, force_download: bool = False) -> PreTrainedTokenizer:
        """Load tokenizer.

        Args:
            tokenizer_name: Tokenizer name or HF model ID
            force_download: Force download even if cached

        Returns:
            PreTrainedTokenizer: Loaded tokenizer
        """
        # Check cache
        if tokenizer_name in self.tokenizers and not force_download:
            logger.debug(f"Using cached tokenizer: {tokenizer_name}")
            return self.tokenizers[tokenizer_name]

        # Load from HuggingFace
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizers[tokenizer_name] = tokenizer
            logger.info(f"Loaded tokenizer: {tokenizer_name}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer {tokenizer_name}: {e}")
            raise

    def save_tokenizer(self, tokenizer: PreTrainedTokenizer, name: str) -> None:
        """Save tokenizer.

        Args:
            tokenizer: Tokenizer to save
            name: Tokenizer name
        """
        try:
            path = Path(self.cache_dir) / name
            path.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(path))
            logger.info(f"Saved tokenizer: {name}")
        except Exception as e:
            logger.error(f"Failed to save tokenizer {name}: {e}")
            raise

    def tokenize(self, tokenizer_name: str, text: str, **kwargs) -> dict:
        """Tokenize text.

        Args:
            tokenizer_name: Tokenizer name
            text: Text to tokenize
            **kwargs: Additional arguments for tokenizer

        Returns:
            dict: Tokenization result
        """
        tokenizer = self.load_tokenizer(tokenizer_name)
        return tokenizer(text, **kwargs)

    def decode(self, tokenizer_name: str, token_ids: list[int], **kwargs) -> str:
        """Decode tokens to text.

        Args:
            tokenizer_name: Tokenizer name
            token_ids: Token IDs
            **kwargs: Additional arguments for tokenizer

        Returns:
            str: Decoded text
        """
        tokenizer = self.load_tokenizer(tokenizer_name)
        return tokenizer.decode(token_ids, **kwargs)

    def get_vocab_size(self, tokenizer_name: str) -> int:
        """Get vocabulary size.

        Args:
            tokenizer_name: Tokenizer name

        Returns:
            int: Vocabulary size
        """
        tokenizer = self.load_tokenizer(tokenizer_name)
        return len(tokenizer.get_vocab())

    def get_special_tokens(self, tokenizer_name: str) -> dict[str, int]:
        """Get special token IDs.

        Args:
            tokenizer_name: Tokenizer name

        Returns:
            dict: Special token name to ID mapping
        """
        tokenizer = self.load_tokenizer(tokenizer_name)
        return {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }

    def clear_cache(self) -> None:
        """Clear tokenizer cache."""
        self.tokenizers.clear()
        logger.info("Cleared tokenizer cache")
