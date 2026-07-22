"""Streaming generation."""

import asyncio
from typing import AsyncGenerator, Optional

import torch

from beacon.logging import get_logger

logger = get_logger(__name__)


class StreamingGenerator:
    """Handles streaming text generation."""

    def __init__(self, model: torch.nn.Module, tokenizer):
        """Initialize streaming generator.

        Args:
            model: Model for generation
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        logger.info("Initialized streaming generator")

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.9,
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling

        Yields:
            str: Generated tokens
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids])

        for _ in range(max_tokens):
            # Generate next token
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    output_hidden_states=False,
                )
                next_logits = outputs["logits"][:, -1, :]

            # Apply temperature
            next_logits = next_logits / temperature

            # Top-k filtering
            if top_k is not None:
                indices_to_remove = (
                    next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                )
                next_logits[indices_to_remove] = float("-inf")

            # Top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_logits, descending=True
                )
                cumsum = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumsum > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Decode token
            token_text = self.tokenizer.decode(next_token[0].tolist())
            yield token_text

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Allow async operations
            await asyncio.sleep(0)
