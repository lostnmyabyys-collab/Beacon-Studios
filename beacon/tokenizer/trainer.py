"""Tokenizer training."""

from pathlib import Path
from typing import Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, lowercase, strip
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from beacon.logging import get_logger
from beacon.storage.manager import StorageManager

logger = get_logger(__name__)


class TokenizerTrainer:
    """Trains new tokenizers."""

    def __init__(self, storage: StorageManager) -> None:
        """Initialize tokenizer trainer.

        Args:
            storage: Storage manager
        """
        self.storage = storage
        logger.info("Initialized tokenizer trainer")

    def train_bpe(
        self,
        training_files: list[str],
        vocab_size: int = 50257,
        min_frequency: int = 2,
        output_name: str = "tokenizer",
    ) -> Tokenizer:
        """Train BPE tokenizer.

        Args:
            training_files: List of training file paths
            vocab_size: Vocabulary size
            min_frequency: Minimum frequency threshold
            output_name: Output tokenizer name

        Returns:
            Tokenizer: Trained tokenizer
        """
        try:
            # Create tokenizer
            tokenizer = Tokenizer(BPE())

            # Configure
            tokenizer.normalizer = Sequence([
                strip(),
                lowercase(),
            ])
            tokenizer.pre_tokenizer = Whitespace()

            # Train
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=["<unk>", "<s>", "</s>"],
            )

            tokenizer.train(training_files, trainer=trainer)
            logger.info(f"Trained BPE tokenizer with vocab size {vocab_size}")

            return tokenizer
        except Exception as e:
            logger.error(f"Failed to train tokenizer: {e}")
            raise

    def save_tokenizer(self, tokenizer: Tokenizer, name: str) -> None:
        """Save trained tokenizer.

        Args:
            tokenizer: Tokenizer to save
            name: Tokenizer name
        """
        try:
            path = f"tokenizers/{name}.json"
            tokenizer_json = tokenizer.to_str()
            self.storage.write(path, tokenizer_json.encode())
            logger.info(f"Saved tokenizer: {name}")
        except Exception as e:
            logger.error(f"Failed to save tokenizer: {e}")
            raise
