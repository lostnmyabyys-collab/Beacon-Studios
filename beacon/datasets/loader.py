"""Dataset loader for multiple formats."""

import json
from pathlib import Path
from typing import Any, Generator

import pandas as pd

from beacon.logging import get_logger

logger = get_logger(__name__)


class DatasetLoader:
    """Loads datasets from various formats."""

    @staticmethod
    def load_jsonl(path: str | Path) -> Generator[dict[str, Any], None, None]:
        """Load JSONL file.

        Args:
            path: Path to JSONL file

        Yields:
            dict: Data rows
        """
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    @staticmethod
    def load_json(path: str | Path) -> list[dict[str, Any]]:
        """Load JSON file.

        Args:
            path: Path to JSON file

        Returns:
            list: Data rows
        """
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    @staticmethod
    def load_csv(path: str | Path) -> list[dict[str, Any]]:
        """Load CSV file.

        Args:
            path: Path to CSV file

        Returns:
            list: Data rows
        """
        df = pd.read_csv(path)
        return df.to_dict(orient="records")

    @staticmethod
    def load_txt(path: str | Path, delimiter: str = "\n") -> list[str]:
        """Load TXT file.

        Args:
            path: Path to TXT file
            delimiter: Line delimiter

        Returns:
            list: Lines
        """
        with open(path) as f:
            content = f.read()
        return content.split(delimiter)

    @staticmethod
    def load_markdown(path: str | Path) -> str:
        """Load Markdown file.

        Args:
            path: Path to Markdown file

        Returns:
            str: Content
        """
        with open(path) as f:
            return f.read()

    @staticmethod
    def load_from_kaggle(dataset_name: str, force_download: bool = False) -> Path:
        """Load dataset from Kaggle.

        Args:
            dataset_name: Kaggle dataset identifier (e.g., 'user/dataset-name')
            force_download: Force download even if cached

        Returns:
            Path: Path to downloaded dataset
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise ImportError(
                "kaggle library not installed. Install with: pip install kaggle"
            )

        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Download dataset
        download_path = Path("./datasets") / dataset_name.replace("/", "_")
        download_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading Kaggle dataset: {dataset_name}")
        api.dataset_download_files(
            dataset_name,
            path=str(download_path),
            unzip=True,
            quiet=False,
        )

        logger.info(f"Downloaded to: {download_path}")
        return download_path

    @staticmethod
    def auto_load(path: str | Path) -> Any:
        """Auto-detect and load file format.

        Args:
            path: Path to file

        Returns:
            Loaded data
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".jsonl":
            return list(DatasetLoader.load_jsonl(path))
        elif suffix == ".json":
            return DatasetLoader.load_json(path)
        elif suffix == ".csv":
            return DatasetLoader.load_csv(path)
        elif suffix == ".txt":
            return DatasetLoader.load_txt(path)
        elif suffix == ".md":
            return DatasetLoader.load_markdown(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
