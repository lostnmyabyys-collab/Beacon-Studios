"""Dataset manager."""

from pathlib import Path
from typing import Any

from beacon.datasets.cleaner import DatasetCleaner
from beacon.datasets.loader import DatasetLoader
from beacon.logging import get_logger
from beacon.storage.manager import StorageManager
from beacon.utils.serialization import load_json, save_json

logger = get_logger(__name__)


class DatasetManager:
    """Manages datasets."""

    def __init__(self, storage: StorageManager, datasets_path: str = "datasets") -> None:
        """Initialize dataset manager.

        Args:
            storage: Storage manager
            datasets_path: Path to datasets directory
        """
        self.storage = storage
        self.datasets_path = datasets_path
        self.dataset_registry: dict[str, dict[str, Any]] = {}
        self.storage.makedirs(datasets_path)
        logger.info(f"Initialized dataset manager at {datasets_path}")

    def import_dataset(
        self,
        name: str,
        source: str,
        format: str = "auto",
        clean: bool = True,
        **metadata: Any,
    ) -> str:
        """Import dataset.

        Args:
            name: Dataset name
            source: Source path or Kaggle ID
            format: File format (auto, jsonl, json, csv, txt, md)
            clean: Apply cleaning
            **metadata: Additional metadata

        Returns:
            str: Dataset path
        """
        try:
            # Check if Kaggle dataset
            if "/" in source and not Path(source).exists():
                logger.info(f"Loading from Kaggle: {source}")
                source = str(DatasetLoader.load_from_kaggle(source))

            # Load data
            if format == "auto":
                data = DatasetLoader.auto_load(source)
            else:
                loader_method = getattr(
                    DatasetLoader, f"load_{format}", None
                )
                if not loader_method:
                    raise ValueError(f"Unknown format: {format}")
                data = loader_method(source)

            # Convert to list if generator
            if not isinstance(data, list):
                data = list(data)

            # Clean
            if clean:
                data = DatasetCleaner.clean(data)

            # Save
            dataset_path = f"{self.datasets_path}/{name}"
            self.storage.makedirs(dataset_path)

            # Save data as JSONL
            jsonl_path = f"{dataset_path}/data.jsonl"
            for row in data:
                self.storage.write(
                    jsonl_path,
                    f"{{".encode() + str(row).encode() + "}}\n".encode(),
                )

            # Save metadata
            metadata_dict = {
                "name": name,
                "source": source,
                "format": format,
                "size": len(data),
                **metadata,
            }
            save_json(
                metadata_dict,
                Path(f"{dataset_path}/metadata.json"),
            )

            self.dataset_registry[name] = metadata_dict
            logger.info(
                f"Imported dataset: {name} ({len(data)} samples)"
            )
            return dataset_path

        except Exception as e:
            logger.error(f"Failed to import dataset: {e}")
            raise

    def list_datasets(self) -> list[str]:
        """List all datasets.

        Returns:
            list: Dataset names
        """
        try:
            datasets = self.storage.list_files(self.datasets_path)
            return [d for d in datasets if d != "metadata.json"]
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return []

    def get_dataset_info(self, name: str) -> dict[str, Any]:
        """Get dataset info.

        Args:
            name: Dataset name

        Returns:
            dict: Dataset metadata
        """
        try:
            metadata_path = f"{self.datasets_path}/{name}/metadata.json"
            metadata_data = self.storage.read(metadata_path)
            return load_json(metadata_path)
        except Exception as e:
            logger.error(f"Failed to get dataset info: {e}")
            return {}

    def delete_dataset(self, name: str) -> None:
        """Delete dataset.

        Args:
            name: Dataset name
        """
        try:
            dataset_path = f"{self.datasets_path}/{name}"
            self.storage.delete(dataset_path)
            if name in self.dataset_registry:
                del self.dataset_registry[name]
            logger.info(f"Deleted dataset: {name}")
        except Exception as e:
            logger.error(f"Failed to delete dataset: {e}")
            raise
