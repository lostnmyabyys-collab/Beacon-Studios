"""Model registry manager."""

from pathlib import Path

from beacon.logging import get_logger
from beacon.registry.metadata import ModelMetadata
from beacon.registry.versioning import VersionInfo
from beacon.storage.manager import StorageManager
from beacon.utils.serialization import load_json, save_json

logger = get_logger(__name__)


class ModelRegistryManager:
    """Manages model registry and metadata."""

    def __init__(self, storage: StorageManager, registry_path: str = "models") -> None:
        """Initialize registry manager.

        Args:
            storage: Storage manager
            registry_path: Path to registry
        """
        self.storage = storage
        self.registry_path = registry_path
        self.models: dict[str, ModelMetadata] = {}
        self.storage.makedirs(registry_path)
        self._load_registry()
        logger.info(f"Initialized model registry at {registry_path}")

    def _load_registry(self) -> None:
        """Load registry from storage."""
        try:
            files = self.storage.list_files(self.registry_path)
            for file in files:
                if file.endswith(".json"):
                    try:
                        path = f"{self.registry_path}/{file}"
                        data = self.storage.read(path)
                        metadata_dict = load_json(path)
                        metadata = ModelMetadata(**metadata_dict)
                        self.models[metadata.model_id] = metadata
                    except Exception as e:
                        logger.error(f"Error loading model metadata {file}: {e}")
        except Exception as e:
            logger.error(f"Error loading registry: {e}")

    def register_model(self, metadata: ModelMetadata) -> None:
        """Register model.

        Args:
            metadata: Model metadata
        """
        self.models[metadata.model_id] = metadata
        # Save to storage
        path = f"{self.registry_path}/{metadata.model_id}.json"
        save_json(metadata.model_dump(), path)
        logger.info(f"Registered model: {metadata.model_name} v{metadata.version}")

    def get_model(self, model_id: str) -> ModelMetadata | None:
        """Get model metadata.

        Args:
            model_id: Model ID

        Returns:
            ModelMetadata: Model metadata or None
        """
        return self.models.get(model_id)

    def list_models(self, model_type: str | None = None) -> list[ModelMetadata]:
        """List models.

        Args:
            model_type: Optional filter by type (vx, neo, cz)

        Returns:
            list: List of models
        """
        models = list(self.models.values())
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return sorted(models, key=lambda m: m.created_at, reverse=True)

    def get_latest_model(self, model_name: str) -> ModelMetadata | None:
        """Get latest version of model.

        Args:
            model_name: Model name

        Returns:
            ModelMetadata: Latest model or None
        """
        models = [m for m in self.models.values() if m.model_name == model_name]
        if not models:
            return None

        # Sort by version
        models.sort(
            key=lambda m: VersionInfo.from_string(m.version),
            reverse=True,
        )
        return models[0]

    def update_model(self, model_id: str, metadata: ModelMetadata) -> None:
        """Update model metadata.

        Args:
            model_id: Model ID
            metadata: Updated metadata
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")

        self.models[model_id] = metadata
        # Save to storage
        path = f"{self.registry_path}/{model_id}.json"
        save_json(metadata.model_dump(), path)
        logger.info(f"Updated model: {model_id}")

    def delete_model(self, model_id: str) -> None:
        """Delete model from registry.

        Args:
            model_id: Model ID
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")

        del self.models[model_id]
        # Delete from storage
        path = f"{self.registry_path}/{model_id}.json"
        self.storage.delete(path)
        logger.info(f"Deleted model: {model_id}")
