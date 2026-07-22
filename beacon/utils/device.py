"""Device management and detection utilities."""

import os
from typing import Literal

import torch


class DeviceManager:
    """Manages device detection and configuration."""

    def __init__(self) -> None:
        """Initialize device manager."""
        self._device = self._detect_device()
        self._cuda_available = torch.cuda.is_available()
        self._cuda_device_count = torch.cuda.device_count() if self._cuda_available else 0

    def _detect_device(self) -> torch.device:
        """Detect available device.

        Returns:
            torch.device: Available device (cuda, mps, or cpu)
        """
        # Check environment override
        device_env = os.getenv("DEVICE", "").lower()
        if device_env == "cpu":
            return torch.device("cpu")
        if device_env == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")

        # Auto-detect
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        """Get primary device.

        Returns:
            torch.device: Primary device
        """
        return self._device

    @property
    def device_name(self) -> str:
        """Get device name.

        Returns:
            str: Device type name
        """
        return str(self._device)

    @property
    def is_cuda(self) -> bool:
        """Check if CUDA is available.

        Returns:
            bool: True if CUDA available
        """
        return self._device.type == "cuda"

    @property
    def is_mps(self) -> bool:
        """Check if MPS is available.

        Returns:
            bool: True if MPS available
        """
        return self._device.type == "mps"

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU.

        Returns:
            bool: True if using CPU
        """
        return self._device.type == "cpu"

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available.

        Returns:
            bool: True if CUDA available
        """
        return self._cuda_available

    @property
    def device_count(self) -> int:
        """Get number of available devices.

        Returns:
            int: Number of CUDA devices or 1 for CPU/MPS
        """
        if self.is_cuda:
            return self._cuda_device_count
        return 1

    def get_memory_usage(self) -> dict:
        """Get current memory usage.

        Returns:
            dict: Memory usage statistics
        """
        if self.is_cuda:
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "cached_mb": torch.cuda.memory_cached() / 1024 / 1024 if hasattr(torch.cuda, "memory_cached") else 0,
            }
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "cached_mb": 0.0}

    def clear_cache(self) -> None:
        """Clear device cache."""
        if self.is_cuda:
            torch.cuda.empty_cache()


_device_manager: DeviceManager | None = None


def get_device() -> torch.device:
    """Get the primary device (singleton).

    Returns:
        torch.device: Primary device
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager.device
