"""Small, opt-in safeguards for sequential real vision evaluation."""

from evaluation.real.safety.cuda import CudaMemory
from evaluation.real.safety.telemetry import SafetyMonitor, VisionSafetyStop

__all__ = ["CudaMemory", "SafetyMonitor", "VisionSafetyStop"]


def create_vision_runtime(config):
    """Construct an explicit device and its monitor before loading any weights."""
    from evaluation.real.safety.runtime import VisionRuntime

    device = config.vision_device
    if device is None:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    monitor = SafetyMonitor(
        config.fixture_dir.parent,
        max_temperature_c=config.vision_max_temperature_c,
        gpu_enabled=device != "cpu",
    )
    return VisionRuntime(monitor, device=device, imgsz=config.vision_image_size)
