"""Vehicle-part baseline using the step-29 segmentation orchestration."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomllib

from vision.carparts_dataset import carparts_provenance, load_carparts_dataset
from vision.training import REPO_ROOT, DamageTrainingConfig, run_training

DEFAULT_PARTS_CONFIG = REPO_ROOT / "configs/vision/carparts_yolo26n_seg_v1.toml"


@dataclass(frozen=True)
class VehiclePartTrainingConfig(DamageTrainingConfig):
    confidence_threshold: float = 0.25

    def __post_init__(self) -> None:
        super().__post_init__()
        import math

        if Path(self.base_model).name != "yolo26n-seg.pt":
            raise ValueError(
                "Vehicle-part V1 must initialize from independent COCO yolo26n-seg.pt; never damage best.pt"
            )
        if (
            isinstance(self.confidence_threshold, bool)
            or not math.isfinite(self.confidence_threshold)
            or not 0 <= self.confidence_threshold <= 1
        ):
            raise ValueError("Part baseline confidence threshold must be finite and between zero and one")


def load_parts_training_config(path: Path = DEFAULT_PARTS_CONFIG) -> VehiclePartTrainingConfig:
    with Path(path).open("rb") as stream:
        payload = tomllib.load(stream)
    unknown = set(payload) - {field.name for field in fields(VehiclePartTrainingConfig)}
    if unknown:
        raise ValueError(f"Unknown part training keys: {sorted(unknown)}")
    payload["dataset_yaml"] = os.environ.get("PART_DATASET_YAML", payload["dataset_yaml"])
    for key in ("dataset_yaml", "output_dir"):
        value = Path(payload[key]).expanduser()
        payload[key] = (REPO_ROOT / value).resolve() if not value.is_absolute() else value.resolve()
    return VehiclePartTrainingConfig(**payload)


def require_requested_device(config: VehiclePartTrainingConfig, environment: dict[str, Any]) -> dict[str, Any]:
    """Fail explicitly if this exact interpreter cannot honor a CUDA request."""
    requested = config.device.strip().lower()
    gpu_requested = requested not in {"cpu", "mps"}
    if gpu_requested and not environment.get("cuda_available"):
        raise RuntimeError(
            "GPU training/evaluation requested but torch.cuda.is_available() is False in "
            f"{environment.get('python_executable')}; torch={environment.get('torch_version')}, "
            f"CUDA build={environment.get('cuda_runtime_version')}. Select a CUDA-enabled Python; CPU fallback is forbidden."
        )
    if gpu_requested and requested != "0":
        raise ValueError("Carparts V1 CUDA baseline requires explicit device=0")
    return environment


def run_parts_training(
    config: VehiclePartTrainingConfig,
    *,
    model_factory: Callable[[str], Any],
    environment_provider: Callable[[], dict[str, Any]],
    resume: bool = False,
    dataset_loader: Callable[[Path], dict[str, Any]] = load_carparts_dataset,
    tracking_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    from vision.parts_tracking import parts_tracking_run

    def observed_model(checkpoint: str) -> Any:
        model = model_factory(checkpoint)

        def record_gpu_runtime(trainer: Any) -> None:
            epoch = getattr(trainer, "epoch", -1)
            if epoch not in {-1, 0} and (int(epoch) + 1) % 10:
                return
            device = str(getattr(trainer, "device", "unobserved"))
            if config.device == "0" and not device.startswith("cuda"):
                raise RuntimeError(f"GPU baseline selected unexpected trainer device: {device}")
            path = config.output_dir / "gpu_runtime_evidence.txt"
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False)
                evidence = (
                    result.stdout if result.returncode == 0 else f"nvidia-smi unavailable (exit {result.returncode})"
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                evidence = f"nvidia-smi unavailable: {type(exc).__name__}"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"\nUTC={datetime.now(timezone.utc).isoformat()} Python PID={os.getpid()} trainer.device={device} epoch={int(epoch) + 1}\n{evidence}\n"
                )

        # Recording is explicit training-only and never runs during imports/tests.
        if config.device == "0":
            model.add_callback("on_train_start", record_gpu_runtime)
            model.add_callback("on_train_epoch_end", record_gpu_runtime)
        return model

    return run_training(
        config,
        model_factory=observed_model,
        environment_provider=lambda: require_requested_device(config, environment_provider()),
        dataset_loader=dataset_loader,
        resume=resume,
        provenance_provider=carparts_provenance,
        confidence_threshold=config.confidence_threshold,
        tracking_factory=tracking_factory or parts_tracking_run,
    )
