"""GPU runtime evidence recorded only for an explicit device=0 baseline."""

import os
import subprocess
from datetime import datetime, timezone
from functools import partial
from typing import Any

from vision.parts_training._config import VehiclePartTrainingConfig


def record_gpu_runtime(config: VehiclePartTrainingConfig, trainer: Any) -> None:
    """Append the observed trainer device and nvidia-smi output, at most every tenth epoch."""
    epoch = getattr(trainer, "epoch", -1)
    if epoch not in {-1, 0} and (int(epoch) + 1) % 10:
        return
    device = str(getattr(trainer, "device", "unobserved"))
    if config.device == "0" and not device.startswith("cuda"):
        raise RuntimeError(f"GPU baseline selected unexpected trainer device: {device}")
    path = config.output_dir / "gpu_runtime_evidence.txt"
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False)
        evidence = result.stdout if result.returncode == 0 else f"nvidia-smi unavailable (exit {result.returncode})"
    except (OSError, subprocess.TimeoutExpired) as exc:
        evidence = f"nvidia-smi unavailable: {type(exc).__name__}"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\nUTC={datetime.now(timezone.utc).isoformat()} Python PID={os.getpid()} "
            f"trainer.device={device} epoch={int(epoch) + 1}\n{evidence}\n"
        )


def observed_model(config: VehiclePartTrainingConfig, factory: Any, checkpoint: str) -> Any:
    """Recording is explicit training-only and never runs during imports/tests."""
    model = factory(checkpoint)
    if config.device == "0":
        model.add_callback("on_train_start", partial(record_gpu_runtime, config))
        model.add_callback("on_train_epoch_end", partial(record_gpu_runtime, config))
    return model
