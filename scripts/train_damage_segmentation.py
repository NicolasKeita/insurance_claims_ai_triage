"""Explicit pretrained segmentation fine-tuning; no model loaded at import."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from vision.training import DEFAULT_TRAINING_CONFIG, load_training_config, observe_environment, run_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26 segmentation on prepared CarDD TRAIN/VAL; TEST is held out.")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--device", help="Override config device, e.g. cpu or 0 for CUDA device 0")
    args = parser.parse_args()
    try:
        config = load_training_config(args.config)
        if args.device is not None:
            config = replace(config, device=args.device)

        def load_model(checkpoint: str):
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError("Install vision dependencies with python -m pip install -e '.[vision]'") from error
            return YOLO(checkpoint)

        def environment():
            try:
                import torch
                import ultralytics
            except ImportError as error:
                raise RuntimeError("Install vision dependencies with python -m pip install -e '.[vision]'") from error
            observed = observe_environment(torch, ultralytics)
            print(json.dumps(observed, indent=2))
            print(f"Requested device: {config.device}")
            return observed

        summary = run_training(config, model_factory=load_model, environment_provider=environment)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Training not executed/completed: {error}\n")
    print(f"Selected best checkpoint: {config.output_dir / summary['best_checkpoint']}")
    print(f"Training summary: {config.output_dir / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
