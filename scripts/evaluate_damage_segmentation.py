"""Evaluate the exact VAL-selected checkpoint on held-out TEST only."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from vision.damage.evaluation import EvaluationParams, run_evaluation
from vision.damage.training import DEFAULT_TRAINING_CONFIG, load_training_config, observe_environment


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate only the completed training run's best.pt on TEST.")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--device", help="Override config device, e.g. cpu or 0")
    parser.add_argument("--error-examples", type=int, default=8, help="Maximum annotated diagnostic examples; 0 disables analysis")
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

        metrics = run_evaluation(
            config,
            params=EvaluationParams(
                model_factory=load_model, environment_provider=environment, max_error_examples=args.error_examples
            ),
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Evaluation not executed/completed: {error}\n")
    print(json.dumps({"box": metrics["box"], "mask": metrics["mask"]}, indent=2))
    print(f"Held-out metrics: {config.output_dir / 'test_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
