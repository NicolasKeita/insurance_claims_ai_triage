"""Evaluate exactly the completed VAL-selected best.pt on supplied TEST once."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from vision.parts_evaluation import run_parts_evaluation
from vision.parts_training import DEFAULT_PARTS_CONFIG, load_parts_training_config
from vision.training import observe_environment


def main() -> int:
    parser = argparse.ArgumentParser(description="Final Carparts-Seg TEST, no TEST checkpoint or threshold selection.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PARTS_CONFIG)
    parser.add_argument("--device")
    parser.add_argument("--error-examples", type=int, default=8)
    args = parser.parse_args()
    try:
        config = load_parts_training_config(args.config)
        if args.device is not None:
            config = replace(config, device=args.device)

        def environment():
            import torch
            import ultralytics
            observed = observe_environment(torch, ultralytics)
            print(json.dumps(observed, indent=2), flush=True)
            print(f"Requested Ultralytics device: {config.device}", flush=True)
            return observed

        def load_model(checkpoint: str):
            from ultralytics import YOLO
            return YOLO(checkpoint)

        metrics = run_parts_evaluation(config, model_factory=load_model, environment_provider=environment, max_error_examples=args.error_examples)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Part evaluation FAILED or NOT EXECUTED: {error}\n")
    print(json.dumps({"box": metrics["box"], "mask": metrics["mask"]}, indent=2))
    print(f"Held-out metrics: {config.output_dir / 'test_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
