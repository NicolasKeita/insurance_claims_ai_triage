"""Explicit official part baseline with same-interpreter CUDA preflight."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from vision.dataset.carparts_dataset import prepare_carparts_dataset
from vision.parts.training import DEFAULT_PARTS_CONFIG, load_parts_training_config, run_parts_training
from vision.damage.training import observe_environment, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="COCO yolo26n-seg transfer learning on official Carparts-Seg TRAIN/VAL.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PARTS_CONFIG)
    parser.add_argument("--device", help="Explicit device override; GPU requests fail if CUDA is unavailable")
    parser.add_argument("--prepare", action="store_true", help="Explicitly permit official dataset download before training")
    parser.add_argument("--resume", action="store_true", help="Resume validated incomplete last.pt with original total epoch budget")
    parser.add_argument("--evaluate-after-training", action="store_true", help="Run final TEST once after TRAIN/VAL completes")
    args = parser.parse_args()
    try:
        config = load_parts_training_config(args.config)
        if args.device is not None:
            config = replace(config, device=args.device)
        if args.prepare:
            prepare_carparts_dataset(dataset_root=config.dataset_yaml.parent)

        def environment():
            import torch
            import ultralytics
            observed = observe_environment(torch, ultralytics)
            print(json.dumps(observed, indent=2), flush=True)
            print(f"Requested Ultralytics device: {config.device}", flush=True)
            config.output_dir.parent.mkdir(parents=True, exist_ok=True)
            write_json(config.output_dir.parent / "carparts_gpu_preflight.json", observed | {"requested_device": config.device})
            return observed

        def load_model(checkpoint: str):
            from ultralytics import YOLO
            return YOLO(checkpoint)

        summary = run_parts_training(config, model_factory=load_model, environment_provider=environment, resume=args.resume)
        print(f"Selected VAL best.pt: {config.output_dir / summary['best_checkpoint']}", flush=True)
        if args.evaluate_after_training:
            from vision.parts.evaluation import run_parts_evaluation
            metrics = run_parts_evaluation(config, model_factory=load_model, environment_provider=environment)
            print(json.dumps({"box": metrics["box"], "mask": metrics["mask"]}, indent=2), flush=True)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Part training/evaluation FAILED or NOT EXECUTED: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
