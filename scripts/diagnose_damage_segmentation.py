"""Diagnose an existing Damage checkpoint on VAL, without retraining or using TEST."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from vision.damage.training import DEFAULT_TRAINING_CONFIG, load_training_config, observe_environment
from vision.damage.validation import ValidationParams, prepare_validation, run_validation


def _load_model(checkpoint: str):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install vision dependencies with python -m pip install -e '.[vision]'") from error
    return YOLO(checkpoint)


def _environment():
    try:
        import torch

        import ultralytics
    except ImportError as error:
        raise RuntimeError("Install vision dependencies with python -m pip install -e '.[vision]'") from error
    observed = observe_environment(torch, ultralytics)
    print(json.dumps(observed, indent=2))
    return observed


def _arguments():
    parser = argparse.ArgumentParser(description="VAL-only AP/PR plots and separate fixed-threshold box/mask counts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument(
        "--output-dir", type=Path, help="New directory outside the training run; existing output is refused"
    )
    parser.add_argument("--device", help="Override configured device, e.g. cpu or 0")
    parser.add_argument("--imgsz", type=int, help="Evaluate the existing checkpoint at this size without retraining")
    parser.add_argument("--batch", type=int, help="Override diagnostic batch size; use 4 at 960 px on an 8 GiB GPU")
    parser.add_argument(
        "--operating-confidence",
        type=float,
        help="Fixed VAL operating point, otherwise DAMAGE_CONFIDENCE_THRESHOLD or 0.25",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify data/checkpoint provenance without importing torch, inference or writing files",
    )
    return parser, parser.parse_args()


def main() -> int:
    parser, args = _arguments()
    try:
        config = load_training_config(args.config)
        if args.device is not None:
            config = replace(config, device=args.device)
        params = ValidationParams(
            model_factory=_load_model,
            environment_provider=_environment,
            operating_confidence=args.operating_confidence,
            image_size=args.imgsz,
            batch_size=args.batch,
        )
        if args.check_only:
            request = prepare_validation(config, output_dir=args.output_dir, params=params)
            print(json.dumps({"status": "ready", **request.payload()}, indent=2))
        else:
            result = run_validation(config, output_dir=args.output_dir, params=params)
            print(json.dumps({"mask": result["mask"], "operating_point": result["operating_point"]}, indent=2))
            print(f"VAL diagnostics: {Path(result['output_dir']) / 'val_metrics.json'}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"VAL diagnostics not executed/completed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
