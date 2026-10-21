"""Held-out checkpoint selection with full provenance verification."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vision.damage.training import DamageTrainingConfig, dataset_provenance, file_sha256


def selected_checkpoint(
    config: DamageTrainingConfig,
    manifest: dict[str, Any],
    provenance_provider: Callable[[Path, dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    summary_path = config.output_dir / "training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"No completed training summary: {summary_path}. Train the baseline first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("checkpoint_selection_split") != "val":
        raise ValueError("Evaluation requires a completed run whose checkpoint was selected on VAL")
    if summary.get("task") != "segment" or summary.get("config", {}).get("version") != config.version:
        raise ValueError("Training task/version does not match this evaluation configuration")
    if summary.get("best_checkpoint") != "train/weights/best.pt":
        raise ValueError("Only the training-selected train/weights/best.pt may be evaluated")
    checkpoint = config.output_dir / summary["best_checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint absent: {checkpoint}")
    if file_sha256(checkpoint) != summary.get("best_checkpoint_sha256"):
        raise ValueError("Selected checkpoint SHA-256 differs from the completed training run")
    provenance_provider = provenance_provider or dataset_provenance
    if provenance_provider(config.dataset_yaml, manifest) != summary.get("dataset"):
        raise ValueError("Prepared dataset provenance differs from the training run; evaluation refused")
    return checkpoint, summary
