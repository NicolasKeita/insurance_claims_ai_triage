"""Shared configuration and fixed dataset contracts."""

from __future__ import annotations

from pathlib import Path

SPLITS = ("train", "val", "test")

SPLIT_VERSION = "cardd_hf_split_v2"

DEFAULT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

DEFAULT_SPLIT_MANIFEST_PATH = Path(__file__).resolve().parents[4] / "artifacts" / "vision" / "cardd_hf_split_v2.json"

ALGORITHM = "deterministic_multilabel_greedy_grouped_pair_balance_v2"

SUPPORTED_SPLIT_ALGORITHMS = {
    "cardd_hf_split_v1": "deterministic_multilabel_greedy_grouped_v1",
    SPLIT_VERSION: ALGORITHM,
}
