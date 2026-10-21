"""Identity."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from vision.cardd_hf_split._constants import DEFAULT_RATIOS, SPLITS
from vision.dataset import DatasetValidationError


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _parameters(ratios: dict[str, float] | None, seed: int) -> dict[str, float]:
    values = DEFAULT_RATIOS if ratios is None else ratios
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise DatasetValidationError("HF split seed must be an integer")
    if not isinstance(values, dict) or set(values) != set(SPLITS):
        raise DatasetValidationError("HF split ratios must define exactly train, val and test")
    result = {}
    for split in SPLITS:
        value = values[split]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise DatasetValidationError("HF split ratios must be finite, positive numbers")
        result[split] = float(value)
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0, abs_tol=1e-10):
        raise DatasetValidationError("HF split ratios must sum to 1")
    return result


def _counts(total: int, ratios: dict[str, float], *, nonempty: bool = False) -> dict[str, int]:
    """Largest remainders, with stable train/val/test ordering for tied fractions."""
    raw = {name: total * ratios[name] for name in SPLITS}
    counts = {name: math.floor(raw[name]) for name in SPLITS}
    order = sorted(SPLITS, key=lambda name: (-(raw[name] - counts[name]), SPLITS.index(name)))
    for name in order[: total - sum(counts.values())]:
        counts[name] += 1
    if nonempty and total >= len(SPLITS):
        for name in SPLITS:
            if not counts[name]:
                donor = max(
                    SPLITS, key=lambda candidate: (counts[candidate], ratios[candidate], -SPLITS.index(candidate))
                )
                counts[donor] -= 1
                counts[name] = 1
    return counts


@dataclass(frozen=True)
class _Group:
    ids: tuple[str, ...]
    image_counts: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.ids)
