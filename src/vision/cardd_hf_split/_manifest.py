"""Manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vision.cardd_hf import HfDamageDataset

from vision.cardd_hf_split._constants import (
    ALGORITHM,
    DEFAULT_SPLIT_MANIFEST_PATH,
    SPLIT_VERSION,
)
from vision.cardd_hf_split._identity import _counts, _digest, _parameters
from vision.cardd_hf_split._metadata import _split_metadata
from vision.cardd_hf_split._validation import validate_cardd_hf_split
from vision.cardd_hf_split.allocation._greedy import _assign
from vision.cardd_hf_split.grouping._source import _source_context
from vision.dataset import DatasetValidationError


def create_or_load_cardd_hf_split(
    dataset: HfDamageDataset,
    manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST_PATH,
    *,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Create once, otherwise reuse and validate the same frozen manifest.

    Exclusive creation also prevents two concurrent jobs replacing a split.
    """
    ratios = _parameters(ratios, seed)
    path = Path(manifest_path).expanduser().resolve()
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DatasetValidationError(
                f"Cannot read frozen HF split manifest; it will not be regenerated: {path}"
            ) from exc
        return validate_cardd_hf_split(dataset, manifest, ratios=ratios, seed=seed)
    manifest = _new_split_manifest(dataset, ratios, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        return create_or_load_cardd_hf_split(dataset, path, ratios=ratios, seed=seed)
    return manifest


def _new_split_manifest(dataset, ratios, seed):
    context, groups = _source_context(dataset)
    targets = _counts(context["sample_count"], ratios, nonempty=True)
    assignments = _assign(groups, [item["name"] for item in context["category_mapping"]], targets, ratios, seed)
    manifest = dict(
        context,
        split_version=SPLIT_VERSION,
        algorithm=ALGORITHM,
        seed=seed,
        ratios=ratios,
        target_counts=targets,
        sample_ids=assignments,
        test_split_policy="frozen held-out: never used for training, checkpoint selection or threshold tuning",
    )
    manifest.update(_split_metadata(context, assignments, groups))
    manifest["dataset_fingerprint"] = _digest(
        {
            "source_fingerprint": context["source_fingerprint"],
            "sample_ids": assignments,
            "split_version": SPLIT_VERSION,
            "algorithm": ALGORITHM,
            "seed": seed,
            "ratios": ratios,
        }
    )
    return manifest
