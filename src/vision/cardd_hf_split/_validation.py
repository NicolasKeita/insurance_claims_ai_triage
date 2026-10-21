"""Validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vision.cardd_hf import HfDamageDataset

from vision.cardd_hf_split._constants import SPLITS, SUPPORTED_SPLIT_ALGORITHMS
from vision.cardd_hf_split._identity import _counts, _digest, _parameters
from vision.cardd_hf_split._metadata import _split_metadata
from vision.cardd_hf_split.grouping._source import _source_context
from vision.dataset import DatasetValidationError


def validate_cardd_hf_split(
    dataset: HfDamageDataset, manifest: dict[str, Any], *, ratios: dict[str, float] | None = None, seed: int = 42
) -> dict[str, Any]:
    """Validate frozen assignments against current source, annotations and content.

    A mismatch is an error; this function never repairs or regenerates a split.
    Relocating the image/cache root preserves the portable source fingerprint.
    """
    ratios = _parameters(ratios, seed)
    context, groups = _source_context(dataset)
    if not isinstance(manifest, dict):
        raise DatasetValidationError("Frozen HF split manifest must contain a JSON object")
    version = manifest.get("split_version")
    algorithm = SUPPORTED_SPLIT_ALGORITHMS.get(version) if isinstance(version, str) else None
    if algorithm is None or manifest.get("algorithm") != algorithm:
        raise DatasetValidationError("Frozen HF split manifest has an unsupported version or inconsistent algorithm")
    expected = dict(
        context,
        split_version=version,
        algorithm=algorithm,
        seed=seed,
        ratios=ratios,
        target_counts=_counts(context["sample_count"], ratios, nonempty=True),
        test_split_policy="frozen held-out: never used for training, checkpoint selection or threshold tuning",
    )
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()):
        raise DatasetValidationError(
            "Frozen HF split manifest is inconsistent with the source revision, samples, annotations, taxonomy "
            "or requested parameters. Do not overwrite or regenerate this version; review the change and "
            "use a new split version."
        )
    assignments = _validate_assignments(manifest, context, groups, expected["target_counts"])
    metadata = _split_metadata(context, assignments, groups)
    if any(manifest.get(key) != value for key, value in metadata.items()):
        raise DatasetValidationError(
            "Frozen HF split class distribution or annotation hashes disagree with assigned samples"
        )
    _validate_split_fingerprint(manifest)
    return manifest


def _validate_assignments(manifest, context, groups, target_counts):
    assignments = manifest.get("sample_ids")
    if not isinstance(assignments, dict) or set(assignments) != set(SPLITS):
        raise DatasetValidationError("Frozen HF split sample_ids must define exactly train, val and test")
    seen, split_for_id = set(), {}
    for name in SPLITS:
        ids = assignments[name]
        if (
            not isinstance(ids, list)
            or any(not isinstance(item, str) for item in ids)
            or len(set(ids)) != len(ids)
            or seen.intersection(ids)
        ):
            raise DatasetValidationError("Frozen HF split leakage: duplicate or invalid sample IDs")
        if len(ids) != target_counts[name]:
            raise DatasetValidationError("Frozen HF split image counts disagree with requested ratios")
        seen.update(ids)
        split_for_id.update({sample_id: name for sample_id in ids})
    if seen != {row["sample_id"] for row in context["samples"]}:
        raise DatasetValidationError("Frozen HF split contains unknown or missing source samples")
    for group in groups:
        if len({split_for_id[sample_id] for sample_id in group.ids}) != 1:
            raise DatasetValidationError(
                "Frozen HF split leakage: identical image content or filepath appears in multiple splits"
            )
    return assignments


def _validate_split_fingerprint(manifest):
    fingerprint = _digest(
        {
            "source_fingerprint": manifest["source_fingerprint"],
            "sample_ids": manifest["sample_ids"],
            "split_version": manifest["split_version"],
            "algorithm": manifest["algorithm"],
            "seed": manifest["seed"],
            "ratios": manifest["ratios"],
        }
    )
    if manifest.get("dataset_fingerprint") != fingerprint:
        raise DatasetValidationError("Frozen HF split dataset fingerprint is inconsistent")
