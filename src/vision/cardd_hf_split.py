"""Frozen, portable, leakage-checked splits for the CarDD Hugging Face mirror.

This is a project split, not an official CarDD benchmark split.  The small
stratifier uses image-level multilabels and keeps byte-identical images together.
No FiftyOne, Hugging Face or training framework is imported by this module.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vision.dataset import DatasetValidationError, annotation_hash

if TYPE_CHECKING:
    from vision.cardd_hf import HfDamageDataset


SPLITS = ("train", "val", "test")
SPLIT_VERSION = "cardd_hf_split_v2"
DEFAULT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
DEFAULT_SPLIT_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "vision" / "cardd_hf_split_v2.json"
ALGORITHM = "deterministic_multilabel_greedy_grouped_pair_balance_v2"
SUPPORTED_SPLIT_ALGORITHMS = {
    "cardd_hf_split_v1": "deterministic_multilabel_greedy_grouped_v1",
    SPLIT_VERSION: ALGORITHM,
}


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


def _source_context(dataset: HfDamageDataset) -> tuple[dict[str, Any], list[_Group]]:
    if not isinstance(dataset.repository, str) or not dataset.repository.strip():
        raise DatasetValidationError("HF mirror repository must be identified")
    if dataset.revision is not None and (not isinstance(dataset.revision, str) or not dataset.revision.strip()):
        raise DatasetValidationError("HF mirror revision must be a nonempty string or unavailable (null)")
    classes = tuple(dataset.classes)
    if (
        not classes
        or len(set(classes)) != len(classes)
        or any(not isinstance(name, str) or not name.strip() for name in classes)
    ):
        raise DatasetValidationError("HF mirror classes must be unique, nonempty labels")
    samples = sorted(dataset.samples, key=lambda sample: sample.sample_id)
    if len(samples) < 3:
        raise DatasetValidationError("At least three HF mirror samples are needed for train/val/test")
    rows, seen_ids = [], set()
    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    paths, hashes, names = defaultdict(list), defaultdict(list), defaultdict(list)
    sample_labels = {}
    for index, sample in enumerate(samples):
        sample_id = sample.sample_id
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen_ids:
            raise DatasetValidationError("HF mirror has an empty or duplicate sample ID")
        seen_ids.add(sample_id)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (sample.width, sample.height)
        ):
            raise DatasetValidationError(f"HF mirror sample {sample_id}: invalid dimensions")
        path = Path(sample.filepath).expanduser().resolve()
        if not path.is_file():
            raise DatasetValidationError(f"HF mirror sample {sample_id}: image file is missing")
        if (
            not isinstance(sample.image_sha256, str)
            or len(sample.image_sha256) != 64
            or any(c not in "0123456789abcdef" for c in sample.image_sha256)
        ):
            raise DatasetValidationError(f"HF mirror sample {sample_id}: invalid image SHA256")
        if annotation_hash(path) != sample.image_sha256:
            raise DatasetValidationError(f"HF mirror sample {sample_id}: image content changed after inspection")
        annotations = []
        labels = set()
        for instance in sample.instances:
            if instance.label not in classes:
                raise DatasetValidationError(f"HF mirror sample {sample_id}: annotation uses an unknown class")
            labels.add(instance.label)
            annotations.append(
                {
                    "label": instance.label,
                    "polygons": instance.polygons,
                    "annotation_field": getattr(instance, "annotation_field", None),
                    "source_kind": getattr(instance, "source_kind", None),
                }
            )
        try:
            annotation_fingerprint = _digest(annotations)
        except (ValueError, TypeError) as exc:
            raise DatasetValidationError(f"HF mirror sample {sample_id}: annotations cannot be fingerprinted") from exc
        rows.append(
            {
                "sample_id": sample_id,
                "file_name": path.name,
                "image_sha256": sample.image_sha256,
                "width": sample.width,
                "height": sample.height,
                "annotation_sha256": annotation_fingerprint,
                "annotation_count": len(annotations),
                "annotations_per_category": dict(sorted(Counter(item["label"] for item in annotations).items())),
            }
        )
        sample_labels[sample_id] = labels
        paths[str(path).casefold()].append(index)
        hashes[sample.image_sha256].append(index)
        names[path.name.casefold()].append(index)
    for indices in (*paths.values(), *hashes.values()):
        for index in indices[1:]:
            union(indices[0], index)
    members = defaultdict(list)
    for index, sample in enumerate(samples):
        members[find(index)].append(sample.sample_id)
    groups = []
    for ids in sorted(members.values()):
        counts = Counter(label for sample_id in ids for label in sample_labels[sample_id])
        groups.append(_Group(tuple(ids), {label: counts[label] for label in classes if counts[label]}))
    category_mapping = [
        {"category_id": index + 1, "name": name, "yolo_class_id": index} for index, name in enumerate(classes)
    ]
    context = {
        "source_name": f"CarDD HF mirror ({dataset.repository})",
        "source_repo": dataset.repository,
        "source_revision": dataset.revision,
        "source_format": getattr(dataset, "source_format", "FiftyOneDataset"),
        "sample_count": len(samples),
        "official_splits_used": False,
        "annotation_fields": list(dataset.annotation_fields),
        "field_schema": dict(dataset.field_schema),
        "category_mapping": category_mapping,
        "samples": rows,
        "annotation_fingerprint": _digest(
            [{key: value for key, value in row.items() if key != "file_name"} for row in rows]
        ),
        "duplicate_grouping": {
            "policy": "same_resolved_filepath_or_identical_image_sha256_together",
            "group_count": len(groups),
            "duplicate_group_count": sum(group.size > 1 for group in groups),
            "samples_in_duplicate_groups": sum(group.size for group in groups if group.size > 1),
            "duplicate_filepath_groups": sum(len(indices) > 1 for indices in paths.values()),
            "duplicate_image_sha256_groups": sum(len(indices) > 1 for indices in hashes.values()),
            "duplicate_filename_groups": sum(len(indices) > 1 for indices in names.values()),
            "groups": [list(group.ids) for group in groups if group.size > 1],
        },
    }
    context["source_fingerprint"] = _digest(context)
    return context, groups


@lru_cache(maxsize=10000)
def _packing_exists(sizes: tuple[int, ...], capacities: tuple[int, ...], singles: int) -> bool:
    """Keep duplicate groups intact while retaining exact split capacities.

    Greedy packing usually proves feasibility immediately. A bounded search
    resolves small difficult cases; it never relaxes leakage or split counts.
    """
    sizes = tuple(sorted(sizes, reverse=True))
    capacities = tuple(sorted(capacities, reverse=True))
    if any(value < 0 for value in capacities) or sum(sizes) + singles != sum(capacities):
        return False
    if not sizes:
        return True
    if sizes[0] > capacities[0]:
        return False
    if singles == 0:
        divisor = math.gcd(*sizes)
        if any(capacity % divisor for capacity in capacities):
            return False
    free = list(capacities)
    for size in sizes:
        free.sort(reverse=True)
        if free[0] < size:
            break
        free[0] -= size
    else:
        return True
    if len(sizes) > 100:
        raise DatasetValidationError(
            "Duplicate image groups require complex packing; cannot safely preserve exact split sizes. Use a new reviewed split version or remove duplicate samples explicitly."
        )
    visited = 0

    @lru_cache(maxsize=50000)
    def search(index: int, free: tuple[int, ...]) -> bool:
        nonlocal visited
        visited += 1
        if visited > 50000:
            raise DatasetValidationError(
                "Duplicate image grouping exceeded safe split search limits; no manifest was created"
            )
        if index == len(sizes):
            return True
        previous = None
        for position, capacity in enumerate(free):
            if capacity == previous or capacity < sizes[index]:
                continue
            previous = capacity
            remaining = list(free)
            remaining[position] -= sizes[index]
            if search(index + 1, tuple(sorted(remaining, reverse=True))):
                return True
        return False

    return search(0, capacities)


def _balance_groups(
    groups: list[_Group],
    assignments: dict[str, list[str]],
    classes: list[str],
    desired: dict[str, dict[str, int]],
    tie_keys: dict[tuple[str, ...], str],
) -> dict[str, list[str]]:
    """Improve all label distributions with deterministic equal-size swaps.

    Rare-label-first allocation can leave a common co-occurring class heavily
    concentrated in train. Swapping whole, equally sized image groups preserves
    exact capacities and duplicate isolation. Search label-pattern pairs rather
    than all image pairs, minimizing sum(squared residual / class image count).
    This uses annotation incidence only, before training or model evaluation.

    At six multilabel classes, singleton incidence has at most 64 patterns. Duplicate groups contribute their
    actual image-count vectors separately.
    """
    split_for_id = {sample_id: name for name in SPLITS for sample_id in assignments[name]}
    buckets = {name: defaultdict(list) for name in SPLITS}
    actual = {name: [0] * len(classes) for name in SPLITS}
    patterns = set()
    for group in groups:
        name = split_for_id[group.ids[0]]
        vector = tuple(group.image_counts.get(label, 0) for label in classes)
        pattern = (group.size, vector)
        patterns.add(pattern)
        heapq.heappush(buckets[name][pattern], (tie_keys[group.ids], group.ids, group))
        actual[name] = [old + count for old, count in zip(actual[name], vector)]
    totals = [sum(actual[name][index] for name in SPLITS) for index in range(len(classes))]
    weights = [1 / max(total, 1) for total in totals]
    residual = {
        name: [actual[name][index] - desired[label][name] for index, label in enumerate(classes)] for name in SPLITS
    }
    pattern_pairs = []
    for first in sorted(patterns):
        for second in sorted(patterns):
            if first[0] != second[0] or first == second:
                continue
            delta = tuple(b - a for a, b in zip(first[1], second[1]))
            weighted = tuple(weight * change for weight, change in zip(weights, delta))
            quadratic = sum(change * scaled for change, scaled in zip(delta, weighted))
            pattern_pairs.append((first, second, delta, weighted, quadratic))
    split_pairs = [(SPLITS[i], SPLITS[j]) for i in range(len(SPLITS)) for j in range(i + 1, len(SPLITS))]
    while True:
        best = None
        best_key = None
        for left, right in split_pairs:
            difference = [a - b for a, b in zip(residual[left], residual[right])]
            for first, second, delta, weighted, quadratic in pattern_pairs:
                if not buckets[left][first] or not buckets[right][second]:
                    continue
                gain = -2 * (sum(scaled * value for scaled, value in zip(weighted, difference)) + quadratic)
                if gain <= 1e-12:
                    continue
                key = (
                    -gain,
                    SPLITS.index(left),
                    SPLITS.index(right),
                    buckets[left][first][0][0],
                    buckets[right][second][0][0],
                    first,
                    second,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = (left, right, first, second, delta)
        if best is None:
            break
        left, right, first, second, delta = best
        first_group = heapq.heappop(buckets[left][first])
        second_group = heapq.heappop(buckets[right][second])
        heapq.heappush(buckets[right][first], first_group)
        heapq.heappush(buckets[left][second], second_group)
        residual[left] = [value + change for value, change in zip(residual[left], delta)]
        residual[right] = [value - change for value, change in zip(residual[right], delta)]
    return {
        name: sorted(
            sample_id for values in buckets[name].values() for _, _, group in values for sample_id in group.ids
        )
        for name in SPLITS
    }


def _assign(
    groups: list[_Group], classes: list[str], targets: dict[str, int], ratios: dict[str, float], seed: int
) -> dict[str, list[str]]:
    """
    Compute seeded tie keys once; repeated hashing would dominate a real run.
    """
    total_per_class = Counter(
        label for group in groups for label, count in group.image_counts.items() for _ in range(count)
    )
    desired = {label: _counts(total_per_class[label], ratios, nonempty=True) for label in classes}
    actual = {name: Counter() for name in SPLITS}
    capacity = dict(targets)
    output = {name: [] for name in SPLITS}
    tie_keys = {group.ids: _digest([seed, group.ids]) for group in groups}

    def tie(group):
        return tie_keys[group.ids]

    duplicates = sorted(
        (group for group in groups if group.size > 1),
        key=lambda group: (
            -group.size,
            min((total_per_class[label] for label in group.image_counts), default=0),
            tie(group),
        ),
    )
    singles = [group for group in groups if group.size == 1]
    if not _packing_exists(tuple(group.size for group in duplicates), tuple(targets.values()), len(singles)):
        raise DatasetValidationError(
            "Duplicate image groups cannot fit the requested exact train/val/test sizes without data leakage"
        )

    def put(group: _Group, primary: str | None, remaining_duplicate_sizes: tuple[int, ...] = ()) -> None:
        candidates = []
        for name in SPLITS:
            if capacity[name] < group.size:
                continue
            after = tuple(capacity[split] - (group.size if split == name else 0) for split in SPLITS)
            if remaining_duplicate_sizes and not _packing_exists(remaining_duplicate_sizes, after, len(singles)):
                continue
            primary_deficit = desired[primary][name] - actual[name][primary] if primary else 0
            other_deficit = sum(
                (desired[label][name] - actual[name][label]) / max(total_per_class[label], 1)
                for label in group.image_counts
            )
            candidates.append(
                (
                    (
                        primary_deficit,
                        other_deficit,
                        capacity[name] / max(targets[name], 1),
                        capacity[name],
                        -SPLITS.index(name),
                    ),
                    name,
                )
            )
        if not candidates:
            raise DatasetValidationError("Duplicate groups prevent a leakage-free split with the requested sizes")
        name = max(candidates)[1]
        output[name].extend(group.ids)
        capacity[name] -= group.size
        actual[name].update(group.image_counts)

    for index, group in enumerate(duplicates):
        primary = (
            min(group.image_counts, key=lambda label: (total_per_class[label], label)) if group.image_counts else None
        )
        put(group, primary, tuple(item.size for item in duplicates[index + 1 :]))
    remaining = {group.ids[0]: group for group in singles}
    while remaining:
        label_groups = {
            label: [group for group in remaining.values() if label in group.image_counts] for label in classes
        }
        nonempty = [label for label in classes if label_groups[label]]
        if nonempty:
            primary = min(nonempty, key=lambda label: (len(label_groups[label]), label))
            group = min(label_groups[primary], key=lambda item: (-len(item.image_counts), tie(item)))
        else:
            primary = None
            group = min(remaining.values(), key=tie)
        remaining.pop(group.ids[0])
        put(group, primary)
    if any(capacity.values()):
        raise DatasetValidationError("Internal HF split allocation did not fill exact split sizes")
    return _balance_groups(groups, output, classes, desired, tie_keys)


def _split_metadata(context: dict[str, Any], assignments: dict[str, list[str]], groups: list[_Group]) -> dict[str, Any]:
    rows = {row["sample_id"]: row for row in context["samples"]}
    classes = [item["name"] for item in context["category_mapping"]]
    split_counts, hashes = {}, {}
    for name in SPLITS:
        samples = [rows[sample_id] for sample_id in assignments[name]]
        instances = Counter(
            label for row in samples for label, count in row["annotations_per_category"].items() for _ in range(count)
        )
        images = Counter(label for row in samples for label in row["annotations_per_category"])
        split_counts[name] = {
            "image_count": len(samples),
            "annotation_count": sum(row["annotation_count"] for row in samples),
            "annotations_per_category": {label: instances[label] for label in classes},
            "images_per_category": {label: images[label] for label in classes},
            "images_without_annotations": sum(row["annotation_count"] == 0 for row in samples),
        }
        hashes[name] = _digest([{key: value for key, value in row.items() if key != "file_name"} for row in samples])
    warnings = []
    for label in classes:
        independent = sum(label in group.image_counts for group in groups)
        if independent < len(SPLITS):
            warnings.append(
                f"Class {label!r} has only {independent} independent image groups; representation in all three splits is impossible."
            )
        elif any(not split_counts[name]["images_per_category"][label] for name in SPLITS):
            warnings.append(
                f"Class {label!r} is absent from at least one split: multilabel overlap, duplicate grouping and exact capacities prevented full coverage."
            )
    return {
        "splits": split_counts,
        "class_distribution": {
            name: {
                "images": split_counts[name]["images_per_category"],
                "instances": split_counts[name]["annotations_per_category"],
            }
            for name in SPLITS
        },
        "annotation_hashes": hashes,
        "warnings": warnings,
        "split_overlap_check": "passed: sample IDs, resolved filepaths and image SHA256 are disjoint across splits",
    }


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
            "Frozen HF split manifest is inconsistent with the source revision, samples, annotations, taxonomy or requested parameters. Do not overwrite or regenerate this version; review the change and use a new split version."
        )
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
        if len(ids) != expected["target_counts"][name]:
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
    metadata = _split_metadata(context, assignments, groups)
    if any(manifest.get(key) != value for key, value in metadata.items()):
        raise DatasetValidationError(
            "Frozen HF split class distribution or annotation hashes disagree with assigned samples"
        )
    fingerprint = _digest(
        {
            "source_fingerprint": context["source_fingerprint"],
            "sample_ids": assignments,
            "split_version": version,
            "algorithm": algorithm,
            "seed": seed,
            "ratios": ratios,
        }
    )
    if manifest.get("dataset_fingerprint") != fingerprint:
        raise DatasetValidationError("Frozen HF split dataset fingerprint is inconsistent")
    return manifest


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
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        return create_or_load_cardd_hf_split(dataset, path, ratios=ratios, seed=seed)
    return manifest
