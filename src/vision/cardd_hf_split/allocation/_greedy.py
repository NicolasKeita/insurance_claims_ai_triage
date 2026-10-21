"""Allocation."""

from __future__ import annotations

from collections import Counter

from vision.cardd_hf_split._constants import SPLITS
from vision.cardd_hf_split._identity import _counts, _digest, _Group
from vision.cardd_hf_split.allocation._balance import _balance_groups
from vision.cardd_hf_split.allocation._packing import _packing_exists
from vision.dataset import DatasetValidationError


class _Allocation:
    """Exact split capacities and class deficits during grouped greedy allocation."""

    def __init__(self, groups, classes, targets, ratios, seed):
        total_per_class = Counter(
            label for group in groups for label, count in group.image_counts.items() for _ in range(count)
        )
        desired = {label: _counts(total_per_class[label], ratios, nonempty=True) for label in classes}
        actual = {name: Counter() for name in SPLITS}
        capacity = dict(targets)
        output = {name: [] for name in SPLITS}
        tie_keys = {group.ids: _digest([seed, group.ids]) for group in groups}

        self.classes = classes
        self.targets = targets
        self.total_per_class = total_per_class
        self.desired = desired
        self.actual = actual
        self.capacity = capacity
        self.output = output
        self.tie_keys = tie_keys

    def ordered_groups(self, groups):
        duplicates = sorted(
            (group for group in groups if group.size > 1),
            key=lambda group: (
                -group.size,
                min((self.total_per_class[label] for label in group.image_counts), default=0),
                self.tie_keys[group.ids],
            ),
        )
        singles = [group for group in groups if group.size == 1]
        self.single_count = len(singles)
        return duplicates, singles

    def put(self, group: _Group, primary: str | None, remaining_duplicate_sizes: tuple[int, ...] = ()) -> None:
        candidates = []
        for name in SPLITS:
            if self.capacity[name] < group.size:
                continue
            after = tuple(self.capacity[split] - (group.size if split == name else 0) for split in SPLITS)
            if remaining_duplicate_sizes and not _packing_exists(remaining_duplicate_sizes, after, self.single_count):
                continue
            primary_deficit = self.desired[primary][name] - self.actual[name][primary] if primary else 0
            other_deficit = sum(
                (self.desired[label][name] - self.actual[name][label]) / max(self.total_per_class[label], 1)
                for label in group.image_counts
            )
            candidates.append(
                (
                    (
                        primary_deficit,
                        other_deficit,
                        self.capacity[name] / max(self.targets[name], 1),
                        self.capacity[name],
                        -SPLITS.index(name),
                    ),
                    name,
                )
            )
        if not candidates:
            raise DatasetValidationError("Duplicate groups prevent a leakage-free split with the requested sizes")
        name = max(candidates)[1]
        self.output[name].extend(group.ids)
        self.capacity[name] -= group.size
        self.actual[name].update(group.image_counts)

    def assign(self, duplicates, singles):
        for index, group in enumerate(duplicates):
            primary = (
                min(group.image_counts, key=lambda label: (self.total_per_class[label], label))
                if group.image_counts
                else None
            )
            self.put(group, primary, tuple(item.size for item in duplicates[index + 1 :]))
        remaining = {group.ids[0]: group for group in singles}
        while remaining:
            label_groups = {
                label: [group for group in remaining.values() if label in group.image_counts] for label in self.classes
            }
            nonempty = [label for label in self.classes if label_groups[label]]
            if nonempty:
                primary = min(nonempty, key=lambda label: (len(label_groups[label]), label))
                group = min(label_groups[primary], key=lambda item: (-len(item.image_counts), self.tie_keys[item.ids]))
            else:
                primary = None
                group = min(remaining.values(), key=lambda item: self.tie_keys[item.ids])
            remaining.pop(group.ids[0])
            self.put(group, primary)


def _assign(groups, classes, targets, ratios, seed):
    """Seeded grouped allocation followed by equal-size class balancing."""
    allocation = _Allocation(groups, classes, targets, ratios, seed)
    duplicates, singles = allocation.ordered_groups(groups)
    if not _packing_exists(tuple(group.size for group in duplicates), tuple(targets.values()), len(singles)):
        raise DatasetValidationError(
            "Duplicate image groups cannot fit the requested exact train/val/test sizes without data leakage"
        )
    allocation.assign(duplicates, singles)
    if any(allocation.capacity.values()):
        raise DatasetValidationError("Internal HF split allocation did not fill exact split sizes")
    return _balance_groups(groups, allocation.output, classes, allocation.desired, allocation.tie_keys)
