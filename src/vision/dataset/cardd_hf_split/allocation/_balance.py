"""Balance."""

from __future__ import annotations

import heapq
from collections import defaultdict

from vision.dataset.cardd_hf_split._constants import SPLITS
from vision.dataset.cardd_hf_split._identity import _Group


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
    buckets, actual, patterns = _balance_buckets(groups, assignments, classes, tie_keys)
    totals = [sum(actual[name][index] for name in SPLITS) for index in range(len(classes))]
    weights = [1 / max(total, 1) for total in totals]
    residual = {
        name: [actual[name][index] - desired[label][name] for index, label in enumerate(classes)] for name in SPLITS
    }
    pattern_pairs = _pattern_pairs(patterns, weights)
    _improve_balance(buckets, residual, pattern_pairs)
    return {
        name: sorted(
            sample_id for values in buckets[name].values() for _, _, group in values for sample_id in group.ids
        )
        for name in SPLITS
    }


def _pattern_pairs(patterns, weights):
    pattern_pairs = []
    for first in sorted(patterns):
        for second in sorted(patterns):
            if first[0] != second[0] or first == second:
                continue
            delta = tuple(b - a for a, b in zip(first[1], second[1]))
            weighted = tuple(weight * change for weight, change in zip(weights, delta))
            quadratic = sum(change * scaled for change, scaled in zip(delta, weighted))
            pattern_pairs.append((first, second, delta, weighted, quadratic))
    return pattern_pairs


def _improve_balance(buckets, residual, pattern_pairs):
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


def _balance_buckets(groups, assignments, classes, tie_keys):
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
    return buckets, actual, patterns
