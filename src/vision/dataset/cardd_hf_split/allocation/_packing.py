"""Packing."""

from __future__ import annotations

import math
from functools import lru_cache

from vision.dataset import DatasetValidationError


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
            "Duplicate image groups require complex packing; cannot safely preserve exact split sizes. "
            "Use a new reviewed split version or remove duplicate samples explicitly."
        )
    return _packing_search(sizes, capacities)


def _packing_search(sizes, capacities):
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
