"""Shared configuration and fixed dataset contracts."""

from __future__ import annotations

import threading

DEFAULT_HF_REPOSITORY = "harpreetsahota/CarDD"

_LOAD_LOCK = threading.RLock()

Point = tuple[float, float]

Polygon = tuple[Point, ...]
