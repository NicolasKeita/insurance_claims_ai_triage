"""Durable stage events and a cooperative NVIDIA temperature guard."""

import math
from pathlib import Path
from threading import Event, Lock
from time import monotonic
from uuid import uuid4

from evaluation.real.progress.gpu import read_gpu
from evaluation.real.safety.journal import SafetyJournal, VisionSafetyStop
from evaluation.real.safety.sampling import GpuSampling

__all__ = ["SafetyMonitor", "VisionSafetyStop"]


class SafetyMonitor(SafetyJournal, GpuSampling):
    """Request a stop between GPU operations, without interrupting active kernels."""

    def __init__(self, output_dir, max_temperature_c=80, interval_seconds=1, gpu_enabled=True, *, reader=None):
        if not math.isfinite(max_temperature_c) or max_temperature_c <= 0:
            raise ValueError("max_temperature_c must be finite and positive")
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and positive")
        self.output_dir = Path(output_dir)
        self.max_temperature_c = max_temperature_c
        self.interval_seconds = interval_seconds
        self.gpu_enabled = gpu_enabled
        self.run_id = uuid4().hex
        self._reader = reader or read_gpu
        self._lock, self._stop = Lock(), Event()
        self._failure = None
        self._events = self._gpu = self._csv = self._thread = None
        self._started = self._closed = False
        self._last_sample = {"available": False, "reason": "Safety monitor has not sampled a GPU"}
        self._last_sample_at = None

    @staticmethod
    def _now():
        return monotonic()

    def start(self):
        if self._closed:
            raise VisionSafetyStop("Safety monitor is already closed")
        if self._started:
            self.check()
            return self
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._events = (self.output_dir / "events.jsonl").open("a", encoding="utf-8")
            if self.gpu_enabled:
                self._open_gpu()
            self._started = True
            self.event(
                "monitor_started",
                gpu_enabled=self.gpu_enabled,
                max_temperature_c=self.max_temperature_c,
                interval_seconds=self.interval_seconds,
            )
            if self.gpu_enabled:
                self._start_sampling()
            return self
        except Exception as error:
            self._latch(f"Safety monitoring could not start: {error}")
            self.close()
            if isinstance(error, VisionSafetyStop):
                raise
            raise VisionSafetyStop(self._failure) from error

    def check(self):
        with self._lock:
            if self._failure is not None:
                raise VisionSafetyStop(self._failure)
            if not self._started or self._closed:
                raise VisionSafetyStop("Safety monitor is not running")
            stale = self.gpu_enabled and (
                self._last_sample_at is None or self._now() - self._last_sample_at > max(3, 3 * self.interval_seconds)
            )
        if stale:
            self._latch("GPU telemetry has stopped updating")
            raise VisionSafetyStop(self._failure)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._close_streams()
