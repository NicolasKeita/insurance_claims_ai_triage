"""Reuse the bounded NVIDIA probe for CSV samples and a cooperative stop request."""

import csv
import math
from copy import deepcopy
from threading import Thread

CSV_FIELDS = (
    "timestamp",
    "run_id",
    "index",
    "memory_used_mib",
    "memory_total_mib",
    "temperature_c",
    "power_w",
    "utilization_percent",
)


class GpuSampling:
    """The first sample is synchronous; later samples use one daemon worker."""

    def _open_gpu(self):
        self._gpu = (self.output_dir / "gpu.csv").open("a", encoding="utf-8", newline="")
        self._csv = csv.DictWriter(self._gpu, fieldnames=CSV_FIELDS)
        if self._gpu.tell() == 0:
            self._csv.writeheader()
            self._flush(self._gpu)

    def _start_sampling(self):
        self._sample()
        self.check()
        self._thread = Thread(target=self._watch, daemon=True, name="vision-safety")
        self._thread.start()

    def last_sample(self):
        with self._lock:
            return deepcopy(self._last_sample)

    def _sample(self):
        sample = self._reader()
        with self._lock:
            if self._closed or self._stop.is_set():
                return
            self._last_sample = deepcopy(sample)
        if not sample.get("available") or not sample.get("devices"):
            raise ValueError(sample.get("reason", "No NVIDIA GPU temperature was reported"))
        rows = sample["devices"]
        for row in rows:
            temperature = row.get("temperature_c")
            if temperature is None or not math.isfinite(temperature) or temperature < 0:
                raise ValueError("NVIDIA GPU temperature is unavailable")
        self._write_sample(rows)
        hot = next((row for row in rows if row["temperature_c"] >= self.max_temperature_c), None)
        if hot is not None:
            self._latch(
                f"GPU {hot['index']} reached {hot['temperature_c']:g} C (stop threshold: {self.max_temperature_c:g} C)"
            )

    def _write_sample(self, rows):
        with self._lock:
            if self._closed or self._stop.is_set():
                return
            self._last_sample_at = self._now()
            timestamp = self._timestamp()
            self._csv.writerows(
                {"timestamp": timestamp, "run_id": self.run_id, **{key: row.get(key) for key in CSV_FIELDS[2:]}}
                for row in rows
            )
            self._flush(self._gpu)

    def _watch(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                self._sample()
            except Exception as error:
                self._latch(f"GPU telemetry is unavailable: {error}")
