"""Optional local NVIDIA telemetry, sampled without importing GPU frameworks."""

import csv
import io
import shutil
import subprocess

FIELDS = ("index", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw", "power.limit")
KEYS = (
    "index",
    "utilization_percent",
    "memory_used_mib",
    "memory_total_mib",
    "temperature_c",
    "power_w",
    "power_limit_w",
)


def read_gpu():
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=" + ",".join(FIELDS), "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        devices = []
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) != len(KEYS):
                raise ValueError("Unexpected telemetry fields")
            devices.append({key: number(value) for key, value in zip(KEYS, row, strict=True)})
        return {"available": bool(devices), "devices": devices}
    except (OSError, subprocess.SubprocessError, ValueError, csv.Error):
        return {"available": False, "reason": "NVIDIA telemetry could not be read"}


def number(value):
    try:
        number = float(value.strip())
        return number if number >= 0 and number < float("inf") else None
    except ValueError:
        return None


def gpu_text(sample):
    if not sample.get("available"):
        return "GPU telemetry unavailable: " + sample.get("reason", "no NVIDIA device reported")

    def value(device, key):
        number = device.get(key)
        return f"{number:g}" if number is not None else "N/A"

    return "; ".join(
        f"GPU{value(device, 'index')} {value(device, 'utilization_percent')}% | "
        f"VRAM {value(device, 'memory_used_mib')}/{value(device, 'memory_total_mib')} MiB | "
        f"{value(device, 'temperature_c')}C | {value(device, 'power_w')}/{value(device, 'power_limit_w')}W"
        for device in sample["devices"]
    )
