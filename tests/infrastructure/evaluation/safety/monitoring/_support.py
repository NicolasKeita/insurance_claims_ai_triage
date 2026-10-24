"""Small NVIDIA samples and disk failures shared by monitoring unit tests."""

import json


def gpu_sample(temperature=55):
    return {
        "available": True,
        "devices": [
            {
                "index": 0,
                "memory_used_mib": 300,
                "memory_total_mib": 8192,
                "temperature_c": temperature,
                "power_w": None,
                "power_limit_w": None,
                "utilization_percent": 100,
            }
        ],
    }


def events(directory):
    return [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def broken_disk(stream):
    raise OSError("disk full")
