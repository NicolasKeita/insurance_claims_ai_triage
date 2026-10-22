"""Monitoring unit tests must never inspect the machine's actual GPU."""

import pytest


@pytest.fixture(autouse=True)
def offline_telemetry(monkeypatch):
    monkeypatch.setattr(
        "evaluation.real.progress.monitor.read_gpu", lambda: {"available": False, "reason": "offline unit test"}
    )
