"""Mock the NVIDIA executable; tests never query physical GPU hardware."""

import subprocess
from types import SimpleNamespace

import pytest

from evaluation.real.progress.gpu import gpu_text, read_gpu


def test_gpu_sample_parses_utilization_memory_temperature_and_power(monkeypatch):
    monkeypatch.setattr("evaluation.real.progress.gpu.shutil.which", lambda name: "fake-nvidia-smi")

    def run(command, **options):
        assert command[0] == "fake-nvidia-smi" and options["timeout"] == 1
        assert options["check"] is True and "--format=csv,noheader,nounits" in command
        return SimpleNamespace(stdout="0, 48, 2048, 8192, 65, 95.5, 150\n1, N/A, 0, 4096, N/A, N/A, N/A\n")

    monkeypatch.setattr("evaluation.real.progress.gpu.subprocess.run", run)
    sample = read_gpu()
    assert sample["available"] is True
    assert sample["devices"][0]["utilization_percent"] == 48
    assert "GPU0 48% | VRAM 2048/8192 MiB | 65C | 95.5/150W" in gpu_text(sample)
    assert "GPU1 N/A%" in gpu_text(sample)


@pytest.mark.parametrize("failure", [OSError("missing"), subprocess.TimeoutExpired("nvidia-smi", 1)])
def test_gpu_failure_is_nonfatal(monkeypatch, failure):
    monkeypatch.setattr("evaluation.real.progress.gpu.shutil.which", lambda name: "fake-nvidia-smi")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr("evaluation.real.progress.gpu.subprocess.run", fail)
    sample = read_gpu()
    assert sample["available"] is False
    assert "GPU telemetry unavailable" in gpu_text(sample)


def test_missing_nvidia_executable_does_not_launch_subprocess(monkeypatch):
    monkeypatch.setattr("evaluation.real.progress.gpu.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "evaluation.real.progress.gpu.subprocess.run", lambda *a, **k: pytest.fail("Unexpected process")
    )
    assert read_gpu() == {"available": False, "reason": "nvidia-smi not found"}
