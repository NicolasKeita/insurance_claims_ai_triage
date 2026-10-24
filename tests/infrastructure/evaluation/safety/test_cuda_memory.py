"""PyTorch measurements are lazy and tied to the selected device."""

from contextlib import contextmanager
from types import SimpleNamespace

from evaluation.real.safety import cuda
from evaluation.real.safety.cuda import CudaMemory


def test_cpu_counter_operations_never_import_torch(monkeypatch):
    def forbidden(name):
        raise AssertionError("CPU telemetry must not import torch")

    monkeypatch.setattr(cuda.importlib, "import_module", forbidden)
    memory = CudaMemory("cpu")
    memory.synchronize()
    memory.reset_peak()
    memory.empty_cache()
    assert memory.enabled is False
    assert memory.snapshot() == {}


def test_cuda_snapshot_and_reset_synchronize_and_cache_uses_selected_device(monkeypatch):
    calls = []

    def record(name, result=None):
        def method(device=None):
            calls.append((name, device))
            return result

        return method

    @contextmanager
    def selected(device):
        calls.append(("enter_device", device))
        yield
        calls.append(("leave_device", device))

    simulated = SimpleNamespace(
        synchronize=record("synchronize"),
        memory_allocated=record("allocated", 100),
        memory_reserved=record("reserved", 200),
        max_memory_allocated=record("max_allocated", 150),
        max_memory_reserved=record("max_reserved", 250),
        reset_peak_memory_stats=record("reset_peak"),
        device=selected,
        empty_cache=record("empty_cache"),
    )
    monkeypatch.setattr(cuda.importlib, "import_module", lambda name: SimpleNamespace(cuda=simulated))
    memory = CudaMemory("cuda:1")
    assert memory.enabled is True
    assert memory.snapshot() == {
        "allocated_bytes": 100,
        "reserved_bytes": 200,
        "max_allocated_bytes": 150,
        "max_reserved_bytes": 250,
    }
    assert calls[:2] == [("synchronize", "cuda:1"), ("allocated", "cuda:1")]
    memory.reset_peak()
    assert calls[-2:] == [("synchronize", "cuda:1"), ("reset_peak", "cuda:1")]
    memory.empty_cache()
    assert calls[-3:] == [("enter_device", "cuda:1"), ("empty_cache", None), ("leave_device", "cuda:1")]
