"""Test environment."""

from types import SimpleNamespace

from tests.intelligence.vision.training.damage._dataset import (
    offline_environment as offline_environment,
)
from tests.intelligence.vision.training.damage._dataset import prepared as prepared
from vision.damage.training import (
    observe_environment,
)


def test_environment_records_observed_cpu_without_claiming_cuda():
    torch = SimpleNamespace(
        __version__="observed-test",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    observed = observe_environment(torch, SimpleNamespace(__version__="observed-ultralytics"))
    assert observed["torch_version"] == "observed-test"
    assert observed["cuda_available"] is False and observed["cuda_devices"] == []
    assert observed["cuda_device_properties"] == []


def test_environment_records_actual_cuda_runtime_and_vram():
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda _: "Observed test GPU",
        get_device_properties=lambda _: SimpleNamespace(total_memory=8 * 1024**3),
        get_device_capability=lambda _: (8, 6),
    )
    observed = observe_environment(
        SimpleNamespace(__version__="test+cuda", version=SimpleNamespace(cuda="13.2"), cuda=cuda),
        SimpleNamespace(__version__="test"),
    )
    assert observed["cuda_available"] is True and observed["cuda_runtime_version"] == "13.2"
    assert observed["cuda_device_properties"][0]["total_memory_gib"] == 8.0
    assert observed["cuda_device_properties"][0]["compute_capability"] == [8, 6]
