"""Framework-free doubles shared by vision lifecycle tests."""

from pathlib import Path
from types import SimpleNamespace

from evaluation.real.safety.runtime import VisionRuntime
from evaluation.real.safety.telemetry import VisionSafetyStop


class Monitor:
    def __init__(self):
        self.events = []
        self.closed = 0
        self.reason = None

    def check(self):
        if self.reason:
            raise VisionSafetyStop(self.reason)

    def event(self, event, **details):
        self.events.append((event, details))

    def close(self):
        self.closed += 1


class Memory:
    def __init__(self):
        self.operations = []

    def synchronize(self):
        self.operations.append("sync")

    def reset_peak(self):
        self.operations.append("reset")

    def empty_cache(self):
        self.operations.append("empty")

    def snapshot(self):
        self.operations.append("snapshot")
        return {"allocated_bytes": 12, "reserved_bytes": 20, "max_allocated_bytes": 30, "max_reserved_bytes": 40}


class Model:
    def __init__(self):
        self.overrides = {}
        self.device = None
        self.predictor = SimpleNamespace(results=[object()], batch=object(), model=self)

    def to(self, device):
        self.device = device


class Adapter:
    def __init__(self, name="damage"):
        self.settings = SimpleNamespace(model_path=Path(name + ".pt"))
        self.model_metadata = SimpleNamespace(version="local-v1")
        self.confidence_threshold = 0.25
        self._model = None
        self.load_count = 0
        self.infer_count = 0

    def _load_model(self):
        if self._model is None:
            self.load_count += 1
            self._model = Model()
        return self._model

    def assess_image(self, path):
        self.infer_count += 1
        self._load_model()
        return SimpleNamespace(width=640, height=480, filename=path.name)


def runtime(device="cpu"):
    monitor, memory = Monitor(), Memory()
    return VisionRuntime(monitor, device=device, memory=memory), monitor, memory
