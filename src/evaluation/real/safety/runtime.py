"""Serialize real vision inference and retain only one loaded YOLO model."""

import gc
from threading import RLock

from evaluation.real.safety.cuda import CudaMemory
from evaluation.real.safety.managed import ManagedDetector, cuda_failure, discard_model
from evaluation.real.safety.telemetry import VisionSafetyStop


class VisionRuntime:
    """One lock covers model changes, inference and cleanup for both adapters."""

    def __init__(self, monitor, device="cpu", imgsz=640, *, memory=None):
        self.monitor = monitor
        self.device = "cuda:" + str(device) if str(device).isdigit() else str(device)
        if self.device in {"None", ""}:
            raise ValueError("Resolve an explicit vision device before starting the safety monitor")
        if not isinstance(imgsz, int) or imgsz < 32 or imgsz % 32:
            raise ValueError("Vision inference size must be a positive multiple of 32")
        self.imgsz = imgsz
        self.memory = memory if memory is not None else CudaMemory(self.device)
        self._lock = RLock()
        self._active = None
        self._abort_reason = None
        self._closed = False

    def manage(self, adapter, *, name=None):
        return ManagedDetector(self, adapter, name=name)

    def _check(self):
        if self._closed:
            raise VisionSafetyStop("The vision runtime is already closed")
        if self._abort_reason is not None:
            raise VisionSafetyStop(self._abort_reason)
        self.monitor.check()

    def _failed(self, operation, error, **details):
        self.monitor.event(operation + "_error", error_type=type(error).__name__, error=str(error), **details)
        if isinstance(error, VisionSafetyStop):
            self._abort_reason = str(error)
            raise error
        if cuda_failure(error):
            self._abort_reason = f"Vision stopped after {operation}: {type(error).__name__}: {error}"
            self.monitor.event("vision_abort", reason=self._abort_reason)
            raise VisionSafetyStop(self._abort_reason) from error
        raise error

    def _load(self, managed):
        self._check()
        if self._active is managed and managed.adapter._model is not None:
            return managed.adapter._model
        self._unload()
        self._check()
        details = managed._details()
        self.monitor.event("model_load_start", **details)
        self._active = managed
        model = None
        try:
            self.memory.synchronize()
            before = self.memory.snapshot()
            model = managed.adapter._load_model()
            if callable(getattr(model, "to", None)):
                model.to(self.device)
            model.overrides.update(device=self.device, imgsz=self.imgsz, batch=1, workers=0, augment=False)
            self.memory.synchronize()
            self.monitor.event("model_load_end", memory_before=before, memory=self.memory.snapshot(), **details)
            self._check()
            return model
        except Exception as error:
            model = None
            try:
                self._unload()
            except Exception as cleanup_error:
                self.monitor.event("model_cleanup_error", error=str(cleanup_error), **details)
            self._failed("model_load", error, **details)

    def _unload(self):
        if self._active is None:
            return
        managed, self._active = self._active, None
        details = managed._details()
        try:
            try:
                self.monitor.event("model_unload_start", **details)
            finally:
                discard_model(managed.adapter)
                gc.collect()
                self.memory.empty_cache()
            self.monitor.event("model_unload_end", memory=self.memory.snapshot(), **details)
        except Exception as error:
            self._failed("model_unload", error, **details)

    def release(self):
        """Cleanup remains possible after a thermal or CUDA safety stop."""
        with self._lock:
            self._unload()

    def close(self):
        """Close the monitor without masking an earlier CUDA cleanup failure."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._unload()
            except Exception:
                pass
            finally:
                self.monitor.close()
