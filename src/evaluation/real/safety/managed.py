"""Managed detector contract and disposal of framework-held GPU references."""

from pathlib import Path


def cuda_failure(error):
    """Adapters wrap framework errors, so inspect the complete exception chain."""
    pending, seen = [error], set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        text = f"{type(item).__name__}: {item}".casefold()
        if any(term in text for term in ("cuda", "cudnn", "cublas", "out of memory", "outofmemory")):
            return True
        pending.extend(cause for cause in (item.__cause__, item.__context__) if cause is not None)
    return False


def clear_results(model, *, unload=False):
    """Drop mapped image results, and the predictor network when unloading."""
    if model is None:
        return
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        fields = ("results", "model", "batch") if unload else ("results", "batch")
        for field in fields:
            if hasattr(predictor, field):
                setattr(predictor, field, None)
        if unload:
            model.predictor = None
    if hasattr(model, "results"):
        model.results = None


def discard_model(adapter):
    """Clear all local model references before the caller collects CUDA memory."""
    model, adapter._model = adapter._model, None
    clear_results(model, unload=True)


class ManagedDetector:
    """Preserve the detector contract while routing all work through its runtime."""

    def __init__(self, runtime, adapter, *, name=None):
        self.runtime, self.adapter = runtime, adapter
        self.name = name or type(adapter).__name__

    @property
    def confidence_threshold(self):
        return self.adapter.confidence_threshold

    @property
    def model_metadata(self):
        return self.adapter.model_metadata

    def _details(self):
        return {"model": self.name, "checkpoint": str(self.adapter.settings.model_path), "device": self.runtime.device}

    def load(self):
        """Load for preflight inspection; callers must not retain the returned model."""
        with self.runtime._lock:
            return self.runtime._load(self)

    def assess_image(self, path: Path):
        path = Path(path)
        runtime = self.runtime
        with runtime._lock:
            runtime._load(self)
            details = {**self._details(), "image": str(path), "imgsz": runtime.imgsz, "batch": 1}
            runtime.monitor.event("inference_start", **details)
            try:
                runtime._check()
                runtime.memory.synchronize()
                runtime.memory.reset_peak()
                assessment = self.adapter.assess_image(path)
                runtime.memory.synchronize()
                runtime.monitor.event(
                    "inference_end",
                    memory=runtime.memory.snapshot(),
                    width=assessment.width,
                    height=assessment.height,
                    **details,
                )
                runtime._check()
                return assessment
            except Exception as error:
                runtime._failed("inference", error, **details)
            finally:
                clear_results(self.adapter._model)
