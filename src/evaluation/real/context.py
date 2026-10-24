"""Per-run state and component recording for the real evaluation backend."""

from dataclasses import dataclass, field
from time import perf_counter

from evaluation.models import ComponentResult, PipelineObservation, TraceEvent


def payload(value):
    """Return a JSON-compatible payload for models, dicts and scalars."""
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


@dataclass
class RealRun:
    """Mutable state carried across the real stages of one scenario."""

    scenario: object
    observed: PipelineObservation = field(default_factory=PipelineObservation)
    started: float = field(default_factory=perf_counter)
    use_reference: bool = False
    case: object = None
    extractions: dict = field(default_factory=dict)
    form: object = None
    report: object = None
    quote: object = None
    consistency: object = None
    features: object = None
    triage: object = None
    anomaly: object = None
    registered_anomaly: object = None
    history: object = None
    vision: object = None


class ComponentRecorder:
    """Record pass/fail components, explicit skips and observed versions."""

    def _notify(self, observation, component, status, **details):
        if getattr(self, "progress", None) is not None:
            self.progress(component, status, observation, **details)

    def _record(self, observation, component, operation, *, error_from_result=None, **trace_details):
        runtime = getattr(self, "resources", {}).get("vision_runtime")
        if runtime is not None:
            runtime.monitor.check()
        self._notify(observation, component, "RUNNING")
        started = perf_counter()
        try:
            result = operation()
            if runtime is not None:
                runtime.monitor.check()
            value = payload(result)
            outputs = value if isinstance(value, dict) else {"result": value}
            reason = error_from_result(result) if error_from_result is not None else None
            status = "FAIL" if reason is not None else "PASS"
        except Exception as error:
            from evaluation.real.safety.telemetry import VisionSafetyStop

            if isinstance(error, VisionSafetyStop):
                raise
            result, outputs = None, {}
            status, reason = "FAIL", f"{type(error).__name__}: {error}"
        if status == "FAIL":
            observation.errors.append(f"{component}: {reason}")
        duration = perf_counter() - started
        observation.components[component] = ComponentResult(status=status, duration_seconds=duration, reason=reason)
        observation.trace.append(
            TraceEvent(
                component=component,
                status=status,
                inputs=trace_details.get("inputs") or {},
                outputs=outputs,
                duration_seconds=duration,
                source_references=trace_details.get("sources") or [],
            )
        )
        self._notify(observation, component, status, duration_seconds=duration, reason=reason)
        return result

    def _skip(self, observation, component, reason, *, dependency=False):
        status = self.config.dependency_unavailable if dependency else "SKIP"
        observation.components[component] = ComponentResult(status=status, reason=reason)
        observation.trace.append(TraceEvent(component=component, status=status, outputs={"reason": reason}))
        if status == "FAIL":
            observation.errors.append(f"{component}: {reason}")
        self._notify(observation, component, status, reason=reason)

    def _available(self, observation, component, dependencies):
        missing = [name for name in dependencies if not self.preflight.available(name)]
        if missing:
            self._skip(
                observation,
                component,
                "; ".join(f"{name}: {self.preflight.reason(name)}" for name in missing),
                dependency=True,
            )
            return False
        return True

    def _version(self, observation, kind):
        row = next((row for row in self.preflight.rows if row["component"] == kind), {})
        if row.get("version"):
            observation.model_versions[kind] = str(row["version"])
        observation.provenance[kind] = {key: value for key, value in row.items() if key != "duration_seconds"}

    @staticmethod
    def _finish(observation, started):
        observation.components["total"] = ComponentResult(
            status="PASS",
            duration_seconds=perf_counter() - started,
            metadata={"meaning": "wall clock completion; inspect individual components for actual availability"},
        )
