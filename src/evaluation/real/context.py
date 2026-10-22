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

    def _record(self, observation, component, operation, *, inputs=None, sources=None):
        self._notify(observation, component, "RUNNING")
        started = perf_counter()
        try:
            result = operation()
            value = payload(result)
            outputs = value if isinstance(value, dict) else {"result": value}
            status, reason = "PASS", None
        except Exception as error:
            result, outputs = None, {}
            status, reason = "FAIL", f"{type(error).__name__}: {error}"
            observation.errors.append(f"{component}: {reason}")
        duration = perf_counter() - started
        observation.components[component] = ComponentResult(status=status, duration_seconds=duration, reason=reason)
        observation.trace.append(
            TraceEvent(
                component=component,
                status=status,
                inputs=inputs or {},
                outputs=outputs,
                duration_seconds=duration,
                source_references=sources or [],
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

    def _capture_vision(self, observation, vision):
        if vision is None:
            return
        observation.vision = vision.model_dump(mode="json")
        observation.facts["vision"] = observation.vision
        observation.model_versions.update(
            damage_vision=vision.damage_model_metadata.version, part_vision=vision.part_model_metadata.version
        )
        observation.policy_versions["fusion"] = vision.fusion_policy.version
        observation.provenance["vision"] = {
            "images": [image.filename for image in vision.images],
            "damage_model": vision.damage_model_metadata.model_dump(mode="json"),
            "part_model": vision.part_model_metadata.model_dump(mode="json"),
            "geometry": "Production schema validates normalized geometry and source identity",
        }

    @staticmethod
    def _finish(observation, started):
        observation.components["total"] = ComponentResult(
            status="PASS",
            duration_seconds=perf_counter() - started,
            metadata={"meaning": "wall clock completion; inspect individual components for actual availability"},
        )
