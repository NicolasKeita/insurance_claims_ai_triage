"""Latency summaries with explicit sample counts and guarded tail statistics."""

from math import ceil
from statistics import mean, median

LATENCY_NAMES = (
    "document",
    "vision",
    "triage",
    "anomaly",
    "ml",
    "retrieval",
    "rag_generation",
    "agent",
    "total",
)
TAIL_SAMPLE_MINIMUM = 20


def _samples(results, name):
    return [
        result.timings[name]
        for result in results
        if name in result.timings
        and (
            name in {"ml", "total"}
            or result.component_results.get(name)
            and result.component_results[name].status != "SKIP"
        )
    ]


def latency_summary(results, names=LATENCY_NAMES):
    """Tail summaries require enough samples; small runs stay diagnostics."""
    timings = {}
    for name in names:
        samples = _samples(results, name)
        timings[name] = {
            "count": len(samples),
            "individual_seconds": samples,
            "mean_seconds": mean(samples) if samples else None,
        }
        if len(samples) >= TAIL_SAMPLE_MINIMUM:
            ordered = sorted(samples)
            timings[name].update(
                median_seconds=median(samples),
                p95_seconds=ordered[max(0, ceil(0.95 * len(samples)) - 1)],
            )
    return timings