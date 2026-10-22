"""Render scenario estimates alongside stage status and optional GPU telemetry."""

from evaluation.real.progress.estimate import eta_text
from evaluation.real.progress.gpu import gpu_text

LABELS = {
    "document": "Extract documents",
    "rag_generation": "Generate grounded answer with Ollama",
    "agent": "Review claim with the Ollama agent",
    "similar_indexing": "Index similar claims",
}


def print_started(directory, reference, estimate, interval):
    print(f"  Progress files: {directory}", flush=True)
    if "total" in reference:
        print(f"  Previous recorded run: {reference['total']:.1f}s (scenario execution only).", flush=True)
    print(f"  {eta_text(estimate)}; updates every {interval:g}s.", flush=True)


def print_stage(event, observation):
    component, status = event["component"], event["status"]
    label = LABELS.get(component, component.replace("_", " ").capitalize())
    duration = f" | {event['duration_seconds']:.1f}s" if "duration_seconds" in event else ""
    print(
        f"  [{status}] {component}: {label}{duration} | total {event['elapsed_seconds']:.1f}s | {eta_text(event)}",
        flush=True,
    )
    reason = observation.trace[-1].outputs.get("error") if observation.trace else None
    if component == "agent" and status == "FAIL" and reason == event.get("reason") and reason:
        print(f"    Reason: {reason}", flush=True)


def print_heartbeat(component, stage_elapsed, elapsed_seconds, sample, estimate):
    print(
        f"  [RUNNING] {component} | step {stage_elapsed:.1f}s | total {elapsed_seconds:.1f}s | "
        + eta_text(estimate)
        + " | "
        + gpu_text(sample),
        flush=True,
    )
