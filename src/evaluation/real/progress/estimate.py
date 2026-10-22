"""Indicative remaining time from a matching completed scenario's stage timings."""


class HistoricalEstimate:
    """Replace completed historical stages with actual progress, without inventing timings."""

    def __init__(self, reference):
        self.total = reference.get("total")
        self.stages = {name: duration for name, duration in reference.items() if name not in {"total", "ml"}}
        self.completed = set()

    def snapshot(self, elapsed_seconds, component=None, stage_elapsed_seconds=0):
        remaining, reason = None, "no_matching_history"
        if self.total is not None:
            reason = None
            if not self.stages:
                remaining = self.total - elapsed_seconds
            elif component is not None and component not in self.stages:
                reason = "unrecorded_stage"
            elif component is not None and stage_elapsed_seconds >= self.stages[component]:
                reason = "reference_exceeded"
            else:
                overhead = max(0, self.total - sum(self.stages.values()))
                remaining = (
                    overhead
                    + sum(duration for name, duration in self.stages.items() if name not in self.completed)
                    - stage_elapsed_seconds
                )
            if remaining is not None and remaining <= 0:
                remaining, reason = None, "reference_exceeded"
        return {
            "estimated_remaining_seconds": remaining,
            "eta_reason": reason,
            "reference_total_seconds": self.total,
        }


def eta_text(estimate):
    """Use the same estimate in terminal updates and durable progress events."""
    remaining = estimate["estimated_remaining_seconds"]
    if remaining is not None:
        return f"ETA ~{remaining:.1f}s remaining (previous run)"
    reasons = {
        "no_matching_history": "no matching completed real run in this output directory",
        "unrecorded_stage": "active step has no matching recorded duration",
        "reference_exceeded": "previous recorded duration exceeded; waiting for step completion",
    }
    return "ETA unknown (" + reasons[estimate["eta_reason"]] + ")"
