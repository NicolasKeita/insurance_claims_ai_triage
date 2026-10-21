"""Support."""

from types import SimpleNamespace


def make_run(*, status="FINISHED", task="novelty_detection", model_uri="models:/m-1"):
    return SimpleNamespace(
        info=SimpleNamespace(run_id="run-1", status=status),
        data=SimpleNamespace(
            tags={"task": task, "candidate.model_uri": model_uri},
            params={
                "dataset.reference.version": "anomaly_reference_v1",
                "dataset.reference.sha256": "a" * 64,
                "dataset.evaluation.version": "anomaly_evaluation_v1",
                "dataset.evaluation.sha256": "b" * 64,
                "threshold.percentile": "0.98",
                "threshold.value": "0.616",
            },
        ),
    )
