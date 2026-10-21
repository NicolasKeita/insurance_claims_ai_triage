"""Offline-safe MLflow run bootstrap for vision training and evaluation."""

import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from vision.damage.tracking._tracker import VisionMLflowTracking
from vision.damage.training import DamageTrainingConfig


def _start_run(config: DamageTrainingConfig, run_id: str | None) -> tuple[Any, Any] | None:
    """Configure the shared tracking store and start one run; None keeps CV running."""
    try:
        import mlflow
        from mlflow.exceptions import MlflowException

        from ml.tracking import configure_mlflow_tracking
    except ImportError:
        warnings.warn(
            "Vision MLflow tracking unavailable: dependency import failed; continuing CV without tracking.",
            RuntimeWarning,
        )
        return None
    try:
        configure_mlflow_tracking()
        mlflow.set_experiment(config.experiment_name)
        kwargs = {"run_id": run_id} if run_id else {"run_name": config.version}
        active_run = mlflow.start_run(**kwargs)
    except MlflowException:
        warnings.warn(
            "Vision MLflow tracking unavailable: server run initialization failed; continuing CV without tracking.",
            RuntimeWarning,
        )
        return None
    return mlflow, active_run


@contextmanager
def vision_tracking_run(
    config: DamageTrainingConfig,
    *,
    run_id: str | None = None,
) -> Iterator[VisionMLflowTracking | None]:
    """Reuse the repository's tracking configuration, without imports when off.

    Never catch the training body: a CV failure must still propagate. Avoid printing an exception
    containing credentials from a server URI.
    """
    if not (config.track_mlflow or os.environ.get("MLFLOW_TRACKING_URI")):
        yield None
        return
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        warnings.warn(
            "Vision MLflow tracking unavailable: MLFLOW_TRACKING_URI is not configured; "
            "continuing CV without tracking.",
            RuntimeWarning,
        )
        yield None
        return
    started = _start_run(config, run_id)
    if started is None:
        yield None
        return
    mlflow, active_run = started
    with active_run as run:
        yield VisionMLflowTracking(mlflow, run.info.run_id, log_checkpoint=config.log_checkpoint)
