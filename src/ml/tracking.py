import hashlib
import os
import subprocess
import sys
from pathlib import Path

import mlflow
from sklearn.pipeline import Pipeline


def configure_mlflow_tracking() -> str:
    """Use the explicitly configured tracking server for MLOps commands."""
    # MLflow prints Unicode run links when a run ends. Windows consoles may
    # default to cp1252, which otherwise makes a successful run exit with an
    # encoding error.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required")
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def compute_file_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()

def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return None

    return result.stdout.strip()

def get_classifier_params(
    model: Pipeline,
) -> dict[str, str]:
    classifier = model.named_steps[
        "classifier"
    ]

    return {
        f"classifier.{name}": str(value)
        for name, value
        in classifier.get_params(
            deep=False
        ).items()
    }
