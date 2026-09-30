import hashlib
import subprocess
from pathlib import Path

from sklearn.pipeline import Pipeline

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