"""Explicit training orchestration; importing this module loads no CV framework."""

from vision.damage.training._artifacts import (
    dataset_provenance,
    disable_builtin_mlflow_callbacks,
    file_sha256,
    observe_environment,
    verify_class_names,
    write_json,
)
from vision.damage.training._config import (
    DEFAULT_TRAINING_CONFIG,
    REPO_ROOT,
    DamageTrainingConfig,
    TrainingHooks,
    build_hooks,
    load_training_config,
)
from vision.damage.training._flow import run_training
from vision.damage.training._resume import load_resume_metadata, validate_resume_checkpoint

__all__ = [
    "DEFAULT_TRAINING_CONFIG",
    "REPO_ROOT",
    "DamageTrainingConfig",
    "TrainingHooks",
    "build_hooks",
    "dataset_provenance",
    "disable_builtin_mlflow_callbacks",
    "file_sha256",
    "load_resume_metadata",
    "load_training_config",
    "observe_environment",
    "run_training",
    "validate_resume_checkpoint",
    "verify_class_names",
    "write_json",
]
