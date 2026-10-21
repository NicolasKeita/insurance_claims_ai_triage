"""Public entry point for one explicit, resumable damage-training run."""

from typing import Any

from vision.damage.training._config import DamageTrainingConfig, TrainingHooks, build_hooks
from vision.damage.training._flow._preparation import prepare_run
from vision.damage.training._flow._session import TrainingSession


def run_training(
    config: DamageTrainingConfig,
    hooks: TrainingHooks | None = None,
    **hook_overrides: Any,
) -> dict[str, Any]:
    """Train on TRAIN; Ultralytics chooses best/early stopping from VAL only.

    Keyword hook arguments (``model_factory``, ``environment_provider``, ``resume``,
    ...) remain supported. The factory is intentionally injected: the CLI imports
    Ultralytics only after the prepared dataset has passed validation, while tests
    use a lightweight stub.
    """
    hooks = build_hooks(hooks, hook_overrides, TrainingHooks)
    session = TrainingSession.start(config, hooks, prepare_run(config, hooks))
    from vision.damage.tracking import vision_tracking_run

    tracking_factory = hooks.tracking_factory or vision_tracking_run
    with tracking_factory(config) as tracking:
        session.train()
        return session.complete(tracking)
