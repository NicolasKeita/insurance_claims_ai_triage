"""One explicit training execution: model loading, callbacks and the framework call."""

import time
from dataclasses import dataclass
from typing import Any

from vision.training._artifacts import disable_builtin_mlflow_callbacks
from vision.training._config import DamageTrainingConfig, TrainingHooks
from vision.training._flow._callbacks import checkpoint_progress_recorder, resume_stopper_restorer
from vision.training._flow._summary import complete_training
from vision.training._records import PreparedRun, RunIdentity, base_checkpoint_identity
from vision.training._resume import validate_resume_checkpoint


@dataclass
class TrainingSession:
    """Mutable execution state for one run; TEST is never observed here."""

    config: DamageTrainingConfig
    prepared: PreparedRun
    model: Any
    identity: RunIdentity
    saved_checkpoint: dict[str, Any] | None = None
    trainer: Any = None
    result: Any = None

    @classmethod
    def start(cls, config: DamageTrainingConfig, hooks: TrainingHooks, prepared: PreparedRun) -> "TrainingSession":
        environment = hooks.environment_provider()
        initial_checkpoint = str(config.output_dir / "train/weights/last.pt") if hooks.resume else config.base_model
        model = hooks.model_factory(initial_checkpoint)
        if getattr(model, "task", None) != "segment":
            raise ValueError("Base weights must contain an instance segmentation model (task=segment)")
        model.add_callback("on_pretrain_routine_start", disable_builtin_mlflow_callbacks)
        recovery = prepared.recovery
        saved_checkpoint = validate_resume_checkpoint(model, config, recovery, prepared.manifest) if recovery else None
        if not hooks.resume:
            config.output_dir.mkdir(parents=True, exist_ok=False)
        base_checkpoint, base_sha256 = base_checkpoint_identity(model, config, recovery)
        identity = RunIdentity(
            config=config,
            prepared=prepared,
            environment=environment,
            provenance=prepared.provenance_provider(config.dataset_yaml, prepared.manifest),
            base_checkpoint=base_checkpoint,
            base_checkpoint_sha256=base_sha256,
            resume=hooks.resume,
            previous_elapsed=float(recovery["elapsed_seconds"]) if recovery else 0.0,
            started=time.perf_counter(),
            progress={"best_epoch": recovery["best_epoch"] if recovery else None},
        )
        session = cls(
            config=config, prepared=prepared, model=model, identity=identity, saved_checkpoint=saved_checkpoint
        )
        session._install_callbacks()
        return session

    def _install_callbacks(self) -> None:
        recovery = self.prepared.recovery
        if recovery:
            self.model.add_callback(
                "on_pretrain_routine_end",
                resume_stopper_restorer(self.config, recovery, self.saved_checkpoint),
            )
        self.model.add_callback("on_model_save", checkpoint_progress_recorder(self.identity))

    def training_arguments(self) -> dict[str, Any]:
        """TRAIN fits the model; VAL alone selects best.pt and early stopping."""
        config = self.config
        arguments = dict(
            data=str(config.dataset_yaml),
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            patience=config.patience,
            seed=config.seed,
            device=config.device,
            workers=config.workers,
            optimizer=config.optimizer,
            project=str(config.output_dir),
            name="train",
            exist_ok=False,
            pretrained=True,
            val=True,
            split="val",
            save=True,
            plots=True,
        )
        if self.prepared.recovery:
            arguments.update(
                {
                    "resume": str(config.output_dir / "train/weights/last.pt"),
                    "save_dir": str(config.output_dir / "train"),
                }
            )
        return arguments

    def train(self) -> None:
        self.result = self.model.train(**self.training_arguments())
        self.trainer = self.model.trainer

    def complete(self, tracking: Any) -> dict[str, Any]:
        return complete_training(self.identity, self.model, self.result, tracking)
