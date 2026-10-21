"""Resume."""

import copy
import json
from types import SimpleNamespace

from tests.vision.training.damage._framework import OrchestrationStub, example_metric
from vision.training import (
    dataset_provenance,
    file_sha256,
)


def interrupted_fixture(config, manifest):
    directory = config.output_dir / "train/weights"
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(b"selected best at epoch 10")
    (directory / "last.pt").write_bytes(b"resumable state at epoch 12")
    base = config.output_dir.parent / "original_pretrained.pt"
    base.write_bytes(b"original pretrained fixture")
    metadata = {
        "config": config.payload(),
        "dataset": copy.deepcopy(dataset_provenance(config.dataset_yaml, manifest)),
        "confidence_threshold": 0.25,
        "checkpoint_selection_split": "val",
        "epochs_completed": 12,
        "best_epoch": 10,
        "elapsed_seconds": 123.5,
        "base_checkpoint": str(base),
        "base_checkpoint_sha256": file_sha256(base),
        "best_checkpoint_sha256": file_sha256(directory / "best.pt"),
        "last_checkpoint_sha256": file_sha256(directory / "last.pt"),
    }
    (config.output_dir / "recovery_training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


class ResumeStub(OrchestrationStub):
    def __init__(self, config, metadata):
        super().__init__()
        self.config = config
        self.ckpt = {
            "epoch": 11,
            "optimizer": {"fixture": "retained"},
            "best_fitness": 1.0,
            "stopper": {"best_fitness": 1.0, "best_epoch": 8},
            "train_args": {
                **{key: getattr(config, key) for key in ("epochs", "imgsz", "batch", "patience", "seed", "workers")},
                "optimizer": "AdamW",
                "task": "segment",
                "split": "val",
                "val": True,
                "pretrained": True,
                "data": str(config.dataset_yaml),
                "project": str(config.output_dir),
                "name": "train",
            },
        }

    def train(self, **kwargs):
        self.training_kwargs = kwargs
        self.trainer = SimpleNamespace(
            best=self.config.output_dir / "train/weights/best.pt",
            last=self.config.output_dir / "train/weights/last.pt",
            start_epoch=12,
            epochs=100,
            epoch=12,
            device="cpu",
            fitness=0.8,
            best_fitness=0.0,
            stopper=SimpleNamespace(best_fitness=0.0, best_epoch=0),
        )
        for callback in self.callbacks.get("on_pretrain_routine_end", []):
            callback(self.trainer)
        for callback in self.callbacks["on_model_save"]:
            callback(self.trainer)
        self.trainer.epoch = 99
        self.trainer.last.write_bytes(b"final resumed fixture")
        for callback in self.callbacks["on_model_save"]:
            callback(self.trainer)
        return example_metric()
