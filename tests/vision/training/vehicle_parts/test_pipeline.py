"""Test pipeline."""

import json
from contextlib import nullcontext
from dataclasses import replace

import pytest

from tests.vision.training.vehicle_parts._support import PartStub
from vision.carparts_dataset import MANIFEST_NAME
from vision.parts_evaluation import run_parts_evaluation
from vision.parts_training import (
    load_parts_training_config,
    run_parts_training,
)


def test_reused_pipeline_preserves_train_val_and_final_test_once(tmp_path, monkeypatch):
    """
    Independent part threshold cannot accidentally read the damage environment.
    """
    monkeypatch.delenv("DAMAGE_CONFIDENCE_THRESHOLD", raising=False)
    config = replace(
        load_parts_training_config(),
        dataset_yaml=tmp_path / "dataset.yaml",
        output_dir=tmp_path / "run",
        device="cpu",
        track_mlflow=False,
    )
    config.dataset_yaml.write_text("{}")
    manifest = {
        "category_mapping": [{"yolo_class_id": 0, "category_id": 0, "name": "hood"}],
        "annotation_hashes": {split: "a" * 64 for split in ("train", "val", "test")},
        "splits": {
            split: {"image_count": 1, "annotations_per_category": {"hood": 1}} for split in ("train", "val", "test")
        },
    }
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
    model = PartStub()
    kwargs = {
        "model_factory": lambda _: model,
        "environment_provider": lambda: {"cuda_available": False},
        "dataset_loader": lambda _: manifest,
        "tracking_factory": lambda *_args, **_kwargs: nullcontext(None),
    }
    summary = run_parts_training(config, **kwargs)
    assert model.train_arguments["split"] == "val" and summary["test_used_during_training"] is False
    monkeypatch.setenv("DAMAGE_CONFIDENCE_THRESHOLD", "0.99")
    result = run_parts_evaluation(config, **kwargs, max_error_examples=0)
    assert model.val_arguments["split"] == "test" and model.val_arguments["conf"] == 0.25
    assert result["checkpoint_selection_split"] == "val" and result["environment"]["device_used"] == "cpu"
    with pytest.raises(FileExistsError, match="already exist"):
        run_parts_evaluation(config, **kwargs, max_error_examples=0)
