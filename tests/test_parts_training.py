"""Part orchestration with injected stubs; CPU-only, no real weights/dataset."""

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vision.carparts_dataset import MANIFEST_NAME
from vision.error_analysis import ErrorInstance, analyze_instances
from vision.parts_evaluation import run_parts_evaluation
from vision.parts_training import load_parts_training_config, require_requested_device, run_parts_training


def test_frozen_part_baseline_independent_coco_and_experiment():
    config = load_parts_training_config()
    assert (config.epochs, config.imgsz, config.batch, config.patience, config.seed, config.device, config.workers) == (
        100,
        640,
        8,
        20,
        42,
        "0",
        0,
    )
    assert config.base_model == "yolo26n-seg.pt" and config.confidence_threshold == 0.25
    assert config.experiment_name == "insurance-claims-vision-parts"
    with pytest.raises(ValueError, match="independent COCO"):
        replace(config, base_model="artifacts/vision/damage_segmentation_v1/train/weights/best.pt")


def test_gpu_request_refuses_cpu_build_before_model_loading(tmp_path):
    config = replace(load_parts_training_config(), dataset_yaml=tmp_path / "dataset.yaml", output_dir=tmp_path / "run")
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        run_parts_training(
            config,
            model_factory=lambda _: pytest.fail("must not load model"),
            environment_provider=lambda: {
                "cuda_available": False,
                "python_executable": "cpu-python",
                "torch_version": "cpu",
            },
            dataset_loader=lambda _: {},
        )
    assert not config.output_dir.exists()


def test_gpu_preflight_reports_and_requires_explicit_zero():
    config = load_parts_training_config()
    environment = {"cuda_available": True, "python_executable": "actual-python", "torch_version": "CUDA build"}
    assert require_requested_device(config, environment) is environment
    with pytest.raises(ValueError, match="explicit device=0"):
        require_requested_device(replace(config, device="cuda"), environment)


class PartStub:
    task = "segment"
    names = {0: "hood"}

    def __init__(self):
        self.callbacks = {}

    def add_callback(self, event, callback):
        self.callbacks.setdefault(event, []).append(callback)

    def train(self, **kwargs):
        self.train_arguments = kwargs
        best = Path(kwargs["project"]) / "train/weights/best.pt"
        best.parent.mkdir(parents=True)
        best.write_bytes(b"parts-selected-on-VAL-fixture")
        last = best.with_name("last.pt")
        last.write_bytes(b"parts-last-fixture")
        self.trainer = SimpleNamespace(
            best=best,
            last=last,
            device="cpu",
            epoch=99,
            fitness=0.4,
            best_fitness=0.4,
            stopper=SimpleNamespace(best_epoch=100),
        )
        for callback in self.callbacks["on_model_save"]:
            callback(self.trainer)
        return SimpleNamespace()

    def val(self, **kwargs):
        self.val_arguments = kwargs
        for callback in self.callbacks.get("on_val_start", []):
            callback(SimpleNamespace(device="cpu"))
        directory = Path(kwargs["project"]) / "evaluation"
        directory.mkdir()
        return SimpleNamespace(save_dir=directory)


def test_cuda_available_but_framework_cpu_device_is_rejected(tmp_path):
    config = replace(
        load_parts_training_config(),
        dataset_yaml=tmp_path / "dataset.yaml",
        output_dir=tmp_path / "run",
        track_mlflow=False,
    )
    config.dataset_yaml.write_text("{}")
    (tmp_path / MANIFEST_NAME).write_text("{}")
    model = PartStub()

    def train(**_):
        for callback in model.callbacks["on_train_start"]:
            callback(SimpleNamespace(device="cpu", epoch=-1))
        pytest.fail("Runtime must refuse unexpected CPU before epochs")

    model.train = train
    with pytest.raises(RuntimeError, match="unexpected trainer device"):
        run_parts_training(
            config,
            model_factory=lambda _: model,
            environment_provider=lambda: {"cuda_available": True},
            dataset_loader=lambda _: {},
            tracking_factory=lambda *_args, **_kwargs: nullcontext(None),
        )


def test_reused_pipeline_preserves_train_val_and_final_test_once(tmp_path, monkeypatch):
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
    # Independent part threshold cannot accidentally read the damage environment.
    monkeypatch.setenv("DAMAGE_CONFIDENCE_THRESHOLD", "0.99")
    result = run_parts_evaluation(config, **kwargs, max_error_examples=0)
    assert model.val_arguments["split"] == "test" and model.val_arguments["conf"] == 0.25
    assert result["checkpoint_selection_split"] == "val" and result["environment"]["device_used"] == "cpu"
    with pytest.raises(FileExistsError, match="already exist"):
        run_parts_evaluation(config, **kwargs, max_error_examples=0)


def test_wrong_part_and_incomplete_masks_are_honest_separate_diagnostics():
    full = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    half = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))
    wrong = analyze_instances(
        [ErrorInstance(1, (0, 0, 1, 1), 0.9, full)],
        [ErrorInstance(0, (0, 0, 1, 1), polygon=full)],
        confidence_threshold=0.25,
        image_size=(100, 100),
        include_part_errors=True,
    )
    assert len(wrong["wrong_parts"]) == 1 and wrong["matches"] == []
    incomplete = analyze_instances(
        [ErrorInstance(0, (0, 0, 1, 1), 0.9, half)],
        [ErrorInstance(0, (0, 0, 1, 1), polygon=full)],
        confidence_threshold=0.25,
        image_size=(100, 100),
        include_part_errors=True,
    )
    assert len(incomplete["incomplete_masks"]) == 1
    assert incomplete["incomplete_masks"][0]["truth_mask_coverage"] == pytest.approx(0.51)
