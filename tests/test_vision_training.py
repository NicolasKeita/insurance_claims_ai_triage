"""Offline orchestration stubs: no model imports/downloads or training epochs."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vision.error_analysis import ErrorInstance, analyze_instances, box_iou, save_error_examples
from vision.evaluation import extract_segmentation_metrics, run_evaluation, selected_checkpoint
from vision.tracking import VisionMLflowTracking, vision_tracking_run
from vision.training import (
    DamageTrainingConfig, dataset_provenance, disable_builtin_mlflow_callbacks,
    file_sha256, load_training_config, observe_environment, run_training, verify_class_names,
)


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DAMAGE_BASE_MODEL", raising=False)
    monkeypatch.delenv("DAMAGE_CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("DAMAGE_DATASET_YAML", raising=False)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text('{"path":".","names":["dent","scratch"]}', encoding="utf-8")
    manifest = {
        "preparation_version": "test-fixture-only",
        "annotation_hashes": {split: str(index) * 64 for index, split in enumerate(("train", "val", "test"), 1)},
        "category_mapping": [
            {"category_id": 7, "yolo_class_id": 0, "name": "dent", "damage_type": "dent"},
            {"category_id": 42, "yolo_class_id": 1, "name": "scratch", "damage_type": "scratch"},
        ],
        "splits": {split: {"image_count": 3, "annotation_count": 2, "annotations_per_category": {"dent": 0, "scratch": 2}, "images_per_category": {"dent": 0, "scratch": 1}} for split in ("train", "val", "test")},
    }
    (root / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = DamageTrainingConfig(
        version="fixture", epochs=100, imgsz=640, batch=16, patience=100,
        seed=42, device="cpu", workers=0, optimizer="auto", base_model="yolo26n-seg.pt",
        dataset_yaml=dataset_yaml, output_dir=tmp_path / "run", experiment_name="fixture",
    )
    return config, manifest


def example_metric():
    group = SimpleNamespace(mp=0.7, mr=0.5, map50=0.6, map=0.4, ap_class_index=[1], class_result=lambda index: [0.7, 0.5, 0.6, 0.4])
    return SimpleNamespace(box=group, seg=group)


class OrchestrationStub:
    """Simulates framework responses with bytes in tmp_path, never CV weights."""
    task = "segment"
    names = {0: "dent", 1: "scratch"}
    device = "cpu"

    def __init__(self):
        self.callbacks = {}
        self.training_kwargs = None
        self.validation_kwargs = None

    def add_callback(self, event, callback):
        self.callbacks.setdefault(event, []).append(callback)

    def train(self, **kwargs):
        self.training_kwargs = kwargs
        best = Path(kwargs["project"]) / kwargs["name"] / "weights/best.pt"
        best.parent.mkdir(parents=True)
        best.write_bytes(b"stub checkpoint for orchestration tests only")
        last = best.with_name("last.pt")
        last.write_bytes(b"different final stub checkpoint")
        self.trainer = SimpleNamespace(
            best=best, last=last, device="cpu", epoch=42,
            fitness=1.0, best_fitness=1.0,
            stopper=SimpleNamespace(best_epoch=43),
        )
        for callback in self.callbacks.get("on_model_save", []):
            callback(self.trainer)
        self.trainer.epoch = 99
        self.trainer.fitness = 0.5
        for callback in self.callbacks.get("on_model_save", []):
            callback(self.trainer)
        return example_metric()

    def val(self, **kwargs):
        self.validation_kwargs = kwargs
        result = example_metric()
        result.save_dir = Path(kwargs["project"]) / kwargs["name"]
        result.save_dir.mkdir(parents=True)
        return result


def completed_fixture(config, manifest):
    stub = OrchestrationStub()
    summary = run_training(
        config, model_factory=lambda _: stub,
        environment_provider=lambda: {"torch_version": "stub", "cuda_available": False},
        dataset_loader=lambda _: manifest,
    )
    return stub, summary


def test_versioned_baseline_and_environment_base_override(monkeypatch):
    config = load_training_config()
    assert (config.epochs, config.imgsz, config.batch, config.patience, config.seed) == (100, 640, 8, 20, 42)
    assert config.device == "0" and config.workers == 0 and config.optimizer == "auto"
    monkeypatch.setenv("DAMAGE_BASE_MODEL", "yolo26s-seg.pt")
    assert load_training_config().base_model == "yolo26s-seg.pt"


@pytest.mark.parametrize("model", ["yolo26n-seg.yaml", "scratch.yml", "weights.onnx"])
def test_config_rejects_scratch_or_non_pt_models(prepared, model):
    with pytest.raises(ValueError, match="pretrained .pt"):
        replace(prepared[0], base_model=model)


def test_dataset_validation_precedes_any_library_or_model_loading(prepared):
    config, _ = prepared
    calls = []
    def reject(_):
        raise ValueError("invalid prepared dataset")
    with pytest.raises(ValueError, match="invalid prepared"):
        run_training(config, dataset_loader=reject, model_factory=lambda _: calls.append("model"), environment_provider=lambda: calls.append("environment"))
    assert calls == [] and not config.output_dir.exists()


def test_training_stub_preserves_train_val_selection_and_exact_checkpoint(prepared):
    config, manifest = prepared
    stub, summary = completed_fixture(config, manifest)
    assert stub.training_kwargs["data"] == str(config.dataset_yaml)
    assert stub.training_kwargs["pretrained"] is True and stub.training_kwargs["val"] is True
    assert stub.training_kwargs["split"] == "val" and "test" not in stub.training_kwargs
    assert stub.training_kwargs["epochs"] == 100 and stub.training_kwargs["seed"] == 42
    assert stub.training_kwargs["plots"] is True
    assert summary["checkpoint_selection_split"] == "val"
    assert summary["best_checkpoint"] == "train/weights/best.pt"
    assert summary["best_epoch"] == 43
    assert summary["last_checkpoint"] == "train/weights/last.pt"
    assert summary["last_checkpoint_sha256"] != summary["best_checkpoint_sha256"]
    assert summary["test_used_during_training"] is False
    assert summary["early_stopping_triggered"] is False
    assert summary["epochs_completed"] == 100
    assert summary["environment"]["device_used"] == "cpu"
    assert summary["best_checkpoint_sha256"] == file_sha256(config.output_dir / summary["best_checkpoint"])
    assert len(list(config.output_dir.rglob("best.pt"))) == 1
    assert json.loads((config.output_dir / "training_progress.json").read_text())["status"] == "completed"


def test_best_epoch_tracks_actual_checkpoint_write_on_fitness_tie(prepared):
    config, manifest = prepared

    class TiedFitnessStub(OrchestrationStub):
        def train(self, **kwargs):
            result = super().train(**kwargs)
            self.trainer.epoch = 66
            self.trainer.fitness = self.trainer.best_fitness
            for callback in self.callbacks["on_model_save"]:
                callback(self.trainer)
            self.trainer.epoch = 99
            return result

    summary = run_training(config, model_factory=lambda _: TiedFitnessStub(), environment_provider=lambda: {}, dataset_loader=lambda _: manifest)
    assert summary["best_epoch"] == 67
    assert summary["early_stopping_best_epoch"] == 43


def test_existing_training_run_rejected_before_loading_model(prepared):
    config, manifest = prepared
    config.output_dir.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_training(config, model_factory=lambda _: pytest.fail("must not load"), environment_provider=lambda: pytest.fail("must not import"), dataset_loader=lambda _: manifest)


def test_base_checkpoint_task_must_be_segmentation(prepared):
    config, manifest = prepared
    with pytest.raises(ValueError, match="segmentation"):
        run_training(config, model_factory=lambda _: SimpleNamespace(task="detect"), environment_provider=lambda: {}, dataset_loader=lambda _: manifest)
    assert not config.output_dir.exists()


def interrupted_fixture(config, manifest):
    directory = config.output_dir / "train/weights"
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(b"selected best at epoch 10")
    (directory / "last.pt").write_bytes(b"resumable state at epoch 12")
    base = config.output_dir.parent / "original_pretrained.pt"
    base.write_bytes(b"original pretrained fixture")
    metadata = {
        "config": config.payload(), "dataset": copy.deepcopy(dataset_provenance(config.dataset_yaml, manifest)),
        "confidence_threshold": 0.25, "checkpoint_selection_split": "val",
        "epochs_completed": 12, "best_epoch": 10, "elapsed_seconds": 123.5,
        "base_checkpoint": str(base), "base_checkpoint_sha256": file_sha256(base),
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
            "epoch": 11, "optimizer": {"fixture": "retained"}, "best_fitness": 1.0,
            "stopper": {"best_fitness": 1.0, "best_epoch": 8},
            "train_args": {
                **{key: getattr(config, key) for key in ("epochs", "imgsz", "batch", "patience", "seed", "workers")},
                "optimizer": "AdamW", "task": "segment", "split": "val", "val": True, "pretrained": True,
                "data": str(config.dataset_yaml), "project": str(config.output_dir), "name": "train",
            },
        }

    def train(self, **kwargs):
        self.training_kwargs = kwargs
        self.trainer = SimpleNamespace(
            best=self.config.output_dir / "train/weights/best.pt",
            last=self.config.output_dir / "train/weights/last.pt",
            start_epoch=12, epochs=100, epoch=12, device="cpu", fitness=0.8,
            best_fitness=0.0, stopper=SimpleNamespace(best_fitness=0.0, best_epoch=0),
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


def test_resume_retains_original_best_epoch_optimizer_stopper_and_total_budget(prepared):
    config, manifest = prepared
    metadata = interrupted_fixture(config, manifest)
    stub = ResumeStub(config, metadata)
    loaded = []
    def factory(path):
        loaded.append(path)
        return stub
    summary = run_training(config, model_factory=factory, environment_provider=lambda: {}, dataset_loader=lambda _: manifest, resume=True)
    last = str(config.output_dir / "train/weights/last.pt")
    assert loaded == [last] and stub.training_kwargs["resume"] == last
    assert stub.training_kwargs["save_dir"] == str(config.output_dir / "train")
    assert stub.training_kwargs["epochs"] == 100 and summary["epochs_completed"] == 100
    assert summary["best_epoch"] == 10 and summary["early_stopping_best_epoch"] == 8
    assert stub.trainer.best_fitness == stub.trainer.stopper.best_fitness == 1.0
    assert summary["base_checkpoint_sha256"] == metadata["base_checkpoint_sha256"]
    assert summary["best_checkpoint_sha256"] == metadata["best_checkpoint_sha256"]
    assert summary["elapsed_seconds_before_resume"] == 123.5 and summary["elapsed_seconds"] >= 123.5
    assert summary["resumed"] is True and summary["resume_from_epoch"] == 12
    assert not (config.output_dir / "train2").exists()


@pytest.mark.parametrize("change", ["config", "dataset", "threshold", "best", "last", "base", "completed", "evaluated"])
def test_resume_refuses_changed_or_completed_baseline_before_model_loading(prepared, change):
    config, manifest = prepared
    metadata = interrupted_fixture(config, manifest)
    if change == "config":
        metadata["config"]["seed"] = 999
    elif change == "dataset":
        metadata["dataset"]["annotation_hashes"]["val"] = "f" * 64
    elif change == "threshold":
        metadata["confidence_threshold"] = 0.5
    elif change in ("best", "last"):
        (config.output_dir / f"train/weights/{change}.pt").write_bytes(b"changed")
    elif change == "base":
        Path(metadata["base_checkpoint"]).write_bytes(b"changed")
    elif change == "completed":
        (config.output_dir / "training_summary.json").write_text("{}")
    else:
        (config.output_dir / "evaluation").mkdir()
    (config.output_dir / "recovery_training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Resume"):
        run_training(config, model_factory=lambda _: pytest.fail("must not load"), environment_provider=lambda: pytest.fail("must not import"), dataset_loader=lambda _: manifest, resume=True)


def test_resume_rejects_stripped_checkpoint_without_optimizer(prepared):
    config, manifest = prepared
    stub = ResumeStub(config, interrupted_fixture(config, manifest))
    stub.ckpt["optimizer"] = None
    with pytest.raises(ValueError, match="optimizer state"):
        run_training(config, model_factory=lambda _: stub, environment_provider=lambda: {}, dataset_loader=lambda _: manifest, resume=True)
    assert stub.training_kwargs is None


def test_test_evaluation_uses_only_exact_val_selected_best(prepared):
    config, manifest = prepared
    completed_fixture(config, manifest)
    loaded = []
    stub = OrchestrationStub()
    def load(path):
        loaded.append(path)
        return stub
    result = run_evaluation(config, model_factory=load, environment_provider=lambda: {}, dataset_loader=lambda _: manifest, max_error_examples=0)
    assert loaded == [str(config.output_dir / "train/weights/best.pt")]
    assert stub.validation_kwargs["split"] == "test" and stub.validation_kwargs["conf"] == 0.25
    assert result["split"] == "test" and result["checkpoint_selection_split"] == "val"
    assert json.loads((config.output_dir / "test_metrics.json").read_text())["mask"]["map50"] == 0.6
    assert (config.output_dir / "per_class_metrics.csv").is_file()
    assert result["per_class"][0]["mask"]["map50"] is None


@pytest.mark.parametrize("change", ["checkpoint_bytes", "test_selected", "alternate_path", "annotation_hash", "manifest_bytes", "yaml_bytes"])
def test_evaluation_rejects_provenance_changes_before_model_load(prepared, change):
    config, manifest = prepared
    _, summary = completed_fixture(config, manifest)
    if change == "checkpoint_bytes":
        (config.output_dir / "train/weights/best.pt").write_bytes(b"changed")
    elif change == "test_selected":
        summary["checkpoint_selection_split"] = "test"
    elif change == "alternate_path":
        summary["best_checkpoint"] = "train/weights/last.pt"
    elif change == "annotation_hash":
        manifest = copy.deepcopy(manifest)
        manifest["annotation_hashes"]["test"] = "f" * 64
    elif change == "manifest_bytes":
        (config.dataset_yaml.parent / "cardd_dataset_manifest.json").write_text("{}")
    else:
        config.dataset_yaml.write_text("{}")
    (config.output_dir / "training_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        run_evaluation(config, dataset_loader=lambda _: manifest, model_factory=lambda _: pytest.fail("must not load"), environment_provider=lambda: pytest.fail("must not import"), max_error_examples=0)


def test_threshold_change_is_not_selected_using_test(prepared, monkeypatch):
    config, manifest = prepared
    completed_fixture(config, manifest)
    monkeypatch.setenv("DAMAGE_CONFIDENCE_THRESHOLD", "0.5")
    with pytest.raises(ValueError, match="never select it on TEST"):
        run_evaluation(config, dataset_loader=lambda _: manifest, model_factory=lambda _: pytest.fail("must not load"), environment_provider=lambda: {}, max_error_examples=0)


def test_metric_parser_does_not_fill_missing_classes_with_global_map(prepared):
    _, manifest = prepared
    parsed = extract_segmentation_metrics(example_metric(), manifest["category_mapping"], manifest["splits"]["test"])
    assert parsed["per_class"][0]["box"]["map50_95"] is None
    assert not parsed["per_class"][0]["mask_metrics_available"]
    assert parsed["per_class"][1]["mask"]["map50_95"] == 0.4
    assert parsed["weakest_classes_by_mask_map50_95"] == [{"name": "scratch", "mask_map50_95": 0.4, "annotation_count": 2}]
    assert parsed["unavailable_classes"] == ["dent"]


def test_missing_metric_groups_and_nonfinite_values_remain_null(prepared):
    _, manifest = prepared
    result = SimpleNamespace(box=SimpleNamespace(mp=float("nan"), mr=None))
    parsed = extract_segmentation_metrics(result, manifest["category_mapping"], {})
    assert set(parsed["box"].values()) == {None}
    assert set(parsed["mask"].values()) == {None}


def test_metric_counts_use_converted_labels_and_preserve_source_distribution(prepared):
    _, manifest = prepared
    counts = dict(manifest["splits"]["test"], converted_annotations_per_category={"dent": 0, "scratch": 1}, converted_images_per_category={"dent": 0, "scratch": 1})
    parsed = extract_segmentation_metrics(example_metric(), manifest["category_mapping"], counts)
    assert parsed["per_class"][1]["annotation_count"] == 1
    assert parsed["per_class"][1]["source_annotation_count"] == 2


def test_invalid_metric_value_is_not_saved_as_a_score(prepared):
    _, manifest = prepared
    with pytest.raises(ValueError, match="outside"):
        extract_segmentation_metrics(SimpleNamespace(box=SimpleNamespace(mp=1.4)), manifest["category_mapping"], {})


def test_class_names_order_must_match_checkpoint_taxonomy(prepared):
    _, manifest = prepared
    verify_class_names({0: "dent", 1: "scratch"}, manifest["category_mapping"])
    with pytest.raises(ValueError, match="class names/order"):
        verify_class_names({0: "scratch", 1: "dent"}, manifest["category_mapping"])


def test_environment_records_observed_cpu_without_claiming_cuda():
    torch = SimpleNamespace(__version__="observed-test", version=SimpleNamespace(cuda=None), cuda=SimpleNamespace(is_available=lambda: False))
    observed = observe_environment(torch, SimpleNamespace(__version__="observed-ultralytics"))
    assert observed["torch_version"] == "observed-test"
    assert observed["cuda_available"] is False and observed["cuda_devices"] == []
    assert observed["cuda_device_properties"] == []


def test_environment_records_actual_cuda_runtime_and_vram():
    cuda = SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_name=lambda _: "Observed test GPU",
        get_device_properties=lambda _: SimpleNamespace(total_memory=8 * 1024 ** 3),
        get_device_capability=lambda _: (8, 6),
    )
    observed = observe_environment(SimpleNamespace(__version__="test+cuda", version=SimpleNamespace(cuda="13.2"), cuda=cuda), SimpleNamespace(__version__="test"))
    assert observed["cuda_available"] is True and observed["cuda_runtime_version"] == "13.2"
    assert observed["cuda_device_properties"][0]["total_memory_gib"] == 8.0
    assert observed["cuda_device_properties"][0]["compute_capability"] == [8, 6]


def test_builtin_image_logging_callbacks_removed_without_global_settings():
    def raw_image_logger(_):
        pytest.fail("raw images must not be logged")
    raw_image_logger.__module__ = "ultralytics.utils.callbacks.mlflow"
    def harmless(_):
        pass
    trainer = SimpleNamespace(callbacks={"on_pretrain_routine_end": [raw_image_logger, harmless], "on_train_end": [raw_image_logger]})
    disable_builtin_mlflow_callbacks(trainer)
    assert trainer.callbacks == {"on_pretrain_routine_end": [harmless], "on_train_end": []}


def test_tracking_off_without_configuration_needs_no_mlflow(prepared):
    with vision_tracking_run(prepared[0]) as tracker:
        assert tracker is None


def test_tracking_rejects_image_artifacts(tmp_path):
    calls = []
    tracker = VisionMLflowTracking(SimpleNamespace(log_artifact=lambda *args, **kwargs: calls.append(args)), "stub", log_checkpoint=False)
    tracker.log_metadata_artifacts([tmp_path / "metrics.json"])
    with pytest.raises(ValueError, match="no CarDD images"):
        tracker.log_metadata_artifacts([tmp_path / "raw_car.jpg"])
    assert len(calls) == 1


def test_error_matching_is_one_to_one_classwise_and_keeps_low_confidence():
    truth = [ErrorInstance(0, (0, 0, 0.5, 0.5)), ErrorInstance(1, (0.5, 0.5, 1, 1))]
    predictions = [ErrorInstance(0, (0, 0, 0.5, 0.5), 0.9), ErrorInstance(0, (0, 0, 0.5, 0.5), 0.8), ErrorInstance(1, (0.5, 0.5, 1, 1), 0.1)]
    result = analyze_instances(predictions, truth, confidence_threshold=0.25)
    assert len(result["matches"]) == 1
    assert result["false_positives"] == [1] and result["false_negatives"] == [1]
    assert result["low_confidence"] == [2]
    assert box_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1


def test_wrong_class_is_fp_and_fn_not_a_match():
    result = analyze_instances([ErrorInstance(1, (0, 0, 1, 1), 0.9)], [ErrorInstance(0, (0, 0, 1, 1))], confidence_threshold=0.25)
    assert result["false_positives"] == [0] and result["false_negatives"] == [0]


def test_error_examples_follow_manifest_nested_paths(prepared):
    from PIL import Image
    config, manifest = prepared
    root = config.dataset_yaml.parent
    image = root / "images/test/nested/tiny.png"
    label = root / "labels/test/nested/tiny.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(image)
    label.write_text("1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")
    manifest["prepared_files"] = {"test": [{"image": "images/test/nested/tiny.png", "label": "labels/test/nested/tiny.txt"}]}
    paths = []
    def predict(**kwargs):
        paths.append(kwargs["source"])
        return [SimpleNamespace(boxes=None)]
    analysis = save_error_examples(SimpleNamespace(predict=predict), config, manifest, 0.25, max_examples=4)
    assert paths == [str(image)] and analysis["examples_saved"] == 1
    payload = json.loads((config.output_dir / "error_examples/metadata.json").read_text())
    assert payload["examples"][0]["kind"] == "false_negatives"
    assert len(list((config.output_dir / "error_examples").glob("*.jpg"))) == 1
