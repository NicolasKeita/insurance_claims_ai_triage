"""Test configuration."""

from dataclasses import replace

import pytest

from vision.parts_training import (
    load_parts_training_config,
    require_requested_device,
    run_parts_training,
)


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
