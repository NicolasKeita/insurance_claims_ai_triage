"""Support."""

from types import SimpleNamespace

from vision.damage.tracking import VisionMLflowTracking


def tracker_stub():
    params, tags, metrics, artifacts = {}, {}, {}, []
    client = SimpleNamespace(
        log_params=params.update,
        set_tags=tags.update,
        set_tag=lambda key, value: tags.update({key: value}),
        log_metrics=metrics.update,
        log_artifact=lambda path, **kwargs: artifacts.append((path, kwargs)),
    )
    return VisionMLflowTracking(client, "offline-stub", log_checkpoint=False), params, tags, metrics, artifacts


def metric_fixture():
    return {
        "box": {"precision": 0.6, "recall": 0.4},
        "mask": {"map50_95": 0.3},
        "per_class": [
            {
                "name": "glass shatter",
                "damage_type": "glass_shatter",
                "box": {"precision": 0.5},
                "mask": {"map50_95": 0.2},
            },
            {"name": "dent", "damage_type": "dent", "box": {"precision": None}, "mask": {"map50_95": None}},
        ],
    }


def training_summary():
    return {
        "config": {"seed": 42, "base_model": "yolo26n-seg.pt"},
        "environment": {"device_used": "cuda:0"},
        "dataset": {
            "source_name": "CarDD HF mirror",
            "source_repository": "harpreetsahota/CarDD",
            "source_revision": "stub-revision",
            "actual_sample_count": 2816,
            "split_version": 2,
            "official_splits_used": False,
            "split_manifest_sha256": "a" * 64,
            "prepared_manifest_sha256": "b" * 64,
            "dataset_yaml_sha256": "c" * 64,
            "annotation_fields": ["segmentations"],
            "category_mapping": [{"yolo_class_id": 0, "name": "glass shatter"}],
            "annotation_hashes": {"train": "d" * 64},
            "splits": {
                "train": {
                    "image_count": 1971,
                    "annotation_count": 4385,
                    "annotations_per_category": {"glass shatter": 333},
                    "images_per_category": {"glass shatter": 328},
                }
            },
            "metrics_scope": "offline fixture scope",
        },
        "best_checkpoint": "train/weights/best.pt",
        "best_checkpoint_sha256": "e" * 64,
        "confidence_threshold": 0.25,
        "best_epoch": 74,
        "epochs_completed": 94,
        "elapsed_seconds": 12.5,
        "validation_metrics": metric_fixture(),
    }
