"""Support."""

from types import SimpleNamespace

from vision.training import DamageTrainingConfig


def error_example_config(tmp_path):
    from PIL import Image

    root = tmp_path / "prepared"
    root.mkdir()
    entries = []
    for name in ("a", "b", "c"):
        Image.new("RGB", (20, 20)).save(root / f"{name}.jpg")
        (root / f"{name}.txt").write_text(
            "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n2 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n",
            encoding="utf-8",
        )
        entries.append({"image": f"{name}.jpg", "label": f"{name}.txt"})
    config = DamageTrainingConfig(
        version="fixture",
        epochs=100,
        imgsz=640,
        batch=1,
        patience=20,
        seed=42,
        device="cpu",
        workers=0,
        optimizer="auto",
        base_model="yolo26n-seg.pt",
        dataset_yaml=root / "dataset.yaml",
        output_dir=tmp_path / "run",
        experiment_name="fixture",
    )
    return root, entries, config


def error_predictions():
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxyn=[[0.1, 0.1, 0.4, 0.4], [0.6, 0.1, 0.9, 0.4], [0.1, 0.6, 0.4, 0.9]],
            cls=[0, 1, 0],
            conf=[0.9, 0.8, 0.1],
        ),
        masks=SimpleNamespace(
            xyn=[
                [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
                [[0.6, 0.1], [0.9, 0.1], [0.9, 0.4], [0.6, 0.4]],
                [[0.1, 0.6], [0.4, 0.6], [0.4, 0.9], [0.1, 0.9]],
            ]
        ),
    )
