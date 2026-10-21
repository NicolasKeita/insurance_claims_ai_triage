"""Test converter."""

import subprocess
import sys

from tests.vision.datasets.coco._support import synthetic_cardd
from vision.dataset import (
    discover_cardd_dataset,
    prepare_cardd_yolo,
)


def test_official_converter_is_lazy_and_auto_fallback_is_explicit(tmp_path, monkeypatch):
    import vision.dataset.preparation._converter as dataset_module

    root = synthetic_cardd(tmp_path)

    def unavailable(*args, **kwargs):
        raise ImportError("optional dependency unavailable")

    monkeypatch.setattr(dataset_module, "_try_ultralytics_conversion", unavailable)
    manifest = prepare_cardd_yolo(discover_cardd_dataset(root), tmp_path / "prepared", manifest_path=None)
    assert manifest["conversion"]["backend"] == "native_polygon"
    assert manifest["conversion"]["fallback_reason"].startswith("ImportError")


def test_module_import_does_not_import_ultralytics():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import vision.dataset; assert 'ultralytics' not in sys.modules"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
