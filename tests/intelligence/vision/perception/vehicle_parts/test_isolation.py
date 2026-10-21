"""Test isolation."""

import subprocess
import sys


def test_imports_do_not_load_models_frameworks_or_datasets():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vision.parts.detector; import vision.multimodal_service; "
            "assert 'ultralytics' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
