"""The Docker archive API imports without model, retrieval or database clients."""

import subprocess
import sys


def test_import_does_not_require_any_inference_or_storage_driver_module():
    script = """
import importlib.abc
import sys

forbidden = {
    'agent', 'retrieval', 'vision', 'ml', 'mlflow', 'torch', 'transformers',
    'numpy', 'pandas', 'sklearn', 'psycopg', 'qdrant_client', 'pypdf', 'pymupdf',
}
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockModels())
from api.review_app import app
from persistence.repositories import ClaimNotFoundError
assert all(name.split('.')[0] not in forbidden for name in sys.modules)
assert '/readyz' in {route.path for route in app.routes}
assert '/health' not in {route.path for route in app.routes}
assert ClaimNotFoundError.__name__ == 'ClaimNotFoundError'
print('lightweight archive API imported')
"""
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "lightweight archive API imported" in result.stdout
