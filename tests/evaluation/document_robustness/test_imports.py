"""A fresh interpreter proves document evaluation cannot eagerly load vision or ML."""

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.mark.parametrize(
    "entry",
    (
        "import evaluation.document_robustness",
        "from evaluation.deterministic.document_robustness.runtime import preflight; "
        "preflight._inspect_tesseract=lambda result,command: None; "
        "from evaluation.deterministic.document_robustness.runtime.cli import main; main(['--preflight-only'])",
    ),
)
def test_document_entrypoints_do_not_import_vision_ml(entry):
    script = entry + "; import sys,json; print(json.dumps(sorted(sys.modules)))"
    process = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[3],
        timeout=30,
    )
    modules = json.loads(process.stdout.splitlines()[-1])
    forbidden = ("torch", "ultralytics", "mlflow", "vision")
    assert not any(name == root or name.startswith(root + ".") for name in modules for root in forbidden)


def test_lazy_exports_preserve_document_public_api():
    import evaluation.deterministic as deterministic
    from evaluation.deterministic.documents import extract_case_documents

    assert deterministic.extract_case_documents is extract_case_documents
    assert "assess_vision" in deterministic.__all__
    with pytest.raises(AttributeError):
        _ = deterministic.unknown_stage


@pytest.mark.parametrize("missing", ("pymupdf", "pypdf"))
def test_cli_reports_missing_document_dependency_without_traceback(tmp_path, missing):
    script = dedent("""
        import importlib.abc
        import sys

        class BlockedDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == sys.argv[2]:
                    raise ModuleNotFoundError("controlled missing " + fullname)

        sys.meta_path.insert(0, BlockedDependency())
        from evaluation.deterministic.document_robustness.runtime.cli import main
        raise SystemExit(main(["--variant", "scan_clean", "--scenario", "demo_01", "--document-type",
                               "claim_form", "--output-dir", sys.argv[1]]))
        """)
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), missing],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
        timeout=30,
    )
    assert process.returncode == 2
    assert "OCR benchmark: NOT EXECUTED" in process.stdout
    assert f"controlled missing {missing}" in process.stdout
    assert "Traceback" not in process.stderr
    if missing == "pymupdf":
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert summary["metrics"]["not_executed_count"] == 1
