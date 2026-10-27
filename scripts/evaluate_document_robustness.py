"""Run the independent document OCR benchmark serially, without GPU or E2E."""

from evaluation.deterministic.document_robustness.runtime.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
