"""Regenerate the reference claim's stationery from preserved original content."""

import argparse
import json
from pathlib import Path

import pymupdf

from evaluation.deterministic.document_generation import render_page

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/CLAIM-2026-00001/document_sources.json"


def generate(source=DEFAULT_SOURCE, output_dir=None):
    source = Path(source)
    target = Path(output_dir) if output_dir is not None else source.parent / "documents"
    bodies = json.loads(source.read_text(encoding="utf-8"))
    rendered = {}
    for filename, body in bodies.items():
        if Path(filename).name != filename or not filename.endswith(".pdf"):
            raise ValueError("Document sources must use plain PDF filenames")
        with pymupdf.open() as document:
            render_page(document, body, kind=Path(filename).stem)
            rendered[filename] = document.tobytes(no_new_id=True, garbage=4, deflate=True)
    target.mkdir(parents=True, exist_ok=True)
    for filename, payload in rendered.items():
        (target / filename).write_bytes(payload)
    return tuple(target / filename for filename in rendered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in generate(args.source, args.output_dir):
        print(f"Generated: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
