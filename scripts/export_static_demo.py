"""Export already computed demo reviews and professional PDFs for a static build."""

import argparse
from pathlib import Path

from evaluation.reporting.static_demo import export_static_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("analysis_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("frontend/public/demo"))
    parser.add_argument("--replace", action="store_true", help="Replace an existing generated demo directory")
    options = parser.parse_args()
    try:
        target = export_static_demo(options.source_dir, options.output_dir, replace=options.replace)
    except (OSError, ValueError, LookupError) as error:
        print(f"Cannot export static demo: {error}")
        return 1
    print(f"Exported static demo: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
