"""Prepare controlled portfolio snapshots without ML/GPU or an evaluation run."""

import argparse
from pathlib import Path

from claims.review.demo.preparation import prepare_demo
from claims.review.demo.provider import DEFAULT_DEMO_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument(
        "--observations",
        type=Path,
        action="append",
        default=[],
        help="Existing real benchmark run directory or observation JSON; imports saved results without inference.",
    )
    args = parser.parse_args()
    manifest = prepare_demo(args.output_dir, args.observations)
    print(f"Prepared {len(manifest['claims'])} claims in {args.output_dir.resolve()}")
    for entry in manifest["claims"]:
        print(f"  {entry['claim_id']}: {entry['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
