"""Preview or apply the documented local artifact organization without deleting data."""

import argparse
import json
from pathlib import Path

from persistence.artifact_storage import ROOT
from persistence.artifact_storage.migration import migrate_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply checked moves; default is a read-only preview.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository containing the artifacts directory.")
    args = parser.parse_args()
    result = migrate_artifacts(args.root, apply=args.apply)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
