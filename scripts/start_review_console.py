"""Start the prepared snapshot API and Vite together; Ctrl+C stops only these children."""

import argparse
import subprocess
import sys
from pathlib import Path

if __package__:
    from .review_console.runtime import run
else:
    from review_console.runtime import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, help="Prepared snapshots (or CLAIMS_REVIEW_DEMO_DIR)")
    parser.add_argument("--frontend-port", type=int, default=5173)
    options = parser.parse_args()
    try:
        return run(options)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Cannot start Claim Review: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
