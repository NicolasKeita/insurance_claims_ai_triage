"""Publish a complete static demo only after staging and checking all assets."""

import hashlib
import json
import shutil
from pathlib import Path

FORMAT = "claims_static_demo_v1"


def safe_output(path):
    target = Path(path).absolute()
    if any(part.is_symlink() for part in (target, *target.parents)):
        raise ValueError("Static demo path symlinks are not allowed")
    return target


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = path.parent.name == "reviews"
    payload = json.dumps(
        data, indent=None if compact else 2, separators=(",", ":") if compact else None, ensure_ascii=False
    )
    path.write_text(payload + "\n", encoding="utf-8", newline="\n")


def _files(root):
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Static demo output symlinks are not allowed")
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }


def publish(staging, output, replace):
    if output.exists() and any(output.iterdir()):
        manifest = output / "manifest.json"
        if manifest.is_symlink():
            raise ValueError("Static demo output symlinks are not allowed")
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")).get("format") != FORMAT:
            raise ValueError("Output is not a generated static demo; existing files were preserved")
        if _files(staging) == _files(output):
            return output
        if not replace:
            raise FileExistsError("Static demo differs; pass --replace to refresh it")
        if output.name != "demo" or output.resolve().parent != output.parent.resolve():
            raise ValueError("Replacement is restricted to the explicitly selected demo directory")
        shutil.rmtree(output)
    elif output.exists():
        output.rmdir()
    staging.rename(output)
    return output
