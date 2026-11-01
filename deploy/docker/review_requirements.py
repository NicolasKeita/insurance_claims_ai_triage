"""Select the archive API's existing constraints from the application pyproject."""

import re
import sys
from pathlib import Path

import tomllib

RUNTIME_PACKAGES = {"fastapi", "pydantic", "sqlalchemy"}


def review_requirements(project):
    dependencies = project["project"]["dependencies"]
    selected = {}
    for requirement in dependencies:
        name = re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().casefold()
        if name in RUNTIME_PACKAGES:
            if name in selected:
                raise ValueError(f"Duplicate runtime dependency: {name}")
            selected[name] = requirement
    if selected.keys() != RUNTIME_PACKAGES:
        raise ValueError("Archive runtime dependencies must exist in pyproject.toml")
    if "[standard]" not in selected["fastapi"]:
        raise ValueError("The existing fastapi[standard] dependency must supply Uvicorn")
    return [selected[name] for name in sorted(selected)]


if __name__ == "__main__":
    with Path(sys.argv[1]).open("rb") as stream:
        print("\n".join(review_requirements(tomllib.load(stream))))
