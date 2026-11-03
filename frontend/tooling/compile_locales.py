"""Validate frontend TOML catalogues with Python's standard library."""

import json
import re
import sys
from pathlib import Path

import tomllib


def read_messages(path):
    with path.open("rb") as source:
        payload = tomllib.load(source)
    if set(payload) != {"messages"} or not isinstance(payload["messages"], dict):
        raise ValueError(f"{path.name}: expected a single [messages] table")
    messages = payload["messages"]
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in messages.items()):
        raise ValueError(f"{path.name}: message keys and values must be strings")
    return messages


def compile_catalogues(root):
    complete = [(root / f"{locale}.toml").is_file() for locale in ("fr", "en")]
    if any(complete) and not all(complete):
        raise ValueError("Both fr.toml and en.toml must exist together")
    result = {}
    for locale in ("fr", "en"):
        paths = [root / f"{locale}.toml"] if all(complete) else sorted(root.glob(f"*.{locale}.toml"))
        if not paths:
            raise ValueError(f"No {locale} TOML catalogue found in {root}")
        merged = {}
        for path in paths:
            for key, value in read_messages(path).items():
                merged[key] = value
        result[locale] = merged
    if result["fr"].keys() != result["en"].keys():
        missing_fr = sorted(result["en"].keys() - result["fr"].keys())
        missing_en = sorted(result["fr"].keys() - result["en"].keys())
        raise ValueError(f"Catalogue key mismatch: missing FR={missing_fr}, missing EN={missing_en}")
    for key, english in result["en"].items():
        if placeholders(english) != placeholders(result["fr"][key]):
            raise ValueError(f"Interpolation placeholders differ for {key!r}")
    return result


def placeholders(text):
    return set(re.findall(r"\{([A-Za-z_][A-Za-z_0-9]*)\}", text))


def main():
    try:
        print(json.dumps(compile_catalogues(Path(sys.argv[1]).resolve()), ensure_ascii=False))
        return 0
    except (OSError, ValueError) as error:
        print(f"Translation catalogue error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
