"""Shared errors, label maps and line helpers for document extraction."""

import re
from datetime import datetime

from claims.enums import CollisionType, DamageType


class DocumentExtractionError(Exception):
    pass


MONEY_LINE_PATTERN = re.compile(
    r"^(?P<label>[^:]+):\s*" r"(?P<amount>\d+(?:[.,]\d{1,2})?)\s+" r"(?P<currency>[A-Z]{3})$"
)
DAMAGE_LABEL_MAP = {
    "front bumper": DamageType.FRONT_BUMPER,
    "left headlight": DamageType.LEFT_HEADLIGHT,
    "hood": DamageType.HOOD,
}
COLLISION_TYPE_MAP = {
    "front collision": CollisionType.FRONT_COLLISION,
}


def text_lines(extraction):
    return [line.strip() for line in extraction.text.splitlines() if line.strip()]


def find_prefixed_value(lines: list[str], prefix: str) -> str:
    values = set()
    for line in lines:
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            if value:
                values.add(value)
    if len(values) > 1:
        raise DocumentExtractionError(f"Conflicting values for field: {prefix}")
    if values:
        return values.pop()
    raise DocumentExtractionError(f"Missing field: {prefix}")


def find_section(lines: list[str], start: str, stop_prefixes: tuple[str, ...]) -> list[str]:
    try:
        start_index = lines.index(start)
    except ValueError as error:
        raise DocumentExtractionError(f"Missing section: {start}") from error
    values = []
    for line in lines[start_index + 1 :]:
        if line.startswith(stop_prefixes):
            break
        values.append(line)
    return values


def normalize_damage(value: str) -> DamageType:
    try:
        return DAMAGE_LABEL_MAP[value.strip().lower()]
    except KeyError as error:
        raise DocumentExtractionError(f"Unknown damage type: {value}") from error


def normalize_collision_type(value: str) -> CollisionType:
    try:
        return COLLISION_TYPE_MAP[value.strip().lower()]
    except KeyError as error:
        raise DocumentExtractionError(f"Unknown collision type: {value}") from error


def parse_yes_no(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    raise DocumentExtractionError(f"Expected Yes/No value, got: {value}")


def parse_document_date(value: str):
    try:
        return datetime.strptime(value, "%d/%m/%Y").date()
    except ValueError as error:
        raise DocumentExtractionError(f"Invalid document date: {value}") from error


def split_vehicle(value: str) -> tuple[str, str]:
    parts = value.split(maxsplit=1)
    if len(parts) != 2:
        raise DocumentExtractionError(f"Unable to parse vehicle: {value}")
    return parts[0], parts[1]
