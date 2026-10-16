"""Physical damage types, deliberately separate from claims.enums.DamageType.

Official baseline names are documented in CarDD's DCN_plus_cfg.py:
https://github.com/CarDD-USTC/CarDD-USTC.github.io/blob/main/code/CarDD_detection/configs/car_damage/DCN_plus_cfg.py
Category IDs are always read from annotations/model names, never assumed here.
"""

from enum import Enum

from claims.enums import DamageType


class VisualDamageType(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    LAMP_BROKEN = "lamp_broken"
    TIRE_FLAT = "tire_flat"


def map_category_name(name: str) -> VisualDamageType:
    """Accept spelling separators only; unsupported classes fail explicitly."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Damage category name must be a nonempty string")
    canonical = "_".join(name.strip().casefold().replace("-", " ").replace("_", " ").split())
    try:
        return VisualDamageType(canonical)
    except ValueError as error:
        raise ValueError(f"Unsupported visual damage category: {name!r}") from error


class VehiclePartType(str, Enum):
    """Visual Carparts categories, separate from the three business damage parts.

    The generic dataset ``object`` class is OTHER and excluded from fusion.
    Model class IDs are read from its names; this enum's order is not an ID map.
    """

    BACK_BUMPER = "back_bumper"
    BACK_DOOR = "back_door"
    BACK_GLASS = "back_glass"
    BACK_LEFT_DOOR = "back_left_door"
    BACK_LEFT_LIGHT = "back_left_light"
    BACK_LIGHT = "back_light"
    BACK_RIGHT_DOOR = "back_right_door"
    BACK_RIGHT_LIGHT = "back_right_light"
    FRONT_BUMPER = "front_bumper"
    FRONT_DOOR = "front_door"
    FRONT_GLASS = "front_glass"
    FRONT_LEFT_DOOR = "front_left_door"
    FRONT_LEFT_LIGHT = "front_left_light"
    FRONT_LIGHT = "front_light"
    FRONT_RIGHT_DOOR = "front_right_door"
    FRONT_RIGHT_LIGHT = "front_right_light"
    HOOD = "hood"
    LEFT_MIRROR = "left_mirror"
    OTHER = "other"
    RIGHT_MIRROR = "right_mirror"
    TAILGATE = "tailgate"
    TRUNK = "trunk"
    WHEEL = "wheel"


CARPARTS_CLASS_TO_PART = {part.value: part for part in VehiclePartType if part is not VehiclePartType.OTHER} | {
    "object": VehiclePartType.OTHER
}

# Intentionally incomplete: neither unknown sides nor unsupported business parts
# may be guessed from a segmentation category.
VEHICLE_PART_TO_DAMAGE = {
    VehiclePartType.FRONT_BUMPER: DamageType.FRONT_BUMPER,
    VehiclePartType.FRONT_LEFT_LIGHT: DamageType.LEFT_HEADLIGHT,
    VehiclePartType.HOOD: DamageType.HOOD,
}


def map_vehicle_part_category_name(name: str) -> VehiclePartType:
    """Require the real dataset's exact class spelling, including ``object``."""
    try:
        return CARPARTS_CLASS_TO_PART[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unsupported Carparts-Seg category: {name!r}") from error


def map_vehicle_part_to_damage(part_type: VehiclePartType) -> DamageType | None:
    return VEHICLE_PART_TO_DAMAGE.get(part_type)
