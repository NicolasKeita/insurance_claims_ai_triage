"""Physical damage types, deliberately separate from claims.enums.DamageType.

Official baseline names are documented in CarDD's DCN_plus_cfg.py:
https://github.com/CarDD-USTC/CarDD-USTC.github.io/blob/main/code/CarDD_detection/configs/car_damage/DCN_plus_cfg.py
Category IDs are always read from annotations/model names, never assumed here.
"""

from enum import Enum


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
