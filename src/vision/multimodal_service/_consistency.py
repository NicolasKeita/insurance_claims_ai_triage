"""Compare domain parts only; quote text is never reparsed by vision."""

from collections.abc import Iterable

from claims.enums import DamageType
from vision.models import ImageVisualAssessment, MultimodalConsistencyAssessment


def _ordered_damage_types(values: set[DamageType]) -> list[DamageType]:
    return [damage for damage in DamageType if damage in values]


def assess_multimodal_consistency(
    images: Iterable[ImageVisualAssessment],
    *,
    declared_damage_types: Iterable[DamageType],
    quoted_damage_types: Iterable[DamageType] = (),
) -> MultimodalConsistencyAssessment:
    """Compare declared, quoted and observed damage without deduplicating photos."""
    declared = {DamageType(item) for item in declared_damage_types}
    quoted = {DamageType(item) for item in quoted_damage_types}
    evidence = [item for image in images for item in image.evidence]
    mapped = {item.mapped_damage_type for item in evidence if item.mapped_damage_type is not None}
    return MultimodalConsistencyAssessment(
        declared_damage_types=_ordered_damage_types(declared),
        quoted_damage_types=_ordered_damage_types(quoted),
        visually_supported_declared_damage=_ordered_damage_types(mapped & declared),
        additional_visual_damage=_ordered_damage_types(mapped - declared),
        visually_supported_quote_damage=_ordered_damage_types(mapped & quoted),
        visual_damage_unmapped=[item for item in evidence if item.mapped_damage_type is None],
        declared_damage_not_visually_confirmed=_ordered_damage_types(declared - mapped),
    )
