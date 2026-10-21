"""Translate reliable additional visual damage into one review signal."""

from claims.investigation.models import InvestigationSignal, SignalCategory, SignalCode, SignalSeverity
from claims.investigation.policy import DEFAULT_INVESTIGATION_POLICY, InvestigationPolicy
from vision.models import AssociationStatus, ClaimMultimodalVisionAssessment


def signals_from_multimodal(
    visual_assessment: ClaimMultimodalVisionAssessment,
    policy: InvestigationPolicy = DEFAULT_INVESTIGATION_POLICY,
) -> tuple[InvestigationSignal, ...]:
    """Only reliable additional mapped damage supports a moderate review signal.

    Keep every per-photo observation as evidence, but score a business damage
    category once. Positive corroboration and non-detection never add points.

    Frozen models still contain mutable lists, and model_copy can bypass validation. Validate the complete source
    graph before consuming or snapshotting its derived consistency and geometric evidence.
    """
    if not policy.multimodal_enabled:
        return ()
    visual_assessment = ClaimMultimodalVisionAssessment.model_validate(visual_assessment.model_dump(round_trip=True))
    fusion = visual_assessment.fusion_policy
    additional = set(visual_assessment.consistency.additional_visual_damage) - set(
        visual_assessment.consistency.declared_damage_types
    )
    observed = _reliable_observations(visual_assessment, fusion, additional)
    return tuple(
        _damage_signal(visual_assessment, fusion, damage_type, observed[damage_type])
        for damage_type in sorted(observed, key=lambda value: value.value)
    )


def _reliable_observations(visual_assessment, fusion, additional):
    """Group per-photo observations that satisfy the fusion thresholds."""
    observed = {}
    for item in visual_assessment.observations:
        if (
            item.association_status is not AssociationStatus.ASSIGNED
            or item.fusion_policy != fusion
            or item.mapped_damage_type not in additional
            or item.damage_confidence < fusion.min_damage_confidence
            or item.part_confidence is None
            or item.part_confidence < fusion.min_part_confidence
            or item.containment_ratio is None
            or item.containment_ratio < fusion.min_damage_containment
        ):
            continue
        observed.setdefault(item.mapped_damage_type, []).append(item)
    return observed


def _damage_signal(visual_assessment, fusion, damage_type, observations):
    """Build the single evidence-bearing signal for one mapped damage category."""
    damage_names = sorted({item.visual_damage_type.value.lower() for item in observations})
    part_name = damage_type.value.lower().replace("_", " ")
    return InvestigationSignal(
        code=SignalCode.VISUAL_ADDITIONAL_DAMAGE,
        category=SignalCategory.MULTIMODAL_CONSISTENCY,
        severity=SignalSeverity.MEDIUM,
        title="Additional visually observed damage",
        explanation=(
            f"Damage consistent with {', '.join(damage_names)} was detected on the "
            f"{part_name}, which is not listed among the declared damaged parts. "
            "The model observation supports human review and does not establish intent."
        ),
        evidence={
            "claim_id": visual_assessment.claim_id,
            "mapped_damage_type": damage_type.value,
            "declared_damage_types": [item.value for item in visual_assessment.consistency.declared_damage_types],
            "quoted_damage_types": [item.value for item in visual_assessment.consistency.quoted_damage_types],
            "observation_count": len(observations),
            "observations": [item.model_dump(mode="json") for item in observations],
            "damage_model_metadata": visual_assessment.damage_model_metadata.model_dump(mode="json"),
            "part_model_metadata": visual_assessment.part_model_metadata.model_dump(mode="json"),
            "fusion_policy_version": fusion.version,
            "fusion_policy": fusion.model_dump(mode="json"),
        },
    )
