"""Test snapshots."""

import pytest
from pydantic import ValidationError

from claims.enums import DamageType
from claims.investigation import (
    INVESTIGATION_POLICY_V3,
)
from claims.investigation.signals import signals_from_multimodal
from tests.domain.investigation.multimodal._assessment import make_visual_assessment


def test_mutated_confidence_cannot_create_an_inconsistent_signal_snapshot():
    visual = make_visual_assessment()
    weak = visual.images[0].evidence[0].model_copy(update={"damage_confidence": 0.1})
    visual.images[0].evidence[0] = weak
    with pytest.raises(ValidationError):
        signals_from_multimodal(visual, INVESTIGATION_POLICY_V3)


def test_mutated_additional_summary_cannot_create_an_inconsistent_signal_snapshot():
    visual = make_visual_assessment(declared=(DamageType.HOOD,))
    visual.consistency.additional_visual_damage.append(DamageType.HOOD)
    with pytest.raises(ValidationError):
        signals_from_multimodal(visual, INVESTIGATION_POLICY_V3)
