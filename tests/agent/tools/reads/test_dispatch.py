"""Test dispatch."""

import pytest

from agent.tools import ReadOnlyAgentTools
from tests.agent.tools._evidence import plan
from tests.agent.tools._repositories import configured
from tests.agent.tools._repositories import configured as configured


def test_finish_is_not_executable(configured):
    claim, factory, _, _ = configured
    with pytest.raises(ValueError, match="not an executable tool"):
        ReadOnlyAgentTools(factory).execute(claim.claim_id, plan("FINISH"))
