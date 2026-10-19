"""One validated planning step with action-specific arguments."""

from pydantic import Field, model_validator

from agent.models.base import (
    AgentAction,
    AgentToolName,
    EmptyToolInput,
    KnowledgeSearchInput,
    SimilarClaimsInput,
    StrictModel,
)


class AgentPlanStep(StrictModel):
    action: AgentAction
    reason: str = Field(
        min_length=1, max_length=300, description="Short justification for the selected action, not private reasoning."
    )
    arguments: EmptyToolInput | SimilarClaimsInput | KnowledgeSearchInput = Field(
        description="Required for every action: {} for GET/FINISH, limit for similar, query and limit for search."
    )

    def validated_arguments(self):
        schema = EmptyToolInput
        if self.action == AgentAction.FIND_SIMILAR_CLAIMS:
            schema = SimilarClaimsInput
        elif self.action in (AgentAction.SEARCH_POLICY, AgentAction.SEARCH_PROCEDURE):
            schema = KnowledgeSearchInput
        return schema.model_validate(self.arguments.model_dump())

    @model_validator(mode="after")
    def action_arguments(self):
        self.arguments = self.validated_arguments()
        return self

    @property
    def tool_name(self) -> AgentToolName | None:
        return None if self.action == AgentAction.FINISH else AgentToolName(self.action.value)
