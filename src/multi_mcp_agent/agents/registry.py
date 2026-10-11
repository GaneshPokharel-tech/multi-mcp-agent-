from dataclasses import dataclass
from typing import Any

from multi_mcp_agent.agents.base import SpecialistAgent
from multi_mcp_agent.agents.developer import create_developer_agent
from multi_mcp_agent.agents.document import create_document_agent
from multi_mcp_agent.agents.productivity import create_productivity_agent
from multi_mcp_agent.agents.supervisor import create_supervisor
from multi_mcp_agent.agents.utility import create_utility_agent


# =========================================================
# Agent Registry
# =========================================================

@dataclass(frozen=True)
class AgentRegistry:
    supervisor: Any
    specialists: dict[str, SpecialistAgent]

    def get_specialist(
        self,
        name: str,
    ) -> SpecialistAgent:
        specialist = self.specialists.get(name)

        if specialist is None:
            raise ValueError(
                f"Unknown specialist agent: {name}"
            )

        return specialist


# =========================================================
# Registry Builder
# =========================================================

def create_agent_registry(
    *,
    model: Any,
    all_tools: list[Any],
) -> AgentRegistry:
    """
    Build the supervisor and all specialist agents.

    Each specialist receives only the MCP tools assigned
    to its domain.
    """

    specialists = {
        "productivity": create_productivity_agent(
            model=model,
            all_tools=all_tools,
        ),
        "developer": create_developer_agent(
            model=model,
            all_tools=all_tools,
        ),
        "document": create_document_agent(
            model=model,
            all_tools=all_tools,
        ),
        "utility": create_utility_agent(
            model=model,
            all_tools=all_tools,
        ),
    }

    supervisor = create_supervisor(
        model=model,
    )

    return AgentRegistry(
        supervisor=supervisor,
        specialists=specialists,
    )
