from typing import Literal

from langgraph.graph import END

from multi_mcp_agent.core.state import MultiAgentState
from multi_mcp_agent.graph.approval import requires_approval


# =========================================================
# Route Types
# =========================================================

SpecialistRoute = Literal[
    "productivity",
    "developer",
    "document",
    "utility",
]

SupervisorRoute = Literal[
    "productivity",
    "developer",
    "document",
    "utility",
    "unsupported",
]


# =========================================================
# Tool Call Helpers
# =========================================================

def get_tool_calls(
    state: MultiAgentState,
) -> list[dict]:
    """
    Return tool calls requested by the latest AI message.
    """
    messages = state.get("messages", [])

    if not messages:
        return []

    last_message = messages[-1]

    return (
        getattr(
            last_message,
            "tool_calls",
            None,
        )
        or []
    )


# =========================================================
# Supervisor Routing
# =========================================================

def route_from_supervisor(
    state: MultiAgentState,
) -> SupervisorRoute:
    """
    Route to the specialist selected by the supervisor.
    """
    destination = state.get(
        "active_agent"
    )

    allowed = {
        "productivity",
        "developer",
        "document",
        "utility",
    }

    if destination in allowed:
        return destination

    return "unsupported"


# =========================================================
# Specialist Routing
# =========================================================

def route_after_specialist(
    state: MultiAgentState,
):
    """
    Decide what happens after a specialist responds.

    - no tool call -> finish
    - protected tool -> approval
    - safe tool -> tools
    """
    tool_calls = get_tool_calls(state)

    if not tool_calls:
        return END

    if requires_approval(tool_calls):
        return "approval"

    return "tools"


# =========================================================
# Tool Return Routing
# =========================================================

def route_after_tools(
    state: MultiAgentState,
) -> SpecialistRoute:
    """
    Return tool results to the specialist that requested them.
    """
    active_agent = state.get(
        "active_agent"
    )

    allowed = {
        "productivity",
        "developer",
        "document",
        "utility",
    }

    if active_agent not in allowed:
        raise RuntimeError(
            "Tool execution completed without a valid "
            f"active specialist: {active_agent!r}"
        )

    return active_agent
