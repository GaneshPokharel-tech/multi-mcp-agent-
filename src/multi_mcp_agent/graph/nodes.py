from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
)

from multi_mcp_agent.agents.base import (
    specialist_messages,
)
from multi_mcp_agent.agents.registry import (
    AgentRegistry,
)
from multi_mcp_agent.agents.supervisor import (
    SUPERVISOR_SYSTEM_PROMPT,
)
from multi_mcp_agent.core.config import (
    DEFAULT_TIMEZONE,
)
from multi_mcp_agent.core.model import (
    invoke_model_with_retry,
)
from multi_mcp_agent.core.state import (
    MultiAgentState,
)


# =========================================================
# Specialist Names
# =========================================================

SPECIALIST_NAMES = {
    "productivity",
    "developer",
    "document",
    "utility",
}


# =========================================================
# Runtime Context
# =========================================================

def build_runtime_context(
    state: MultiAgentState,
) -> str:
    """
    Build dynamic context shared with specialist agents.

    This contains information that must be resolved at runtime
    rather than stored permanently in prompts.
    """
    current_datetime = datetime.now(
        ZoneInfo(DEFAULT_TIMEZONE)
    )

    thread_id = state.get(
        "thread_id",
        "",
    )

    task = state.get(
        "task",
        "",
    )

    handoff_reason = state.get(
        "handoff_reason",
        "",
    )

    return f"""
RUNTIME CONTEXT

Current datetime: {current_datetime.isoformat()}
Current date: {current_datetime.date().isoformat()}
Current timezone: {DEFAULT_TIMEZONE}

Current conversation thread ID: {thread_id}

Assigned task:
{task}

Supervisor handoff reason:
{handoff_reason}

IMPORTANT:
- Treat the current date/time above as authoritative.
- Use exactly the supplied thread ID for document/RAG operations.
- Complete only the assigned specialist task.
""".strip()


# =========================================================
# Node Factory
# =========================================================

def create_graph_nodes(
    registry: AgentRegistry,
) -> dict[str, Any]:
    """
    Create all executable nodes used by the multi-agent graph.
    """

    # -----------------------------------------------------
    # Supervisor
    # -----------------------------------------------------

    async def supervisor_node(
        state: MultiAgentState,
    ):
        messages = state.get(
            "messages",
            [],
        )

        decision = await invoke_model_with_retry(
            registry.supervisor,
            [
                SystemMessage(
                    content=SUPERVISOR_SYSTEM_PROMPT
                ),
                *messages,
            ],
        )

        current_agent = state.get(
            "active_agent",
            "supervisor",
        )

        if current_agent not in {
            "supervisor",
            *SPECIALIST_NAMES,
        }:
            current_agent = "supervisor"

        if decision.destination in SPECIALIST_NAMES:
            next_agent = decision.destination
        else:
            next_agent = "supervisor"

        return {
            "previous_agent": current_agent,
            "active_agent": next_agent,
            "task": decision.task,
            "handoff_reason": decision.reason,
            "tools_used": [],
        }

    # -----------------------------------------------------
    # Specialist Factory
    # -----------------------------------------------------

    def make_specialist_node(
        specialist_name: str,
    ):
        specialist = registry.get_specialist(
            specialist_name
        )

        async def specialist_node(
            state: MultiAgentState,
        ):
            runtime_context = (
                build_runtime_context(state)
            )

            messages = specialist_messages(
                specialist=specialist,
                messages=state.get(
                    "messages",
                    [],
                ),
                runtime_context=runtime_context,
            )

            response = await invoke_model_with_retry(
                specialist.model_with_tools,
                messages,
            )

            return {
                "active_agent": specialist_name,
                "messages": [
                    response
                ],
            }

        return specialist_node

    # -----------------------------------------------------
    # Unsupported
    # -----------------------------------------------------

    async def unsupported_node(
        state: MultiAgentState,
    ):
        return {
            "active_agent": "supervisor",
            "messages": [
                AIMessage(
                    content=(
                        "That request is outside the connected "
                        "capabilities of this agent."
                    )
                )
            ],
        }

    # -----------------------------------------------------
    # Registry
    # -----------------------------------------------------

    return {
        "supervisor": supervisor_node,
        "productivity": make_specialist_node(
            "productivity"
        ),
        "developer": make_specialist_node(
            "developer"
        ),
        "document": make_specialist_node(
            "document"
        ),
        "utility": make_specialist_node(
            "utility"
        ),
        "unsupported": unsupported_node,
    }
