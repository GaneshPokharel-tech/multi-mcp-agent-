from collections.abc import Iterable
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt

from multi_mcp_agent.core.state import MultiAgentState


# =========================================================
# Protected MCP Tools
# =========================================================

APPROVAL_REQUIRED_TOOLS = frozenset(
    {
        # Gmail mutation
        "send_email",

        # Git mutations
        "git_add",
        "git_commit",
        "git_create_branch",
        "git_checkout",

        # Calendar mutations
        "create_calendar_event",
        "update_calendar_event",
        "delete_calendar_event",

        # Document mutation
        "delete_document",
    }
)


# =========================================================
# Approval Helpers
# =========================================================

def protected_tool_calls(
    tool_calls: Iterable[dict],
) -> list[dict]:
    return [
        tool_call
        for tool_call in tool_calls
        if tool_call.get("name") in APPROVAL_REQUIRED_TOOLS
    ]


def requires_approval(
    tool_calls: Iterable[dict],
) -> bool:
    return bool(
        protected_tool_calls(tool_calls)
    )


def get_latest_tool_calls(
    state: MultiAgentState,
) -> list[dict]:
    messages = state.get("messages", [])

    if not messages:
        return []

    return (
        getattr(
            messages[-1],
            "tool_calls",
            None,
        )
        or []
    )


# =========================================================
# Approval Node
# =========================================================

def create_approval_node():
    """
    Create the human-in-the-loop LangGraph approval node.

    If approved, execution continues to the shared tool node.

    If rejected, ToolMessages are added for the pending tool
    calls and execution returns to the specialist that requested
    them.
    """

    def approval_node(
        state: MultiAgentState,
    ) -> Any:
        tool_calls = get_latest_tool_calls(
            state
        )

        protected_calls = protected_tool_calls(
            tool_calls
        )

        active_agent = state.get(
            "active_agent"
        )

        if active_agent not in {
            "productivity",
            "developer",
            "document",
            "utility",
        }:
            raise RuntimeError(
                "Approval requested without a valid "
                f"active specialist: {active_agent!r}"
            )

        decision = interrupt(
            {
                "type": "tool_approval",
                "title": "Human approval required",
                "message": (
                    "A specialist wants to perform a "
                    "protected action that can modify data."
                ),
                "agent": active_agent,
                "tool_calls": [
                    {
                        "name": call.get("name"),
                        "args": call.get(
                            "args",
                            {},
                        ),
                        "id": call.get("id"),
                    }
                    for call in protected_calls
                ],
            }
        )

        action = ""

        if isinstance(
            decision,
            dict,
        ):
            action = str(
                decision.get(
                    "action",
                    "",
                )
            ).lower()

        if action == "approve":
            return Command(
                goto="tools"
            )

        rejected_messages = [
            ToolMessage(
                content=(
                    "The user rejected this tool action. "
                    "Do not execute or retry it unless the "
                    "user explicitly asks again."
                ),
                tool_call_id=call.get("id"),
                name=call.get("name"),
            )
            for call in tool_calls
        ]

        return Command(
            goto=active_agent,
            update={
                "messages": rejected_messages,
            },
        )

    return approval_node
