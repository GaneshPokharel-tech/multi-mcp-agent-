from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# =========================================================
# Agent Names
# =========================================================

AgentName = Literal[
    "supervisor",
    "productivity",
    "developer",
    "document",
    "utility",
]


# =========================================================
# Shared Multi-Agent State
# =========================================================

class MultiAgentState(TypedDict, total=False):
    # Full conversation history.
    # add_messages lets LangGraph append/update messages safely.
    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]

    # Persistent conversation identifier.
    thread_id: str

    # Specialist currently responsible for the task.
    active_agent: AgentName

    # Agent that handled the task before the current agent.
    previous_agent: AgentName

    # Normalized task assigned by the supervisor.
    task: str

    # Why the supervisor routed the request to this agent.
    handoff_reason: str

    # MCP tools used during the current turn.
    tools_used: list[str]
