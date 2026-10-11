from typing import Any

from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.prebuilt import ToolNode

from multi_mcp_agent.agents.registry import (
    AgentRegistry,
)
from multi_mcp_agent.core.state import (
    MultiAgentState,
)
from multi_mcp_agent.graph.approval import (
    create_approval_node,
)
from multi_mcp_agent.graph.nodes import (
    create_graph_nodes,
)
from multi_mcp_agent.graph.routing import (
    route_after_specialist,
    route_after_tools,
    route_from_supervisor,
)


# =========================================================
# Multi-Agent Graph Builder
# =========================================================

def build_multi_agent_graph(
    *,
    registry: AgentRegistry,
    all_tools: list[Any],
    checkpointer: Any,
):
    """
    Build and compile the complete multi-agent LangGraph.
    """

    nodes = create_graph_nodes(
        registry
    )

    tool_node = ToolNode(
        all_tools
    )

    approval_node = (
        create_approval_node()
    )

    builder = StateGraph(
        MultiAgentState
    )

    # -----------------------------------------------------
    # Nodes
    # -----------------------------------------------------

    builder.add_node(
        "supervisor",
        nodes["supervisor"],
    )

    builder.add_node(
        "productivity",
        nodes["productivity"],
    )

    builder.add_node(
        "developer",
        nodes["developer"],
    )

    builder.add_node(
        "document",
        nodes["document"],
    )

    builder.add_node(
        "utility",
        nodes["utility"],
    )

    builder.add_node(
        "unsupported",
        nodes["unsupported"],
    )

    builder.add_node(
        "approval",
        approval_node,
    )

    builder.add_node(
        "tools",
        tool_node,
    )

    # -----------------------------------------------------
    # Entry
    # -----------------------------------------------------

    builder.add_edge(
        START,
        "supervisor",
    )

    # -----------------------------------------------------
    # Supervisor → Specialist
    # -----------------------------------------------------

    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "productivity": "productivity",
            "developer": "developer",
            "document": "document",
            "utility": "utility",
            "unsupported": "unsupported",
        },
    )

    # -----------------------------------------------------
    # Specialists → Tool / Approval / Finish
    # -----------------------------------------------------

    for specialist_name in (
        "productivity",
        "developer",
        "document",
        "utility",
    ):
        builder.add_conditional_edges(
            specialist_name,
            route_after_specialist,
            {
                "tools": "tools",
                "approval": "approval",
                END: END,
            },
        )

    # -----------------------------------------------------
    # Tool Result → Same Specialist
    # -----------------------------------------------------

    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "productivity": "productivity",
            "developer": "developer",
            "document": "document",
            "utility": "utility",
        },
    )

    # -----------------------------------------------------
    # Unsupported → Finish
    # -----------------------------------------------------

    builder.add_edge(
        "unsupported",
        END,
    )

    # Approval routing is handled dynamically by Command.

    return builder.compile(
        checkpointer=checkpointer
    )
