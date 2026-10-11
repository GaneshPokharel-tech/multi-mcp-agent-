from dataclasses import dataclass
from typing import Any

from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
)

from multi_mcp_agent.agents.toolsets import (
    tools_for_agent,
)


# =========================================================
# Specialist Agent Definition
# =========================================================

@dataclass(frozen=True)
class SpecialistAgent:
    name: str
    system_prompt: str
    tools: list[Any]
    model_with_tools: Any


# =========================================================
# Builder
# =========================================================

def build_specialist_agent(
    *,
    name: str,
    system_prompt: str,
    model: Any,
    all_tools: list[Any],
) -> SpecialistAgent:
    """
    Build one specialist agent with access only to the MCP
    tools assigned to that specialist.
    """
    specialist_tools = tools_for_agent(
        all_tools,
        name,
    )

    if not specialist_tools:
        raise RuntimeError(
            f"No MCP tools were assigned to specialist: {name}"
        )

    model_with_tools = model.bind_tools(
        specialist_tools
    )

    return SpecialistAgent(
        name=name,
        system_prompt=system_prompt.strip(),
        tools=specialist_tools,
        model_with_tools=model_with_tools,
    )


# =========================================================
# Message Builder
# =========================================================

def specialist_messages(
    *,
    specialist: SpecialistAgent,
    messages: list[AnyMessage],
    runtime_context: str = "",
) -> list[AnyMessage]:
    """
    Build the message list sent to a specialist model.
    """
    system_content = specialist.system_prompt

    if runtime_context.strip():
        system_content += (
            "\n\n"
            + runtime_context.strip()
        )

    return [
        SystemMessage(
            content=system_content
        ),
        *messages,
    ]
