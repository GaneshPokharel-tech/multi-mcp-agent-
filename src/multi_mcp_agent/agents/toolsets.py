from collections.abc import Iterable
from typing import Any


# =========================================================
# Specialist Tool Boundaries
# =========================================================

PRODUCTIVITY_TOOLS = frozenset(
    {
        # Gmail
        "search_emails",
        "get_email",
        "create_draft",
        "send_email",

        # Google Calendar
        "list_calendar_events",
        "create_calendar_event",
        "update_calendar_event",
        "delete_calendar_event",
    }
)


DEVELOPER_TOOLS = frozenset(
    {
        "git_status",
        "git_log",
        "git_diff",
        "git_show",
        "git_branch",
        "git_add",
        "git_commit",
        "git_create_branch",
        "git_checkout",
    }
)


DOCUMENT_TOOLS = frozenset(
    {
        "index_document",
        "search_documents",
        "list_documents",
        "delete_document",
    }
)


UTILITY_TOOLS = frozenset(
    {
        # Weather
        "get_current_weather",
        "get_weather_forecast",

        # Fetch
        "fetch_url",

        # YouTube
        "play_youtube_song",
    }
)


TOOLSETS = {
    "productivity": PRODUCTIVITY_TOOLS,
    "developer": DEVELOPER_TOOLS,
    "document": DOCUMENT_TOOLS,
    "utility": UTILITY_TOOLS,
}


# =========================================================
# Tool Selection Helpers
# =========================================================

def get_tool_name(tool: Any) -> str:
    """
    Return the LangChain/MCP tool name.
    """
    name = getattr(tool, "name", None)

    if not isinstance(name, str) or not name:
        raise ValueError(
            f"Tool does not expose a valid name: {tool!r}"
        )

    return name


def tools_for_agent(
    tools: Iterable[Any],
    agent_name: str,
) -> list[Any]:
    """
    Return only the MCP tools allowed for a specialist agent.
    """
    allowed_names = TOOLSETS.get(agent_name)

    if allowed_names is None:
        raise ValueError(
            f"Unknown specialist agent: {agent_name}"
        )

    return [
        tool
        for tool in tools
        if get_tool_name(tool) in allowed_names
    ]


def group_tools(
    tools: Iterable[Any],
) -> dict[str, list[Any]]:
    """
    Split discovered MCP tools into specialist-specific groups.
    """
    tool_list = list(tools)

    return {
        agent_name: tools_for_agent(
            tool_list,
            agent_name,
        )
        for agent_name in TOOLSETS
    }


def validate_tool_coverage(
    tools: Iterable[Any],
) -> dict:
    """
    Check whether all discovered MCP tools belong to a known
    specialist and whether any expected tools are missing.
    """
    discovered_names = {
        get_tool_name(tool)
        for tool in tools
    }

    expected_names = set().union(
        *TOOLSETS.values()
    )

    return {
        "discovered_count": len(discovered_names),
        "expected_count": len(expected_names),
        "unassigned": sorted(
            discovered_names - expected_names
        ),
        "missing": sorted(
            expected_names - discovered_names
        ),
    }
