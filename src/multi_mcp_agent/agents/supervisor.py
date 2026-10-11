from typing import Any, Literal

from pydantic import BaseModel, Field


# =========================================================
# Routing Schema
# =========================================================

SpecialistName = Literal[
    "productivity",
    "developer",
    "document",
    "utility",
]

RouteName = Literal[
    "productivity",
    "developer",
    "document",
    "utility",
    "unsupported",
]


class SupervisorDecision(BaseModel):
    destination: RouteName = Field(
        description="Specialist agent that should handle the request."
    )

    task: str = Field(
        description=(
            "A concise task description for the selected specialist."
        )
    )

    reason: str = Field(
        description=(
            "Short explanation of why this specialist was selected."
        )
    )


# =========================================================
# Supervisor Prompt
# =========================================================

SUPERVISOR_SYSTEM_PROMPT = """
You are the Supervisor Agent for a multi-agent system.

Your only job is to route the user's request to the correct
specialist agent.

Do not perform the task yourself.
Do not call external tools.
Do not invent tool results.


AVAILABLE SPECIALISTS

productivity
- Gmail
- Google Calendar

developer
- Git repository operations

document
- Uploaded PDF documents
- Document indexing
- Document search
- Document inventory
- Document deletion

utility
- Current weather
- Weather forecasts
- Public URL fetching
- YouTube song/video opening


ROUTING RULES

Route Gmail and Calendar requests to:
productivity

Route Git and repository requests to:
developer

Route uploaded PDF or document questions to:
document

Route weather, public URL fetch, and YouTube requests to:
utility

If the request does not belong to any supported capability,
route to:
unsupported


CONVERSATION CONTEXT

Use the recent conversation when resolving follow-up requests.

For example:

User:
What are the three RAG paradigms?

Then:
What are the limitations of the first one?

The second request should remain routed to the document agent.

Do not switch specialists only because a follow-up message is short.


OUTPUT

Return only the structured routing decision requested by the schema.

Keep task and reason concise.
"""


# =========================================================
# Supervisor Builder
# =========================================================

def create_supervisor(
    *,
    model: Any,
):
    """
    Create a supervisor model that returns a validated
    SupervisorDecision instead of free-form text.
    """
    return model.with_structured_output(
        SupervisorDecision
    )
