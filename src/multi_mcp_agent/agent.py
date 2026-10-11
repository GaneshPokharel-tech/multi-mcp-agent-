import asyncio

import json

import queue

import subprocess

import threading

from pathlib import Path


import os




from dotenv import load_dotenv

from langchain.mcp import MCPAdapter

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_google_genai.chat_models import GoogleRateLimitError





from langgraph.types import Command



from multi_mcp_agent.agents.registry import create_agent_registry
from multi_mcp_agent.core.persistence import create_checkpointer_context
from multi_mcp_agent.graph.builder import build_multi_agent_graph

# =========================================================

# Environment

# =========================================================


load_dotenv()


# =========================================================

# Paths

# =========================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


MCP_SERVER_PATH = PROJECT_ROOT / "src" / "multi_mcp_agent" / "mcp_server.py"


CHECKPOINT_DB = PROJECT_ROOT / ".langgraph_checkpoints.sqlite"


# =========================================================

# Runtime Settings

# =========================================================


MODEL_NAME = "gemini-3.5-flash-lite"


MODEL_TIMEOUT_SECONDS = 35

TURN_TIMEOUT_SECONDS = 90


MODEL_MAX_ATTEMPTS = 3

RETRY_BASE_DELAY = 0.75


DEFAULT_TIMEZONE = os.getenv(
    "CALENDAR_TIMEZONE",
    "Asia/Kathmandu",
)


# =========================================================

# Message Helpers

# =========================================================


def message_to_text(content) -> str:

    if isinstance(content, str):

        return content

    if isinstance(content, list):

        parts = []

        for block in content:

            if not isinstance(block, dict):

                continue

            text = block.get("text")

            if isinstance(text, str) and text:

                parts.append(text)

        return "\n".join(parts)

    if content is None:

        return ""

    return str(content)


# =========================================================

# Tool Result Helpers

# =========================================================


def parse_tool_result(content):

    if isinstance(content, dict):

        return content

    if isinstance(content, str):

        try:

            result = json.loads(content)

            if isinstance(result, dict):

                return result

        except json.JSONDecodeError:

            return None

    if isinstance(content, list):

        for block in content:

            if not isinstance(block, dict):

                continue

            if isinstance(
                block.get("json"),
                dict,
            ):

                return block["json"]

            if isinstance(
                block.get("text"),
                str,
            ):

                result = parse_tool_result(block["text"])

                if result:

                    return result

    return None


def draft_was_created(
    message: ToolMessage,
) -> bool:

    name = getattr(message, "name", "") or ""

    if not name.endswith("create_draft"):

        return False

    result = parse_tool_result(message.content)

    if result:

        return result.get("success") is True

    text = str(message.content).lower()

    return '"success": true' in text or "'success': true" in text


# =========================================================

# Notification

# =========================================================


def notify_draft_created():

    subprocess.run(
        [
            "osascript",
            "-e",
            (
                "display notification "
                '"Gmail draft created successfully." '
                'with title "Multi MCP Agent"'
            ),
        ],
        check=False,
    )


# =========================================================

# Error Handling

# =========================================================


def is_transient_error(
    error: Exception,
) -> bool:

    text = str(error).lower()

    markers = [
        "503",
        "502",
        "504",
        "unavailable",
        "high demand",
        "temporarily unavailable",
        "deadline exceeded",
        "connection reset",
        "connection aborted",
        "connection error",
        "timeout",
        "timed out",
    ]

    return any(marker in text for marker in markers)


def classify_error(
    error: Exception,
) -> dict:

    if isinstance(
        error,
        GoogleRateLimitError,
    ):

        return {
            "kind": "quota",
            "message": ("Gemini API quota or rate limit " "has been reached."),
        }

    if isinstance(
        error,
        (
            asyncio.TimeoutError,
            TimeoutError,
        ),
    ):

        return {
            "kind": "timeout",
            "message": ("The request took too long " "and timed out."),
        }

    text = str(error)

    lowered = text.lower()

    if "429" in lowered or "resource_exhausted" in lowered or "quota" in lowered:

        return {
            "kind": "quota",
            "message": ("Gemini API quota or rate limit " "has been reached."),
        }

    if "503" in lowered or "unavailable" in lowered or "high demand" in lowered:

        return {
            "kind": "unavailable",
            "message": (
                "Gemini is temporarily unavailable. " "Please try again shortly."
            ),
        }

    return {
        "kind": "error",
        "message": text,
    }


# =========================================================

# MCP Configuration

# =========================================================


def get_mcp_config():

    return {
        "mcpServers": {
            "multi": {
                "command": "uv",
                "args": [
                    "run",
                    "mcp",
                    "run",
                    str(MCP_SERVER_PATH),
                ],
            }
        }
    }


# =========================================================

# Agent Runtime

# =========================================================


class AgentRuntime:

    def __init__(self):

        self.loop = asyncio.new_event_loop()

        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
        )

        self.thread.start()

        future = asyncio.run_coroutine_threadsafe(
            self._initialize(),
            self.loop,
        )

        future.result()

    # -----------------------------------------------------

    # Event Loop

    # -----------------------------------------------------

    def _run_loop(self):

        asyncio.set_event_loop(self.loop)

        self.loop.run_forever()

    # -----------------------------------------------------

    # Initialize

    # -----------------------------------------------------

    async def _initialize(self):

        # MCP
        self.adapter = MCPAdapter(
            get_mcp_config()
        )

        await self.adapter.__aenter__()

        self.tools = await self.adapter.list_tools()

        # Gemini
        self.model = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
        )

        # Multi-agent registry
        self.registry = create_agent_registry(
            model=self.model,
            all_tools=self.tools,
        )

        # Persistent SQLite checkpointer
        self.checkpointer_context = (
            create_checkpointer_context()
        )

        self.checkpointer = await (
            self.checkpointer_context.__aenter__()
        )

        # Multi-agent LangGraph
        self.graph = build_multi_agent_graph(
            registry=self.registry,
            all_tools=self.tools,
            checkpointer=self.checkpointer,
        )

    # -----------------------------------------------------

    # Tool Count

    # -----------------------------------------------------

    @property
    def tool_count(self):

        return len(self.tools)

    # -----------------------------------------------------

    # Direct MCP Tool Call

    # -----------------------------------------------------

    def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ):

        future = asyncio.run_coroutine_threadsafe(
            self._call_tool(
                tool_name,
                arguments,
            ),
            self.loop,
        )

        return future.result()

    async def _call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ):

        selected_tool = None

        for tool in self.tools:

            name = getattr(
                tool,
                "name",
                "",
            )

            if name == tool_name or name.endswith(tool_name):

                selected_tool = tool

                break

        if selected_tool is None:

            raise RuntimeError(f"MCP tool not found: {tool_name}")

        result = await selected_tool.ainvoke(arguments)

        parsed = parse_tool_result(result)

        return parsed if parsed is not None else result

    # -----------------------------------------------------

    # Load Messages

    # -----------------------------------------------------

    def load_messages(
        self,
        thread_id: str,
    ):

        future = asyncio.run_coroutine_threadsafe(
            self._load_messages(thread_id),
            self.loop,
        )

        return future.result()

    async def _load_messages(
        self,
        thread_id: str,
    ):

        config = {
            "configurable": {
                "thread_id": (thread_id),
            }
        }

        checkpoint = await self.checkpointer.aget(config)

        if not checkpoint:

            return []

        return checkpoint.get(
            "channel_values",
            {},
        ).get(
            "messages",
            [],
        )

    # -----------------------------------------------------

    # Delete Thread

    # -----------------------------------------------------

    def delete_thread(
        self,
        thread_id: str,
    ):

        future = asyncio.run_coroutine_threadsafe(
            self.checkpointer.adelete_thread(thread_id),
            self.loop,
        )

        return future.result()

    # -----------------------------------------------------

    # Chat History

    # -----------------------------------------------------

    def list_threads(
        self,
        limit: int = 30,
    ):

        future = asyncio.run_coroutine_threadsafe(
            self._list_threads(limit),
            self.loop,
        )

        return future.result()

    async def _list_threads(
        self,
        limit: int,
    ):

        threads = []

        seen = set()

        async for item in self.checkpointer.alist(
            None,
            limit=500,
        ):

            config = item.config or {}

            thread_id = config.get(
                "configurable",
                {},
            ).get("thread_id")

            if not thread_id:

                continue

            if thread_id in seen:

                continue

            seen.add(thread_id)

            checkpoint = item.checkpoint or {}

            messages = checkpoint.get(
                "channel_values",
                {},
            ).get(
                "messages",
                [],
            )

            title = "New chat"

            for message in messages:

                if isinstance(
                    message,
                    HumanMessage,
                ):

                    text = message_to_text(message.content).strip()

                    if text:

                        title = text

                        break

            if len(title) > 42:

                title = title[:39] + "..."

            threads.append(
                {
                    "thread_id": (thread_id),
                    "title": title,
                    "updated_at": (
                        checkpoint.get(
                            "ts",
                            "",
                        )
                    ),
                }
            )

            if len(threads) >= limit:

                break

        return threads

    # -----------------------------------------------------

    # Pending Human Approval

    # -----------------------------------------------------

    def get_pending_approval(
        self,
        thread_id: str,
    ):

        future = asyncio.run_coroutine_threadsafe(
            self._get_pending_approval(thread_id),
            self.loop,
        )

        return future.result()

    async def _get_pending_approval(
        self,
        thread_id: str,
    ):

        config = {
            "configurable": {
                "thread_id": (thread_id),
            }
        }

        snapshot = await self.graph.aget_state(config)

        if not snapshot:

            return None

        tasks = (
            getattr(
                snapshot,
                "tasks",
                None,
            )
            or []
        )

        for task in tasks:

            interrupts = (
                getattr(
                    task,
                    "interrupts",
                    None,
                )
                or []
            )

            for pending in interrupts:

                value = getattr(
                    pending,
                    "value",
                    None,
                )

                if (
                    isinstance(
                        value,
                        dict,
                    )
                    and value.get("type") == "tool_approval"
                ):

                    return value

        return None

    # -----------------------------------------------------

    # Stream New User Turn

    # -----------------------------------------------------

    def stream_turn(
        self,
        user_message: str,
        thread_id: str,
    ):

        event_queue = queue.Queue()

        future = asyncio.run_coroutine_threadsafe(
            self._drive_graph(
                {
                    "messages": [HumanMessage(content=(user_message))],
                    "thread_id": thread_id,
                },
                thread_id,
                event_queue,
            ),
            self.loop,
        )

        yield from self._consume_queue(
            event_queue,
            future,
        )

    # -----------------------------------------------------

    # Resume Human Approval

    # -----------------------------------------------------

    def resume_approval(
        self,
        thread_id: str,
        approved: bool,
    ):

        event_queue = queue.Queue()

        action = "approve" if approved else "reject"

        future = asyncio.run_coroutine_threadsafe(
            self._drive_graph(
                Command(resume={"action": (action)}),
                thread_id,
                event_queue,
            ),
            self.loop,
        )

        yield from self._consume_queue(
            event_queue,
            future,
        )

    # -----------------------------------------------------

    # Queue Consumer

    # -----------------------------------------------------

    def _consume_queue(
        self,
        event_queue,
        future,
    ):

        while True:

            event = event_queue.get()

            yield event

            if event.get("type") in {
                "done",
                "error",
                "approval",
            }:

                break

        try:

            future.result()

        except Exception:

            # Error event is already sent

            # to Streamlit.

            pass

    # -----------------------------------------------------

    # Drive Graph

    # -----------------------------------------------------

    async def _drive_graph(
        self,
        graph_input,
        thread_id: str,
        event_queue,
    ):

        config = {
            "configurable": {
                "thread_id": (thread_id),
            }
        }

        try:

            # -----------------------------------------

            # Existing message count

            # -----------------------------------------

            checkpoint = await self.checkpointer.aget(config)

            previous_messages = []

            if checkpoint:

                previous_messages = checkpoint.get(
                    "channel_values",
                    {},
                ).get(
                    "messages",
                    [],
                )

            previous_count = len(previous_messages)

            # -----------------------------------------

            # Stream Graph

            # -----------------------------------------

            # UI-only execution telemetry.
            # These events do not affect graph routing,
            # state, persistence, or tool execution.
            announced_agent = None
            started_tools = set()
            completed_tools = set()

            if isinstance(graph_input, Command):
                pending_before_resume = await self._get_pending_approval(
                    thread_id
                )

                resume_value = getattr(
                    graph_input,
                    "resume",
                    None,
                )

                resume_action = (
                    str(resume_value.get("action", "")).lower()
                    if isinstance(resume_value, dict)
                    else ""
                )

                event_queue.put(
                    {
                        "type": "activity",
                        "stage": "approval_resume",
                        "action": resume_action,
                        "label": (
                            "Resuming approved action"
                            if resume_action == "approve"
                            else "Applying rejected action"
                        ),
                    }
                )

                # When approval is granted, the original
                # protected tool call already exists in the
                # persisted graph state. Expose it to the UI
                # before ToolNode resumes execution.
                if (
                    resume_action == "approve"
                    and isinstance(
                        pending_before_resume,
                        dict,
                    )
                ):
                    pending_agent = (
                        pending_before_resume.get("agent")
                    )

                    if pending_agent:
                        announced_agent = pending_agent

                        event_queue.put(
                            {
                                "type": "agent",
                                "agent": pending_agent,
                            }
                        )

                    for tool_call in (
                        pending_before_resume.get(
                            "tool_calls",
                            [],
                        )
                    ):
                        tool_name = tool_call.get("name")

                        if not tool_name:
                            continue

                        started_tools.add(tool_name)

                        event_queue.put(
                            {
                                "type": "tool_start",
                                "tool": tool_name,
                                "agent": pending_agent,
                            }
                        )

            else:
                event_queue.put(
                    {
                        "type": "activity",
                        "stage": "routing",
                        "label": "Supervisor routing request",
                    }
                )

            async with asyncio.timeout(TURN_TIMEOUT_SECONDS):

                async for (
                    message_chunk,
                    metadata,
                ) in self.graph.astream(
                    graph_input,
                    config=config,
                    stream_mode="messages",
                ):

                    node_name = metadata.get(
                        "langgraph_node",
                        "",
                    )

                    # ---------------------------------
                    # Tool completion telemetry
                    # ---------------------------------

                    if isinstance(
                        message_chunk,
                        ToolMessage,
                    ):
                        tool_name = getattr(
                            message_chunk,
                            "name",
                            None,
                        )

                        if (
                            tool_name
                            and tool_name
                            not in completed_tools
                        ):
                            completed_tools.add(tool_name)

                            event_queue.put(
                                {
                                    "type": "tool_done",
                                    "tool": tool_name,
                                }
                            )

                        continue

                    if not isinstance(
                        message_chunk,
                        AIMessageChunk,
                    ):

                        continue

                    # Hide internal supervisor output.
                    if node_name not in {
                        "productivity",
                        "developer",
                        "document",
                        "utility",
                    }:

                        continue

                    # ---------------------------------
                    # Specialist telemetry
                    # ---------------------------------

                    if announced_agent != node_name:
                        announced_agent = node_name

                        event_queue.put(
                            {
                                "type": "agent",
                                "agent": node_name,
                            }
                        )

                    # ---------------------------------
                    # Tool-start telemetry
                    # ---------------------------------

                    tool_calls = (
                        getattr(
                            message_chunk,
                            "tool_calls",
                            None,
                        )
                        or []
                    )

                    tool_call_chunks = (
                        getattr(
                            message_chunk,
                            "tool_call_chunks",
                            None,
                        )
                        or []
                    )

                    for tool_call in [
                        *tool_calls,
                        *tool_call_chunks,
                    ]:
                        tool_name = (
                            tool_call.get("name")
                            if isinstance(
                                tool_call,
                                dict,
                            )
                            else None
                        )

                        if (
                            tool_name
                            and tool_name
                            not in started_tools
                        ):
                            started_tools.add(tool_name)

                            event_queue.put(
                                {
                                    "type": "tool_start",
                                    "tool": tool_name,
                                    "agent": node_name,
                                }
                            )

                    # ---------------------------------
                    # User-facing specialist response
                    # ---------------------------------

                    text = message_to_text(
                        message_chunk.content
                    )

                    if text:

                        event_queue.put(
                            {
                                "type": "token",
                                "text": text,
                            }
                        )

            # -----------------------------------------

            # Check whether LangGraph paused

            # for human approval

            # -----------------------------------------

            pending_approval = await self._get_pending_approval(thread_id)

            if pending_approval:

                event_queue.put(
                    {
                        "type": ("approval"),
                        "approval": (pending_approval),
                    }
                )

                return

            # -----------------------------------------

            # Completed State

            # -----------------------------------------

            completed_checkpoint = await self.checkpointer.aget(config)

            if not completed_checkpoint:

                raise RuntimeError("No LangGraph checkpoint " "was created.")

            all_messages = completed_checkpoint.get(
                "channel_values",
                {},
            ).get(
                "messages",
                [],
            )

            new_messages = all_messages[previous_count:]

            # -----------------------------------------

            # Tools Used

            # -----------------------------------------

            tools_used = []

            for message in new_messages:

                tool_calls = getattr(
                    message,
                    "tool_calls",
                    None,
                )

                if not tool_calls:

                    continue

                for tool_call in tool_calls:

                    name = tool_call.get("name")

                    if name and name not in tools_used:

                        tools_used.append(name)

            # -----------------------------------------

            # Gmail Draft Notification

            # -----------------------------------------

            draft_created = False

            for message in new_messages:

                if not isinstance(
                    message,
                    ToolMessage,
                ):

                    continue

                if draft_was_created(message):

                    draft_created = True

                    break

            if draft_created:

                notify_draft_created()

            # -----------------------------------------

            # Final Assistant Response

            # -----------------------------------------

            final_text = ""

            for message in reversed(all_messages):

                if not isinstance(
                    message,
                    AIMessage,
                ):

                    continue

                text = message_to_text(message.content).strip()

                if text:

                    final_text = text

                    break

            event_queue.put(
                {
                    "type": "done",
                    "response": (final_text),
                    "tools_used": (tools_used),
                    "draft_created": (draft_created),
                }
            )

        except Exception as exc:

            error = classify_error(exc)

            event_queue.put(
                {
                    "type": "error",
                    **error,
                }
            )
