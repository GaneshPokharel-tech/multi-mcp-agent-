from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from multi_mcp_agent.core.config import CHECKPOINT_DB


# =========================================================
# LangGraph Checkpointer
# =========================================================

def create_checkpointer_context():
    """
    Create the persistent SQLite checkpointer used by LangGraph.

    The checkpointer stores graph state per conversation thread,
    allowing conversations and interrupted workflows to resume
    across application restarts.
    """
    return AsyncSqliteSaver.from_conn_string(
        str(CHECKPOINT_DB)
    )


# =========================================================
# Thread Configuration
# =========================================================

def thread_config(thread_id: str) -> dict:
    """
    Build the LangGraph configuration used to isolate
    persisted state by conversation thread.
    """
    if not thread_id or not thread_id.strip():
        raise ValueError("thread_id is required")

    return {
        "configurable": {
            "thread_id": thread_id.strip(),
        }
    }
