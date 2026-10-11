import os
from pathlib import Path

from dotenv import load_dotenv


# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ENV_FILE = PROJECT_ROOT / ".env"

MCP_SERVER_PATH = (
    PROJECT_ROOT
    / "src"
    / "multi_mcp_agent"
    / "mcp_server.py"
)

CHECKPOINT_DB = (
    PROJECT_ROOT
    / ".langgraph_checkpoints.sqlite"
)


# =========================================================
# Environment
# =========================================================

load_dotenv(ENV_FILE)


# =========================================================
# LLM Settings
# =========================================================

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

MODEL_TIMEOUT_SECONDS = int(
    os.getenv("MODEL_TIMEOUT_SECONDS", "35")
)

TURN_TIMEOUT_SECONDS = int(
    os.getenv("TURN_TIMEOUT_SECONDS", "90")
)

MODEL_MAX_ATTEMPTS = int(
    os.getenv("MODEL_MAX_ATTEMPTS", "3")
)

RETRY_BASE_DELAY = float(
    os.getenv("RETRY_BASE_DELAY", "0.75")
)


# =========================================================
# Timezone
# =========================================================

DEFAULT_TIMEZONE = os.getenv(
    "CALENDAR_TIMEZONE",
    "Asia/Kathmandu",
)
