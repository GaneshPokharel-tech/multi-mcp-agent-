import asyncio
from typing import Any

from langchain_google_genai.chat_models import GoogleRateLimitError

from multi_mcp_agent.core.config import (
    MODEL_MAX_ATTEMPTS,
    MODEL_TIMEOUT_SECONDS,
    RETRY_BASE_DELAY,
)


# =========================================================
# Transient Error Detection
# =========================================================

def is_transient_error(
    error: Exception,
) -> bool:
    text = str(error).lower()

    markers = (
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
    )

    return any(
        marker in text
        for marker in markers
    )


# =========================================================
# Model Invocation With Retry
# =========================================================

async def invoke_model_with_retry(
    model: Any,
    messages: list,
):
    """
    Invoke a LangChain chat model with timeout and retry
    handling for transient provider/network failures.
    """
    last_error: Exception | None = None

    for attempt in range(
        MODEL_MAX_ATTEMPTS
    ):
        try:
            return await asyncio.wait_for(
                model.ainvoke(messages),
                timeout=MODEL_TIMEOUT_SECONDS,
            )

        except GoogleRateLimitError:
            # Quota/rate-limit failures should be surfaced
            # immediately rather than repeatedly retried.
            raise

        except asyncio.TimeoutError as exc:
            last_error = exc

        except Exception as exc:
            if not is_transient_error(exc):
                raise

            last_error = exc

        if attempt < MODEL_MAX_ATTEMPTS - 1:
            delay = (
                RETRY_BASE_DELAY
                * (2 ** attempt)
            )

            await asyncio.sleep(delay)

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "Model request failed."
    )
