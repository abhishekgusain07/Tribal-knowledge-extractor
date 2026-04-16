"""Central LLM factory — all pipeline agents get their LLM instance from here.

Supports two providers:
  1. OpenRouter (default) — set OPENROUTER_API_KEY
  2. Direct Anthropic       — set ANTHROPIC_API_KEY (fallback if no OpenRouter key)

Switching providers only requires changing env vars; no code changes needed.
"""

from __future__ import annotations

import os

from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

# OpenRouter model IDs for Claude
OPENROUTER_SONNET_MODEL = "anthropic/claude-sonnet-4"

# Direct Anthropic model IDs
ANTHROPIC_SONNET_MODEL = "claude-sonnet-4-6-20250514"


def get_llm(
    *,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """Return a configured chat LLM instance.

    Checks OPENROUTER_API_KEY first, then falls back to ANTHROPIC_API_KEY.
    Raises RuntimeError if neither is set.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if openrouter_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=OPENROUTER_SONNET_MODEL,
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            default_headers={
                "HTTP-Referer": os.environ.get(
                    "OPENROUTER_REFERER"
                ),
                "X-Title": os.environ.get(
                    "OPENROUTER_TITLE"
                ),
            },
        )

    if anthropic_key:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=ANTHROPIC_SONNET_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise RuntimeError(
        "No LLM API key found. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY."
    )


def get_structured_llm(
    schema: type[BaseModel],
    *,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> Any:
    """Return an LLM bound to produce structured output matching *schema*.

    Handles the provider difference: OpenRouter needs method="function_calling",
    while direct Anthropic works with the default method.
    """
    llm = get_llm(temperature=temperature, max_tokens=max_tokens)
    use_openrouter = bool(os.environ.get("OPENROUTER_API_KEY", ""))

    if use_openrouter:
        return llm.with_structured_output(schema, method="function_calling")
    return llm.with_structured_output(schema)


def has_llm_key() -> bool:
    """Return True if any supported LLM API key is configured."""
    return bool(
        os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )
