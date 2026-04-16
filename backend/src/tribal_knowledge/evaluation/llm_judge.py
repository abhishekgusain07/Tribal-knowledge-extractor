"""LLM-as-judge for document usefulness scoring.

Tries Gemini Flash first (cheap and fast).  Falls back to a deterministic
heuristic when no ``GOOGLE_API_KEY`` is configured.
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for LLM judge
# ---------------------------------------------------------------------------

LLM_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of developer context documents.

A "context document" is a concise reference that an AI coding agent or a new
developer reads *before* modifying a codebase module.  Great context documents
surface non-obvious patterns, gotchas, and the "why" behind decisions --
they do NOT rehash API signatures that an agent can already read from code.

Rate the following document on a scale from 0.0 (useless) to 1.0 (excellent)
using these criteria:

- **Actionable insight** (0.4 weight): Does it contain "because", "gotcha",
  "breaks if", "must be", "never", "always", "prevents", or similar causal
  reasoning?
- **Brevity** (0.3 weight): Is it concise?  Penalise parameter lists,
  function signatures, or code blocks that duplicate what's in the source.
- **Orientation value** (0.3 weight): Would a new developer know *where to
  start* and *what to avoid* after reading this?

Return your answer as a JSON object -- nothing else:

{"score": <float 0.0-1.0>, "reasoning": "<1-2 sentences>"}
"""

# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

_POSITIVE_SIGNALS: list[str] = [
    "because",
    "since",
    "ensures",
    "otherwise",
    "prevents",
    "gotcha",
    "breaks if",
    "must be",
    "never",
    "always",
]

_NEGATIVE_SIGNALS: list[str] = [
    "parameter",
    "returns a",
    "takes a",
    "type:",
]


def _heuristic_usefulness(doc_text: str) -> float:
    """Score usefulness 0.0-1.0 based on text signals alone.

    Counts positive signals (causal / tribal language) and negative signals
    (API-doc boilerplate), then combines them into a single score.
    """
    doc_lower = doc_text.lower()

    positive_hits = sum(1 for s in _POSITIVE_SIGNALS if s in doc_lower)
    negative_hits = sum(1 for s in _NEGATIVE_SIGNALS if s in doc_lower)

    # Normalise: positive contributes up to 0.7, negative deducts up to 0.3
    pos_score = min(1.0, positive_hits / max(len(_POSITIVE_SIGNALS) * 0.4, 1))
    neg_penalty = min(0.3, negative_hits * 0.075)

    score = 0.3 + 0.7 * pos_score - neg_penalty
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# LLM judge (Gemini Flash)
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{[^}]*\"score\"\s*:\s*[\d.]+[^}]*\}")


def _call_gemini(doc_text: str, module_name: str) -> tuple[float, str]:
    """Call Gemini Flash to judge usefulness.

    Raises
    ------
    RuntimeError
        If the API key is missing or the response cannot be parsed.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    # Import lazily so the module loads even without langchain installed
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-untyped]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.0,
        max_output_tokens=256,
    )

    user_message = (
        f"Module: {module_name}\n\n"
        f"--- DOCUMENT START ---\n{doc_text}\n--- DOCUMENT END ---"
    )

    response = llm.invoke(
        [
            {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )

    raw = str(response.content).strip()

    # Parse JSON from response (tolerant of markdown wrapping)
    match = _JSON_RE.search(raw)
    if not match:
        raise RuntimeError(f"Could not parse JSON from Gemini response: {raw[:200]}")

    parsed: dict[str, object] = json.loads(match.group())
    score_val = parsed.get("score", 0.5)
    reasoning_val = parsed.get("reasoning", "")

    score = float(score_val) if isinstance(score_val, (int, float)) else 0.5
    reasoning = str(reasoning_val) if reasoning_val else ""

    return max(0.0, min(1.0, score)), reasoning


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def judge_usefulness(doc_text: str, module_name: str) -> tuple[float, str]:
    """Judge the usefulness of a context document.

    Tries Gemini Flash first.  Falls back to the heuristic scorer if the
    API key is missing or the call fails.

    Returns
    -------
    tuple[float, str]
        (score, reasoning)  -- reasoning is empty for the heuristic fallback.
    """
    try:
        return _call_gemini(doc_text, module_name)
    except Exception as exc:
        logger.debug("LLM judge unavailable, using heuristic fallback: %s", exc)
        score = _heuristic_usefulness(doc_text)
        return score, ""
