"""Output module -- writes generated documents to disk and produces a quality report.

Handles file I/O for all pipeline-generated documents and assembles the final
QualityReport with per-document scores, token counts, and cost tracking.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import tiktoken

from tribal_knowledge.pipeline.pipeline_models import DocumentScore, QualityReport
from tribal_knowledge.pipeline.state import PipelineState


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_ENCODING: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    """Lazily load and cache the cl100k_base encoding."""
    global _ENCODING  # noqa: PLW0603
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_doc_tokens(text: str) -> int:
    """Count tokens in *text* using the tiktoken cl100k_base encoding.

    Used for token-budget validation of generated documents.
    """
    if not text:
        return 0
    enc = _get_encoding()
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def _sanitize_module_name(name: str) -> str:
    """Convert a module name to a safe filename component.

    Lowercase, replace spaces/underscores with hyphens, strip non-alphanumeric
    characters (except hyphens and dots).
    """
    sanitized = name.lower().strip()
    sanitized = sanitized.replace(" ", "-").replace("_", "-")
    sanitized = re.sub(r"[^a-z0-9\-.]", "", sanitized)
    # Collapse multiple hyphens
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    return sanitized.strip("-") or "unnamed-module"


# ---------------------------------------------------------------------------
# Quality report builder
# ---------------------------------------------------------------------------


def _build_quality_report(state: PipelineState) -> QualityReport:
    """Assemble a QualityReport from pipeline state data."""
    documents: list[DocumentScore] = []

    generation_log: list[dict[str, object]] = state.get("generation_log", [])
    module_docs: dict[str, str] = state.get("module_docs", {})

    # Build a lookup from generation log entries
    log_by_module: dict[str, dict[str, object]] = {}
    for entry in generation_log:
        module_name = entry.get("module")
        if isinstance(module_name, str):
            log_by_module[module_name] = entry

    # Per-module document scores
    for module_name, doc_text in sorted(module_docs.items()):
        log_entry = log_by_module.get(module_name, {})

        score_val = log_entry.get("score", 0.0)
        score = float(score_val) if isinstance(score_val, (int, float)) else 0.0

        iter_val = log_entry.get("iterations", 1)
        iterations = int(iter_val) if isinstance(iter_val, (int, float)) else 1

        tokens_val = log_entry.get("tokens_used", 0)
        tokens_used = int(tokens_val) if isinstance(tokens_val, (int, float)) else count_doc_tokens(doc_text)

        documents.append(
            DocumentScore(
                name=f"{_sanitize_module_name(module_name)}.md",
                score=score,
                iterations=iterations,
                tokens_used=tokens_used,
            )
        )

    # Synthesis document scores (no iteration data, score based on existence)
    synthesis_docs: list[tuple[str, str]] = [
        ("OVERVIEW.md", state.get("overview_doc", "")),
        ("CONVENTIONS.md", state.get("conventions_doc", "")),
        ("AGENTS.md", state.get("agents_doc", "")),
        ("dependency-index.md", state.get("dependency_index_doc", "")),
    ]
    for doc_name, doc_text in synthesis_docs:
        doc_tokens = count_doc_tokens(doc_text) if doc_text else 0
        doc_score = 1.0 if doc_text else 0.0
        documents.append(
            DocumentScore(
                name=doc_name,
                score=doc_score,
                iterations=1,
                tokens_used=doc_tokens,
            )
        )

    # Aggregate stats
    total_tokens_val = state.get("total_input_tokens", 0)
    total_input = int(total_tokens_val) if isinstance(total_tokens_val, (int, float)) else 0
    total_output_val = state.get("total_output_tokens", 0)
    total_output = int(total_output_val) if isinstance(total_output_val, (int, float)) else 0
    total_tokens = total_input + total_output

    total_cost_val = state.get("total_cost", 0.0)
    total_cost = float(total_cost_val) if isinstance(total_cost_val, (int, float)) else 0.0

    scores = [d.score for d in documents if d.score > 0]
    average_score = sum(scores) / len(scores) if scores else 0.0

    return QualityReport(
        documents=documents,
        total_cost=total_cost,
        total_tokens=total_tokens,
        average_score=round(average_score, 4),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def write_outputs(state: PipelineState, output_dir: str) -> QualityReport:
    """Write all generated documents to disk and return a QualityReport.

    Directory layout::

        {output_dir}/context/
            {module-name}.md    -- per-module docs
            OVERVIEW.md
            CONVENTIONS.md
            AGENTS.md
            dependency-index.md
            quality-report.json

    Parameters
    ----------
    state:
        The final pipeline state containing all generated content.
    output_dir:
        Root output directory.  A ``context/`` subdirectory is created inside.

    Returns
    -------
    QualityReport
        Per-document quality scores and aggregate statistics.
    """
    context_dir = Path(output_dir) / "context"
    context_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write per-module docs
    module_docs: dict[str, str] = state.get("module_docs", {})
    for module_name, doc_text in module_docs.items():
        filename = f"{_sanitize_module_name(module_name)}.md"
        filepath = context_dir / filename
        filepath.write_text(doc_text, encoding="utf-8")

    # 2. Write synthesis docs
    synthesis_files: list[tuple[str, str]] = [
        ("OVERVIEW.md", state.get("overview_doc", "")),
        ("CONVENTIONS.md", state.get("conventions_doc", "")),
        ("AGENTS.md", state.get("agents_doc", "")),
        ("dependency-index.md", state.get("dependency_index_doc", "")),
    ]
    for filename, content in synthesis_files:
        if content:
            filepath = context_dir / filename
            filepath.write_text(content, encoding="utf-8")

    # 3. Build quality report
    report = _build_quality_report(state)

    # 4. Write quality-report.json
    report_path = context_dir / "quality-report.json"
    report_path.write_text(
        json.dumps(report.model_dump(), indent=2, default=str),
        encoding="utf-8",
    )

    return report
