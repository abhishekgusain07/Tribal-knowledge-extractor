"""Critic agent -- scores and critiques a module documentation draft.

This is the fourth agent in the 5-agent generation pipeline.  It receives
a draft document from the Writer, cross-checks it against Analyst findings
and the knowledge graph, and returns a structured critique with scores
on four dimensions: accuracy, coverage, conciseness, and usefulness.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage

from tribal_knowledge.pipeline.helpers import (
    get_all_entity_names,
    get_top_entities_for_module,
    load_graph_from_state,
)
from tribal_knowledge.pipeline.pipeline_models import AnalystFindings, CritiqueResult
from tribal_knowledge.pipeline.prompts import CRITIC_SYSTEM_PROMPT
from tribal_knowledge.pipeline.state import PipelineState


# ── Required sections per the Writer spec ────────────────────────────

_REQUIRED_SECTIONS = [
    "Quick Commands",
    "Key Files",
    "Non-Obvious Patterns",
    "See Also",
]

_MAX_LINES = 35


def _programmatic_critique(
    draft: str,
    findings: AnalystFindings,
    top_entities: list[str],
) -> CritiqueResult:
    """Produce a basic heuristic-based critique without an LLM call.

    This lets the full pipeline run in tests and CI environments where
    ANTHROPIC_API_KEY is not available.
    """
    lines = draft.strip().splitlines()
    line_count = len(lines)
    draft_lower = draft.lower()

    # ── Conciseness ──────────────────────────────────────────────────
    if line_count <= _MAX_LINES:
        conciseness = 1.0
    elif line_count <= _MAX_LINES + 10:
        conciseness = max(0.0, 1.0 - (line_count - _MAX_LINES) * 0.05)
    else:
        conciseness = 0.3

    critiques: list[str] = []
    if line_count > _MAX_LINES:
        critiques.append(
            f"CONCISENESS: Document is {line_count} lines, exceeds the "
            f"{_MAX_LINES}-line limit. Trim {line_count - _MAX_LINES} lines."
        )

    # ── Section coverage ─────────────────────────────────────────────
    sections_found = 0
    missing_sections: list[str] = []
    for section in _REQUIRED_SECTIONS:
        if section.lower() in draft_lower:
            sections_found += 1
        else:
            missing_sections.append(section)

    section_score = sections_found / len(_REQUIRED_SECTIONS)
    for sec in missing_sections:
        critiques.append(
            f"USEFULNESS: Missing required section '{sec}'. "
            "Add it to match the compass format."
        )

    # ── Coverage: top entities mentioned ─────────────────────────────
    if top_entities:
        mentioned = sum(
            1 for entity in top_entities if entity.lower() in draft_lower
        )
        coverage = mentioned / len(top_entities)
        missing_entities = [
            e for e in top_entities if e.lower() not in draft_lower
        ]
        if missing_entities:
            critiques.append(
                f"COVERAGE: Missing top entities: {', '.join(missing_entities[:5])}. "
                "Mention them in the most relevant section."
            )
    else:
        coverage = 0.8  # no entities to check -- give a reasonable default

    # ── Accuracy: rough heuristic (section structure present) ────────
    # Without an LLM we cannot truly verify entity names, so use
    # section_score as a proxy combined with non-empty content check.
    has_content = line_count >= 5
    accuracy = 1.0 if has_content and section_score >= 0.75 else 0.6

    # ── Usefulness: presence of "why" language and cross-refs ────────
    why_signals = sum(
        1
        for keyword in ("because", "since", "this ensures", "otherwise", "gotcha")
        if keyword in draft_lower
    )
    usefulness = min(1.0, 0.5 + why_signals * 0.1 + section_score * 0.3)

    # ── Analyst findings coverage bonus ──────────────────────────────
    if findings.tribal_knowledge:
        tk_mentioned = sum(
            1
            for tk in findings.tribal_knowledge
            if any(
                word.lower() in draft_lower
                for word in tk.split()
                if len(word) > 4
            )
        )
        tk_ratio = tk_mentioned / len(findings.tribal_knowledge)
        coverage = (coverage + tk_ratio) / 2.0

    # ── Overall ──────────────────────────────────────────────────────
    overall = (accuracy + coverage + conciseness + usefulness) / 4.0
    passed = overall >= 0.85

    return CritiqueResult(
        overall_score=round(overall, 3),
        dimension_scores={
            "accuracy": round(accuracy, 3),
            "coverage": round(coverage, 3),
            "conciseness": round(conciseness, 3),
            "usefulness": round(usefulness, 3),
        },
        critique=critiques,
        passed=passed,
    )


def critic_node(state: PipelineState) -> dict[str, object]:
    """LangGraph node: critique the current draft document.

    Returns a partial state dict with keys:
        critique, score, previous_score
    """
    # 1. Get the current draft
    current_draft: str = state.get("current_draft", "")

    # 2. Load AnalystFindings from state
    analyst_raw = state.get("analyst_output", {})
    if isinstance(analyst_raw, dict):
        findings = AnalystFindings.model_validate(analyst_raw)
    else:
        findings = AnalystFindings(module=state.get("current_module", "unknown"))

    # 3. Load knowledge graph from state
    graph = load_graph_from_state(state)  # type: ignore[arg-type]

    # 4. Get all entity names for accuracy checking
    all_entities = get_all_entity_names(graph)

    # 5. Get top-10 entities for this module for coverage checking
    current_module: str = state.get("current_module", "")
    top_entities = get_top_entities_for_module(graph, current_module)

    # 6. Capture previous score before we overwrite
    previous_score: float = state.get("score", 0.0)

    # 7. Decide: LLM critique vs programmatic fallback
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # ── Programmatic fallback ────────────────────────────────────
        result = _programmatic_critique(current_draft, findings, top_entities)
        return {
            "critique": result.model_dump(),
            "score": result.overall_score,
            "previous_score": previous_score,
        }

    # ── LLM-based critique ───────────────────────────────────────────
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model="claude-sonnet-4-6-20250514",
        temperature=0,
        max_tokens=2048,
    )
    structured_llm = llm.with_structured_output(CritiqueResult)

    # Format analyst findings as ground truth
    analyst_summary_parts: list[str] = []
    if findings.overview:
        analyst_summary_parts.append(f"Overview: {findings.overview}")
    if findings.data_flow:
        analyst_summary_parts.append(f"Data flow: {findings.data_flow}")
    if findings.patterns:
        patterns_text = "; ".join(
            f"{p.name} ({p.where}): {p.why}" for p in findings.patterns
        )
        analyst_summary_parts.append(f"Patterns: {patterns_text}")
    if findings.tribal_knowledge:
        analyst_summary_parts.append(
            f"Tribal knowledge: {'; '.join(findings.tribal_knowledge)}"
        )
    if findings.cross_module_dependencies:
        analyst_summary_parts.append(
            f"Cross-module deps: {', '.join(findings.cross_module_dependencies)}"
        )
    if findings.conventions:
        analyst_summary_parts.append(
            f"Conventions: {'; '.join(findings.conventions)}"
        )
    if findings.modification_patterns:
        analyst_summary_parts.append(
            f"Modification patterns: {findings.modification_patterns}"
        )
    if findings.failure_patterns:
        analyst_summary_parts.append(
            f"Failure patterns: {'; '.join(findings.failure_patterns)}"
        )
    analyst_summary = "\n".join(analyst_summary_parts)

    # Build entity lists for prompt
    entity_list_text = ", ".join(all_entities[:200]) if all_entities else "(none)"
    top_entities_text = ", ".join(top_entities) if top_entities else "(none)"

    human_content = (
        f"## Document to Critique\n\n{current_draft}\n\n"
        f"## Analyst Findings (ground truth)\n\n{analyst_summary}\n\n"
        f"## Entity Names from Knowledge Graph\n{entity_list_text}\n\n"
        f"## Top-10 Module Entities (by PageRank)\n{top_entities_text}\n\n"
        f"## Compass Rules\n"
        f"- Maximum {_MAX_LINES} lines\n"
        f"- Exactly 4 sections: {', '.join(_REQUIRED_SECTIONS)}\n"
        f"- 60-second test: a new developer should gain useful insight "
        f"within 60 seconds of reading\n"
    )

    try:
        result = structured_llm.invoke(
            [
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=human_content),
            ]
        )
        if isinstance(result, CritiqueResult):
            critique_result = result
        else:
            # Unexpected return type -- fall back to programmatic
            critique_result = _programmatic_critique(
                current_draft, findings, top_entities
            )
    except Exception as exc:
        print(
            f"[critic] WARNING: Claude API call failed ({exc}). "
            "Falling back to programmatic critique."
        )
        critique_result = _programmatic_critique(
            current_draft, findings, top_entities
        )

    return {
        "critique": critique_result.model_dump(),
        "score": critique_result.overall_score,
        "previous_score": previous_score,
    }
