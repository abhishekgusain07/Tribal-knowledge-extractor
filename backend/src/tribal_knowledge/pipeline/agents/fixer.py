"""Fixer agent — makes surgical edits to a document based on critique."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from tribal_knowledge.pipeline.pipeline_models import AnalystFindings, CritiqueResult
from tribal_knowledge.pipeline.prompts import FIXER_SYSTEM_PROMPT
from tribal_knowledge.pipeline.state import PipelineState


def fixer_node(state: PipelineState) -> dict[str, object]:
    """Revise the current draft based on critique feedback.

    Returns keys: current_draft, iteration.
    """
    # 1. Get current draft
    current_draft = state.get("current_draft", "")
    if not isinstance(current_draft, str):
        current_draft = str(current_draft)

    # 2. Load CritiqueResult from state
    critique_raw = state.get("critique", {})
    if not isinstance(critique_raw, dict):
        critique_raw = {}
    critique = CritiqueResult.model_validate(critique_raw)

    # 3. Load AnalystFindings from state
    analyst_raw = state.get("analyst_output", {})
    if not isinstance(analyst_raw, dict):
        analyst_raw = {}
    findings = AnalystFindings.model_validate(analyst_raw)

    # 4. Get current iteration
    current_iteration: int = state.get("iteration", 0)  # type: ignore[assignment]
    if not isinstance(current_iteration, int):
        current_iteration = 0

    # 5. Build critique bullet list
    critique_bullets = "\n".join(f"- {point}" for point in critique.critique)
    if not critique_bullets:
        critique_bullets = "- (no specific critique points provided)"

    # 6. Build dimension scores summary
    dimension_lines: list[str] = []
    for dim, score in critique.dimension_scores.items():
        dimension_lines.append(f"- {dim}: {score:.2f}")
    dimension_text = "\n".join(dimension_lines) if dimension_lines else "- (no dimension scores)"

    # 7. Build analyst findings summary for reference
    analyst_summary = (
        f"**Module:** {findings.module}\n"
        f"**Overview:** {findings.overview}\n"
        f"**Data Flow:** {findings.data_flow}\n"
    )
    if findings.patterns:
        analyst_summary += "**Patterns:**\n"
        for p in findings.patterns:
            analyst_summary += f"- {p.name} ({p.where}): {p.why}\n"
    if findings.tribal_knowledge:
        analyst_summary += "**Tribal Knowledge:**\n"
        for tk in findings.tribal_knowledge:
            analyst_summary += f"- {tk}\n"
    if findings.cross_module_dependencies:
        analyst_summary += "**Cross-Module Dependencies:**\n"
        for dep in findings.cross_module_dependencies:
            analyst_summary += f"- {dep}\n"
    if findings.failure_patterns:
        analyst_summary += "**Failure Patterns:**\n"
        for fp in findings.failure_patterns:
            analyst_summary += f"- {fp}\n"

    # 8. Compose human message
    human_text = (
        f"## Current Document\n\n"
        f"{current_draft}\n\n"
        f"## Critique Points\n\n"
        f"{critique_bullets}\n\n"
        f"## Dimension Scores\n\n"
        f"{dimension_text}\n\n"
        f"Overall score: {critique.overall_score:.2f}\n\n"
        f"## Original Analyst Findings (for reference)\n\n"
        f"{analyst_summary}"
    )

    # 9. Call LLM
    try:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model="claude-sonnet-4-6-20250514",
            temperature=0,
            max_tokens=2048,
        )
        messages = [
            SystemMessage(content=FIXER_SYSTEM_PROMPT),
            HumanMessage(content=human_text),
        ]
        response = llm.invoke(messages)
        revised_draft = str(response.content)
    except Exception as exc:
        print(f"[fixer] WARNING: LLM call failed ({exc}), returning original draft unchanged.")
        revised_draft = current_draft

    return {
        "current_draft": revised_draft,
        "iteration": current_iteration + 1,
    }
