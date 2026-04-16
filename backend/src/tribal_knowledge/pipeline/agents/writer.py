"""Writer agent — turns Analyst findings into a concise Markdown document."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from tribal_knowledge.pipeline.helpers import format_module_list
from tribal_knowledge.pipeline.pipeline_models import AnalystFindings
from tribal_knowledge.pipeline.prompts import WRITER_SYSTEM_PROMPT
from tribal_knowledge.pipeline.state import PipelineState


def _build_fallback_doc(findings: AnalystFindings) -> str:
    """Generate a basic markdown document directly from AnalystFindings."""
    lines: list[str] = []

    lines.append(f"# {findings.module}")
    lines.append("")

    # Quick Commands
    lines.append("## Quick Commands")
    if findings.modification_patterns:
        for sentence in findings.modification_patterns.split(". ")[:3]:
            stripped = sentence.strip().rstrip(".")
            if stripped:
                lines.append(f"- {stripped}")
    else:
        lines.append("- (no quick commands available)")
    lines.append("")

    # Key Files
    lines.append("## Key Files")
    if findings.cross_module_dependencies:
        for dep in findings.cross_module_dependencies[:5]:
            lines.append(f"- {dep}")
    else:
        lines.append("- (no key files identified)")
    lines.append("")

    # Non-Obvious Patterns
    lines.append("## Non-Obvious Patterns")
    items_added = 0
    for pattern in findings.patterns[:3]:
        lines.append(f"- **{pattern.name}** ({pattern.where}): {pattern.why}")
        items_added += 1
    for tk in findings.tribal_knowledge[:3]:
        lines.append(f"- {tk}")
        items_added += 1
    for fp in findings.failure_patterns[:2]:
        lines.append(f"- {fp}")
        items_added += 1
    if items_added == 0:
        lines.append("- (no non-obvious patterns found)")
    lines.append("")

    # See Also
    lines.append("## See Also")
    if findings.cross_module_dependencies:
        for dep in findings.cross_module_dependencies[:4]:
            lines.append(f"- {dep}")
    else:
        lines.append("- (no cross-references)")

    return "\n".join(lines)


def writer_node(state: PipelineState) -> dict[str, object]:
    """Generate an initial Markdown document from analyst findings.

    Returns keys: current_draft, iteration, score, previous_score.
    """
    # 1. Load AnalystFindings from state
    analyst_raw = state.get("analyst_output", {})
    if not isinstance(analyst_raw, dict):
        analyst_raw = {}
    findings = AnalystFindings.model_validate(analyst_raw)

    # 2. Get module list for cross-references
    module_map: list[dict[str, object]] = state.get("module_map", [])  # type: ignore[assignment]
    if not isinstance(module_map, list):
        module_map = []
    module_list_text = format_module_list(module_map)

    # 3. Build the human message
    human_text = (
        f"Write a navigation document for the **{findings.module}** module.\n\n"
        f"## Analyst Findings\n\n"
        f"**Overview:** {findings.overview}\n\n"
        f"**Data Flow:** {findings.data_flow}\n\n"
        f"**Patterns:**\n"
    )
    for p in findings.patterns:
        human_text += f"- {p.name} ({p.where}): {p.why}\n"

    human_text += f"\n**Tribal Knowledge:**\n"
    for tk in findings.tribal_knowledge:
        human_text += f"- {tk}\n"

    human_text += f"\n**Cross-Module Dependencies:**\n"
    for dep in findings.cross_module_dependencies:
        human_text += f"- {dep}\n"

    human_text += f"\n**Conventions:**\n"
    for conv in findings.conventions:
        human_text += f"- {conv}\n"

    human_text += f"\n**Modification Patterns:** {findings.modification_patterns}\n"

    human_text += f"\n**Failure Patterns:**\n"
    for fp in findings.failure_patterns:
        human_text += f"- {fp}\n"

    human_text += f"\n## Other Modules (for cross-references)\n{module_list_text}\n"

    # 4. Call LLM
    try:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model="claude-sonnet-4-6-20250514",
            temperature=0.3,
            max_tokens=2048,
        )
        messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=human_text),
        ]
        response = llm.invoke(messages)
        draft = str(response.content)
    except Exception as exc:
        print(f"[writer] WARNING: LLM call failed ({exc}), generating fallback document.")
        draft = _build_fallback_doc(findings)

    return {
        "current_draft": draft,
        "iteration": 0,
        "score": 0.0,
        "previous_score": 0.0,
    }
