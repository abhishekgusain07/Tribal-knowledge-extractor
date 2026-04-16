"""Synthesizer agent -- produces cross-cutting documents after all per-module docs.

This is the final synthesis step in the 5-agent generation pipeline.  It reads
all per-module docs and analyst outputs, then produces OVERVIEW.md,
CONVENTIONS.md, AGENTS.md, and dependency-index.md.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from tribal_knowledge.models import GraphAnalysis, ModuleDependency
from tribal_knowledge.pipeline.helpers import format_module_list
from tribal_knowledge.pipeline.pipeline_models import AnalystFindings
from tribal_knowledge.pipeline.prompts import (
    AGENTS_MD_SYSTEM_PROMPT,
    CONVENTIONS_SYSTEM_PROMPT,
    OVERVIEW_SYSTEM_PROMPT,
)
from tribal_knowledge.pipeline.state import PipelineState


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_api_key() -> bool:
    """Return True if a supported LLM API key is available."""
    from tribal_knowledge.pipeline.llm import has_llm_key

    return has_llm_key()


def _call_llm(system_prompt: str, user_content: str) -> str:
    """Invoke the LLM and return the response text."""
    from tribal_knowledge.pipeline.llm import get_llm

    llm = get_llm(temperature=0, max_tokens=1500)
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
    )
    text = response.content
    if isinstance(text, str):
        return text
    # LangChain can return a list of content blocks
    if isinstance(text, list):
        parts: list[str] = []
        for block in text:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text", "")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return str(text)


def _parse_analyst_findings(
    raw_outputs: list[dict[str, object]],
) -> list[AnalystFindings]:
    """Deserialise raw analyst output dicts into AnalystFindings models."""
    results: list[AnalystFindings] = []
    for item in raw_outputs:
        try:
            results.append(AnalystFindings.model_validate(item))
        except Exception:
            continue
    return results


def _collect_conventions(findings_list: list[AnalystFindings]) -> list[str]:
    """Gather all conventions from every analyst output."""
    conventions: list[str] = []
    for f in findings_list:
        for c in f.conventions:
            if c and c not in conventions:
                conventions.append(c)
    return conventions


def _collect_tribal_knowledge(findings_list: list[AnalystFindings]) -> list[str]:
    """Gather all tribal knowledge items from every analyst output."""
    tribal: list[str] = []
    for f in findings_list:
        for t in f.tribal_knowledge:
            if t and t not in tribal:
                tribal.append(t)
    return tribal


def _format_cross_module_deps(deps: list[ModuleDependency]) -> str:
    """Format cross-module dependencies as a markdown table."""
    if not deps:
        return "No cross-module dependencies detected."
    lines: list[str] = [
        "| Module | Depends On | Key Connections |",
        "|--------|-----------|-----------------|",
    ]
    for dep in deps:
        connections = "; ".join(dep.key_connections[:3]) if dep.key_connections else f"{dep.edge_count} edges"
        lines.append(
            f"| {dep.source_module} | {dep.target_module} | {connections} |"
        )
    return "\n".join(lines)


def _build_change_impact_map(deps: list[ModuleDependency]) -> str:
    """Build a 'Change Impact Map' section from dependency data."""
    if not deps:
        return "No dependency data available for impact analysis."

    # Group by source module to show what each module affects
    impact: dict[str, list[str]] = {}
    for dep in deps:
        targets = impact.setdefault(dep.source_module, [])
        if dep.target_module not in targets:
            targets.append(dep.target_module)
    # Also track reverse (what depends on a given module)
    reverse_impact: dict[str, list[str]] = {}
    for dep in deps:
        dependents = reverse_impact.setdefault(dep.target_module, [])
        if dep.source_module not in dependents:
            dependents.append(dep.source_module)

    lines: list[str] = []
    all_modules = sorted(set(list(impact.keys()) + list(reverse_impact.keys())))
    for mod in all_modules:
        uses = impact.get(mod, [])
        used_by = reverse_impact.get(mod, [])
        parts: list[str] = []
        if uses:
            parts.append(f"uses {', '.join(uses)}")
        if used_by:
            parts.append(f"used by {', '.join(used_by)}")
        if parts:
            lines.append(f"- **{mod}**: {'; '.join(parts)}")
    return "\n".join(lines) if lines else "No impact data available."


# ---------------------------------------------------------------------------
# Fallback (template-based) document generators
# ---------------------------------------------------------------------------


def _fallback_overview(
    module_docs: dict[str, str],
    framework: str,
    deps_table: str,
) -> str:
    """Generate a template-based OVERVIEW.md without LLM calls."""
    module_list = "\n".join(
        f"- **{name}** (see {name.lower().replace(' ', '-')}.md)"
        for name in sorted(module_docs.keys())
    )
    return (
        f"# Project Overview -- Codebase Context\n\n"
        f"## What This Is\n"
        f"A software project documented by Tribal Knowledge.\n\n"
        f"## Tech Stack\n"
        f"{framework if framework else 'Not detected'}\n\n"
        f"## Module Map\n"
        f"{module_list}\n\n"
        f"## Cross-Module Dependencies\n"
        f"{deps_table}\n"
    )


def _fallback_conventions(
    conventions: list[str],
    tribal_knowledge: list[str],
) -> str:
    """Generate a template-based CONVENTIONS.md without LLM calls."""
    conv_bullets = "\n".join(f"- {c}" for c in conventions[:15]) if conventions else "- No conventions detected"
    tk_bullets = "\n".join(f"- {t}" for t in tribal_knowledge[:15]) if tribal_knowledge else "- No tribal knowledge detected"
    return (
        f"# Conventions\n\n"
        f"## Coding Conventions\n"
        f"{conv_bullets}\n\n"
        f"## Tribal Knowledge\n"
        f"{tk_bullets}\n"
    )


def _fallback_agents(
    overview_doc: str,
    conventions_doc: str,
    module_names: list[str],
) -> str:
    """Generate a template-based AGENTS.md without LLM calls."""
    module_list = "\n".join(f"- {name}" for name in module_names)
    return (
        f"# Project -- AI Agent Context\n\n"
        f"## Overview\n"
        f"This file provides context for AI coding tools working in this repository.\n\n"
        f"## Architecture\n"
        f"{module_list}\n\n"
        f"## Key Conventions\n"
        f"See CONVENTIONS.md for the full list of codebase conventions.\n\n"
        f"## Gotchas\n"
        f"See individual module docs for module-specific gotchas.\n"
    )


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


def synthesize_node(state: PipelineState) -> dict[str, object]:
    """LangGraph node: produce cross-cutting synthesis documents.

    Returns a partial state dict with keys:
        overview_doc, conventions_doc, agents_doc, dependency_index_doc
    """
    # 1. Gather inputs from state
    module_docs: dict[str, str] = state.get("module_docs", {})
    raw_analyst_outputs: list[dict[str, object]] = state.get("all_analyst_outputs", [])
    framework: str = state.get("framework", "")
    module_map: list[dict[str, object]] = state.get("module_map", [])

    # 2. Parse analyst findings
    findings_list = _parse_analyst_findings(raw_analyst_outputs)
    conventions = _collect_conventions(findings_list)
    tribal_knowledge = _collect_tribal_knowledge(findings_list)

    # 3. Load analysis for cross-module dependency info
    analysis_data: dict[str, object] = state.get("analysis_data", {})
    analysis = GraphAnalysis.model_validate(analysis_data) if analysis_data else GraphAnalysis()
    cross_module_deps = analysis.cross_module_deps

    # 4. Format dependency data (used by both LLM and fallback paths)
    deps_table = _format_cross_module_deps(cross_module_deps)
    change_impact = _build_change_impact_map(cross_module_deps)

    # 5. Build dependency-index.md (mostly template-based)
    dependency_index_doc = (
        f"# Dependency Index\n\n"
        f"## Module Dependencies\n\n"
        f"{deps_table}\n\n"
        f"## Change Impact Map\n\n"
        f"{change_impact}\n"
    )

    # 6. Check for API key and generate documents
    if not _has_api_key():
        print(
            "[synthesizer] WARNING: ANTHROPIC_API_KEY not set. "
            "Generating template-based documents without LLM."
        )
        module_names = [
            entry.get("name", "unknown")
            for entry in module_map
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        ]
        overview_doc = _fallback_overview(module_docs, framework, deps_table)
        conventions_doc = _fallback_conventions(conventions, tribal_knowledge)
        agents_doc = _fallback_agents(overview_doc, conventions_doc, module_names)

        return {
            "overview_doc": overview_doc,
            "conventions_doc": conventions_doc,
            "agents_doc": agents_doc,
            "dependency_index_doc": dependency_index_doc,
        }

    # --- LLM-powered synthesis ---

    # 7a. OVERVIEW.md
    all_module_docs_text = "\n\n---\n\n".join(
        f"### {name}\n{doc}" for name, doc in sorted(module_docs.items())
    )
    module_list_text = format_module_list(module_map)
    overview_user_content = (
        f"## Framework\n{framework}\n\n"
        f"## Module List\n{module_list_text}\n\n"
        f"## Per-Module Documentation\n{all_module_docs_text}\n\n"
        f"## Cross-Module Dependencies\n{deps_table}\n"
    )
    try:
        overview_doc = _call_llm(OVERVIEW_SYSTEM_PROMPT, overview_user_content)
    except Exception as exc:
        print(f"[synthesizer] WARNING: OVERVIEW LLM call failed ({exc}). Using fallback.")
        overview_doc = _fallback_overview(module_docs, framework, deps_table)

    # 7b. CONVENTIONS.md
    conventions_text = "\n".join(f"- {c}" for c in conventions) if conventions else "(none)"
    tribal_text = "\n".join(f"- {t}" for t in tribal_knowledge) if tribal_knowledge else "(none)"
    conventions_user_content = (
        f"## Conventions Collected from All Modules\n{conventions_text}\n\n"
        f"## Tribal Knowledge Collected from All Modules\n{tribal_text}\n"
    )
    try:
        conventions_doc = _call_llm(CONVENTIONS_SYSTEM_PROMPT, conventions_user_content)
    except Exception as exc:
        print(f"[synthesizer] WARNING: CONVENTIONS LLM call failed ({exc}). Using fallback.")
        conventions_doc = _fallback_conventions(conventions, tribal_knowledge)

    # 7c. AGENTS.md
    module_names_list = [
        entry.get("name", "unknown")
        for entry in module_map
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    ]
    module_names_text = "\n".join(f"- {n}" for n in module_names_list)
    agents_user_content = (
        f"## OVERVIEW.md Content\n{overview_doc}\n\n"
        f"## CONVENTIONS.md Content\n{conventions_doc}\n\n"
        f"## Module List\n{module_names_text}\n"
    )
    try:
        agents_doc = _call_llm(AGENTS_MD_SYSTEM_PROMPT, agents_user_content)
    except Exception as exc:
        print(f"[synthesizer] WARNING: AGENTS.md LLM call failed ({exc}). Using fallback.")
        agents_doc = _fallback_agents(overview_doc, conventions_doc, module_names_list)

    return {
        "overview_doc": overview_doc,
        "conventions_doc": conventions_doc,
        "agents_doc": agents_doc,
        "dependency_index_doc": dependency_index_doc,
    }
