"""Analyst agent — extracts tribal knowledge for a single module."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from tribal_knowledge.pipeline.helpers import (
    extract_module_subgraph_summary,
    format_cross_module_calls,
    get_chromadb_collection,
    get_module_entry,
    get_top_entities_for_module,
    get_tribal_signals_for_module,
    load_extractions_from_state,
    load_graph_from_state,
    query_chunks_for_module,
)
from tribal_knowledge.pipeline.pipeline_models import AnalystFindings, PatternFinding
from tribal_knowledge.pipeline.prompts import ANALYST_SYSTEM_PROMPT
from tribal_knowledge.pipeline.state import PipelineState


def analyst_node(state: PipelineState) -> dict[str, object]:
    """LangGraph node that analyses one module and returns structured findings."""

    # 1. Current module
    module_name: str = state["current_module"]

    # 2. Module entry from the explorer-generated module map
    module_entry = get_module_entry(state["module_map"], module_name)

    # 3. Load the knowledge graph and extractions
    G = load_graph_from_state(state)
    extractions = load_extractions_from_state(state)

    # 4. Retrieve relevant code chunks from ChromaDB
    chromadb_path = state.get("chromadb_path", "")
    code_chunks: str = "(no code chunks available)"
    if chromadb_path:
        try:
            collection = get_chromadb_collection(str(chromadb_path))
            code_chunks = query_chunks_for_module(
                collection, module_entry.description, module_entry.name
            )
        except Exception as exc:
            code_chunks = f"(error retrieving code chunks: {exc})"

    # 5. Subgraph summary (internal call chains)
    subgraph_summary = extract_module_subgraph_summary(G, module_entry.files)

    # 6. Cross-module calls
    cross_module = format_cross_module_calls(G, module_entry.name)

    # 7. Tribal signals from code comments / markers
    tribal_signals = get_tribal_signals_for_module(extractions, module_entry.files)

    # 8. Top entities by PageRank
    top_entities = get_top_entities_for_module(G, module_entry.name)

    # ── Build the user prompt with all context ──────────────────────
    user_prompt = _build_user_prompt(
        module_name=module_name,
        module_description=module_entry.description,
        files=module_entry.files,
        key_entities=module_entry.key_entities,
        top_entities=top_entities,
        code_chunks=code_chunks,
        subgraph_summary=subgraph_summary,
        cross_module=cross_module,
        tribal_signals=tribal_signals,
    )

    # ── Call the LLM (or fall back if no API key) ───────────────────
    from tribal_knowledge.pipeline.llm import has_llm_key

    if not has_llm_key():
        print(
            f"[analyst] WARNING: ANTHROPIC_API_KEY not set — "
            f"generating basic AnalystFindings for '{module_name}' without LLM."
        )
        findings = _fallback_findings(
            module_name=module_name,
            module_description=module_entry.description,
            top_entities=top_entities,
            tribal_signals=tribal_signals,
            cross_module=cross_module,
        )
    else:
        findings = _call_llm(user_prompt)

    findings_dict: dict[str, object] = findings.model_dump()

    # ── Return partial state update ─────────────────────────────────
    previous: list[dict[str, object]] = list(state.get("all_analyst_outputs", []) or [])
    previous.append(findings_dict)

    return {
        "analyst_output": findings_dict,
        "all_analyst_outputs": previous,
    }


# ── Private helpers ─────────────────────────────────────────────────


def _build_user_prompt(
    module_name: str,
    module_description: str,
    files: list[str],
    key_entities: list[str],
    top_entities: list[str],
    code_chunks: str,
    subgraph_summary: str,
    cross_module: str,
    tribal_signals: str,
) -> str:
    """Compose the rich user message sent alongside the system prompt."""
    file_list = "\n".join(f"  - {f}" for f in files) if files else "  (none)"
    entity_list = ", ".join(top_entities) if top_entities else "(none)"
    key_entity_list = ", ".join(key_entities) if key_entities else "(none)"

    return f"""\
Analyse the module **{module_name}**.

## Module description
{module_description}

## Files in this module
{file_list}

## Key entities (from Explorer)
{key_entity_list}

## Top entities by PageRank
{entity_list}

## Internal call chains
{subgraph_summary}

## Cross-module dependencies
{cross_module}

## Tribal signals (comments, markers, hacks)
{tribal_signals}

## Relevant code chunks
{code_chunks}
"""


def _call_llm(user_prompt: str) -> AnalystFindings:
    """Invoke Claude Sonnet with structured output."""
    from tribal_knowledge.pipeline.llm import get_structured_llm

    structured_llm = get_structured_llm(AnalystFindings, temperature=0, max_tokens=4096)

    messages = [
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    result = structured_llm.invoke(messages)
    if isinstance(result, AnalystFindings):
        return result
    # Defensive: if the LLM somehow returns a dict, validate it
    if isinstance(result, dict):
        return AnalystFindings.model_validate(result)
    return AnalystFindings(module="unknown")


def _fallback_findings(
    module_name: str,
    module_description: str,
    top_entities: list[str],
    tribal_signals: str,
    cross_module: str,
) -> AnalystFindings:
    """Build a basic AnalystFindings without calling an LLM."""
    # Parse tribal signals into a list of strings
    tribal_list: list[str] = []
    if tribal_signals and tribal_signals != "(no tribal signals found)":
        for line in tribal_signals.strip().splitlines():
            stripped = line.strip()
            if stripped:
                tribal_list.append(stripped)

    # Parse cross-module dependencies into a list
    dep_list: list[str] = []
    if cross_module and cross_module != "(no cross-module edges)":
        for line in cross_module.strip().splitlines():
            stripped = line.strip()
            if stripped:
                dep_list.append(stripped)

    return AnalystFindings(
        module=module_name,
        overview=module_description or f"Module '{module_name}' (no LLM analysis available).",
        data_flow="",
        patterns=[
            PatternFinding(
                name=entity,
                where=module_name,
                why="Top entity by PageRank",
            )
            for entity in top_entities[:5]
        ],
        tribal_knowledge=tribal_list[:10],
        cross_module_dependencies=dep_list[:10],
        conventions=[],
        modification_patterns="",
        failure_patterns=[],
    )
