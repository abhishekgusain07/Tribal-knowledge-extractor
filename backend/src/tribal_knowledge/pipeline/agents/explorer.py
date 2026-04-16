"""Explorer agent — surveys the codebase and identifies logical modules.

This is the first agent in the 5-agent generation pipeline.  It reads Phase-1
analysis data, builds context for a Gemini LLM call, and produces a refined
module map that downstream agents iterate over.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from tribal_knowledge.models import GraphAnalysis, ModuleInfo
from tribal_knowledge.pipeline.helpers import (
    build_file_tree_summary,
    format_modules_for_prompt,
    format_package_configs,
    format_top_entities,
    load_extractions_from_state,
)
from tribal_knowledge.pipeline.pipeline_models import ExplorerOutput, ModuleMapEntry
from tribal_knowledge.pipeline.prompts import EXPLORER_SYSTEM_PROMPT
from tribal_knowledge.pipeline.state import PipelineState


def _fallback_module_map(analysis: GraphAnalysis) -> list[ModuleMapEntry]:
    """Convert Phase-1 ModuleInfo objects into ModuleMapEntry format.

    Used when the Gemini API call fails or is unavailable.
    """
    entries: list[ModuleMapEntry] = []
    for m in analysis.modules:
        entries.append(
            ModuleMapEntry(
                name=m.name,
                description=f"Auto-detected module with {len(m.files)} files",
                files=list(m.files),
                key_entities=list(m.top_entities[:10]),
                importance=m.cohesion if m.cohesion > 0 else 0.5,
            )
        )
    return entries


def _full_codebase_fallback(
    analysis: GraphAnalysis,
) -> list[ModuleMapEntry]:
    """Return a single 'full-codebase' module containing all files."""
    all_files: list[str] = []
    for m in analysis.modules:
        all_files.extend(m.files)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_files: list[str] = []
    for f in all_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    all_entities: list[str] = []
    for item in analysis.top_functions[:10]:
        name = item.get("name", "")
        if isinstance(name, str) and name:
            all_entities.append(name)
    for item in analysis.top_classes[:5]:
        name = item.get("name", "")
        if isinstance(name, str) and name:
            all_entities.append(name)

    return [
        ModuleMapEntry(
            name="full-codebase",
            description="Entire codebase as a single module",
            files=unique_files,
            key_entities=all_entities,
            importance=1.0,
        )
    ]


def explorer_node(state: PipelineState) -> dict[str, object]:
    """LangGraph node: explore the codebase and produce a module map.

    Returns a partial state dict with keys:
        module_map, framework, module_queue, current_module, file_tree_summary
    """
    # 1. Load Phase-1 analysis
    analysis = GraphAnalysis.model_validate(state["analysis_data"])

    # 2. Build file tree summary
    repo_path: str = state["repo_path"]
    file_tree_summary = build_file_tree_summary(repo_path)

    # 3. Format context for the prompt
    modules_text = format_modules_for_prompt(analysis.modules)
    top_entities_text = format_top_entities(analysis)

    extractions = load_extractions_from_state(state)  # type: ignore[arg-type]
    package_configs_text = format_package_configs(extractions)

    human_message_content = (
        f"## File Tree\n```\n{file_tree_summary}\n```\n\n"
        f"## Static Analysis Modules\n{modules_text}\n\n"
        f"## Top Entities (by PageRank)\n{top_entities_text}\n\n"
        f"## Package Configs\n{package_configs_text}\n"
    )

    # 4. Call Gemini Flash with structured output
    explorer_output: ExplorerOutput | None = None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        structured_llm = llm.with_structured_output(ExplorerOutput)

        result = structured_llm.invoke(
            [
                SystemMessage(content=EXPLORER_SYSTEM_PROMPT),
                HumanMessage(content=human_message_content),
            ]
        )
        if isinstance(result, ExplorerOutput):
            explorer_output = result
    except Exception as exc:
        print(
            f"[explorer] WARNING: Gemini API call failed ({exc}). "
            "Falling back to Phase-1 modules."
        )

    # 5. Build the module map from either LLM output or fallback
    if explorer_output is not None and len(explorer_output.modules) > 0:
        module_entries = explorer_output.modules
        framework = explorer_output.framework
    else:
        # Fallback: use Phase-1 modules directly
        module_entries = _fallback_module_map(analysis)
        framework = ""

    # Handle case where no modules were detected at all
    if len(module_entries) == 0:
        module_entries = _full_codebase_fallback(analysis)

    # 6. Build return dict
    module_map: list[dict[str, object]] = [
        entry.model_dump() for entry in module_entries
    ]
    module_names: list[str] = [entry.name for entry in module_entries]
    first_module: str = module_names[0] if module_names else "full-codebase"

    return {
        "module_map": module_map,
        "framework": framework,
        "module_queue": module_names,
        "current_module": first_module,
        "file_tree_summary": file_tree_summary,
    }
