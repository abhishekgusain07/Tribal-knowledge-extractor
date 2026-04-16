"""LangGraph pipeline state definition."""

from __future__ import annotations

from typing import TypedDict


class PipelineState(TypedDict, total=False):
    """Shared state flowing through all pipeline nodes.

    ``total=False`` means every key is optional — nodes only return the
    subset of keys they update, and LangGraph merges them into the
    running state.
    """

    # ── Inputs (set once at start, read-only after) ──────────────────
    repo_path: str
    output_dir: str
    knowledge_graph_data: dict[str, object]
    analysis_data: dict[str, object]
    extractions_data: list[dict[str, object]]
    chromadb_path: str

    # ── Explorer output ──────────────────────────────────────────────
    module_map: list[dict[str, object]]
    framework: str
    file_tree_summary: str

    # ── Per-module iteration state ───────────────────────────────────
    current_module: str
    module_queue: list[str]
    analyst_output: dict[str, object]
    current_draft: str
    critique: dict[str, object]
    score: float
    iteration: int
    previous_score: float

    # ── Accumulated outputs ──────────────────────────────────────────
    module_docs: dict[str, str]
    all_analyst_outputs: list[dict[str, object]]
    overview_doc: str
    conventions_doc: str
    agents_doc: str
    dependency_index_doc: str

    # ── Tracking ─────────────────────────────────────────────────────
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    generation_log: list[dict[str, object]]
    errors: list[str]
