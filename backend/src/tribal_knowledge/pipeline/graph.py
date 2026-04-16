"""LangGraph state machine assembly for the 5-agent generation pipeline.

Wires Explorer -> Analyst -> Writer -> Critic -> (Fixer loop) -> Save ->
(next module or Synthesize -> END).
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from tribal_knowledge.pipeline.state import PipelineState


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def should_fix_or_pass(state: PipelineState) -> str:
    """Route after Critic: pass if quality gate met, otherwise fix.

    Pass conditions (checked in order):
    1. Score >= 0.85 -- meets the quality bar.
    2. Iteration >= 3 -- hard cap (1 write + 3 fix cycles).
    3. Iteration >= 2 AND score improved < 0.02 -- converged, model is stuck.
    """
    score: float = state.get("score", 0.0)
    iteration: int = state.get("iteration", 0)
    previous_score: float = state.get("previous_score", 0.0)

    if score >= 0.85:
        return "pass"

    if iteration >= 3:
        return "pass"

    if iteration >= 2 and (score - previous_score) < 0.02:
        return "pass"

    return "fix"


def has_more_modules(state: PipelineState) -> str:
    """Route after save_module: process next module or move to synthesis."""
    queue: list[str] = state.get("module_queue", [])
    if queue:
        return "next"
    return "done"


# ---------------------------------------------------------------------------
# Inline utility nodes
# ---------------------------------------------------------------------------


def save_module_doc_node(state: PipelineState) -> dict[str, dict[str, str]]:
    """Save the finalised document for the current module into module_docs."""
    module_docs: dict[str, str] = dict(state.get("module_docs", {}))
    current_module: str = state.get("current_module", "unknown")
    current_draft: str = state.get("current_draft", "")
    module_docs[current_module] = current_draft
    return {"module_docs": module_docs}


def advance_module_queue_node(state: PipelineState) -> dict[str, object]:
    """Pop the next module from the queue and reset per-module iteration state."""
    queue: list[str] = list(state.get("module_queue", []))
    next_module: str = queue.pop(0) if queue else ""
    return {
        "current_module": next_module,
        "module_queue": queue,
        "iteration": 0,
        "score": 0.0,
        "previous_score": 0.0,
        "current_draft": "",
        "critique": {},
        "analyst_output": {},
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_pipeline() -> CompiledStateGraph:
    """Build and compile the 5-agent LangGraph pipeline.

    Returns a compiled LangGraph state machine ready for ``.invoke()``.
    """
    from tribal_knowledge.pipeline.agents.analyst import analyst_node
    from tribal_knowledge.pipeline.agents.critic import critic_node
    from tribal_knowledge.pipeline.agents.explorer import explorer_node
    from tribal_knowledge.pipeline.agents.fixer import fixer_node
    from tribal_knowledge.pipeline.agents.synthesizer import synthesize_node
    from tribal_knowledge.pipeline.agents.writer import writer_node

    graph: StateGraph = StateGraph(PipelineState)

    # -- Add nodes --------------------------------------------------------
    graph.add_node("explorer", explorer_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("fixer", fixer_node)
    graph.add_node("save_module", save_module_doc_node)
    graph.add_node("next_module", advance_module_queue_node)
    graph.add_node("synthesize", synthesize_node)

    # -- Entry point ------------------------------------------------------
    graph.set_entry_point("explorer")

    # -- Linear edges -----------------------------------------------------
    graph.add_edge("explorer", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "critic")

    # -- Critic -> conditional: pass or fix -------------------------------
    graph.add_conditional_edges(
        "critic",
        should_fix_or_pass,
        {"pass": "save_module", "fix": "fixer"},
    )

    # -- Fixer -> back to Critic (refinement loop) ------------------------
    graph.add_edge("fixer", "critic")

    # -- Save -> next module or synthesize --------------------------------
    graph.add_conditional_edges(
        "save_module",
        has_more_modules,
        {"next": "next_module", "done": "synthesize"},
    )

    # -- Next module -> Analyst -------------------------------------------
    graph.add_edge("next_module", "analyst")

    # -- Synthesize -> END ------------------------------------------------
    graph.add_edge("synthesize", END)

    return graph.compile()
