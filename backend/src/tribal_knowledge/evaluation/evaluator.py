"""Main evaluation entry point -- scores all generated context documents.

Loads ``.md`` files from the context directory, runs the four automated
scorers plus the LLM-as-judge, and writes ``evaluation-report.json``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from tribal_knowledge.evaluation.llm_judge import judge_usefulness
from tribal_knowledge.evaluation.models import DocumentEvaluation, EvaluationReport
from tribal_knowledge.evaluation.scorer import (
    conciseness_score,
    entity_coverage_score,
    factual_accuracy_score,
    structural_completeness_score,
)
from tribal_knowledge.models import GraphAnalysis
from tribal_knowledge.pipeline.helpers import get_all_entity_names, get_top_entities_for_module


# Synthesis document filenames (not per-module)
_SYNTHESIS_DOCS = frozenset({"OVERVIEW.md", "CONVENTIONS.md", "AGENTS.md"})


def _is_synthesis_doc(filename: str) -> bool:
    """Return True if *filename* is a synthesis-level document."""
    return filename in _SYNTHESIS_DOCS


def _module_name_from_filename(filename: str) -> str:
    """Derive a module name from a markdown filename.

    ``my-module.md`` -> ``my-module``
    """
    return filename.removesuffix(".md")


def _get_top_global_entities(
    graph: nx.DiGraph,
    k: int = 20,
) -> list[str]:
    """Return the top-*k* entity names globally by PageRank."""
    candidates: list[tuple[str, float]] = []
    for _node_id, attrs in graph.nodes(data=True):
        pr = attrs.get("pagerank", 0.0)
        name = attrs.get("name", "")
        if isinstance(pr, (int, float)) and isinstance(name, str) and name:
            candidates.append((name, float(pr)))
    candidates.sort(key=lambda t: t[1], reverse=True)
    return [name for name, _ in candidates[:k]]


def evaluate_document(
    doc_text: str,
    doc_name: str,
    top_entities: list[str],
    entity_names_set: set[str],
    repo_file_paths: set[str],
) -> DocumentEvaluation:
    """Run all scorers against a single document and return its evaluation."""
    # 1. Entity coverage
    cov_score, missing = entity_coverage_score(doc_text, top_entities)

    # 2. Factual accuracy
    acc_score, invalid_refs = factual_accuracy_score(doc_text, entity_names_set, repo_file_paths)

    # 3. Conciseness
    con_score, issues = conciseness_score(doc_text)

    # 4. Structural completeness
    struct_score = structural_completeness_score(doc_text)

    # 5. LLM usefulness judge
    module_name = _module_name_from_filename(doc_name)
    usefulness_score, _reasoning = judge_usefulness(doc_text, module_name)

    return DocumentEvaluation(
        name=doc_name,
        entity_coverage=round(cov_score, 4),
        factual_accuracy=round(acc_score, 4),
        conciseness=round(con_score, 4),
        structural_completeness=round(struct_score, 4),
        llm_usefulness=round(usefulness_score, 4),
        invalid_references=invalid_refs,
        missing_entities=missing,
        issues=issues,
    )


def _aggregate_report(evaluations: list[DocumentEvaluation]) -> EvaluationReport:
    """Build an EvaluationReport from individual document evaluations."""
    if not evaluations:
        return EvaluationReport(evaluated_at=datetime.now(timezone.utc).isoformat())

    composites = [e.composite_score for e in evaluations]
    avg_composite = sum(composites) / len(composites)

    dimension_sums: dict[str, float] = {
        "entity_coverage": 0.0,
        "factual_accuracy": 0.0,
        "conciseness": 0.0,
        "structural_completeness": 0.0,
        "llm_usefulness": 0.0,
    }
    for ev in evaluations:
        dimension_sums["entity_coverage"] += ev.entity_coverage
        dimension_sums["factual_accuracy"] += ev.factual_accuracy
        dimension_sums["conciseness"] += ev.conciseness
        dimension_sums["structural_completeness"] += ev.structural_completeness
        dimension_sums["llm_usefulness"] += ev.llm_usefulness

    count = len(evaluations)
    average_by_dimension = {k: round(v / count, 4) for k, v in dimension_sums.items()}

    total_invalid = sum(len(e.invalid_references) for e in evaluations)
    total_missing = sum(len(e.missing_entities) for e in evaluations)

    return EvaluationReport(
        documents=evaluations,
        average_composite=round(avg_composite, 4),
        average_by_dimension=average_by_dimension,
        total_invalid_references=total_invalid,
        total_missing_entities=total_missing,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )


def evaluate_all(
    context_dir: Path,
    knowledge_graph: nx.DiGraph,
    analysis: GraphAnalysis,
    repo_file_paths: set[str],
) -> EvaluationReport:
    """Score every ``.md`` file in *context_dir* and write an evaluation report.

    Parameters
    ----------
    context_dir:
        Directory containing the generated ``.md`` context files.
    knowledge_graph:
        The full knowledge graph (NetworkX DiGraph) built in Phase 1.
    analysis:
        The GraphAnalysis summary produced by Phase 1.
    repo_file_paths:
        Set of all file paths in the analysed repository (relative paths).

    Returns
    -------
    EvaluationReport
        The complete evaluation with per-document and aggregate scores.
    """
    md_files = sorted(context_dir.glob("*.md"))
    if not md_files:
        report = EvaluationReport(evaluated_at=datetime.now(timezone.utc).isoformat())
        _write_report(report, context_dir)
        return report

    # Pre-compute shared data
    all_entity_names: list[str] = get_all_entity_names(knowledge_graph)
    entity_names_set: set[str] = set(all_entity_names)
    top_global_entities: list[str] = _get_top_global_entities(knowledge_graph, k=20)

    # Build module name -> ModuleInfo lookup for top-entity queries
    module_names: set[str] = {m.name for m in analysis.modules}

    evaluations: list[DocumentEvaluation] = []

    for md_path in md_files:
        doc_name = md_path.name
        doc_text = md_path.read_text(encoding="utf-8")

        if _is_synthesis_doc(doc_name):
            # Synthesis docs use top-20 global entities
            top_entities = top_global_entities
        else:
            # Per-module docs: derive the module name and get top-10 entities
            module_name = _module_name_from_filename(doc_name)

            # Try to match the filename-derived module name against known modules
            matched_module = _find_matching_module(module_name, module_names)
            if matched_module:
                top_entities = get_top_entities_for_module(
                    knowledge_graph, matched_module, k=10,
                )
            else:
                # Fallback: use the module's top_entities from analysis
                top_entities = _get_module_entities_from_analysis(module_name, analysis)

        evaluation = evaluate_document(
            doc_text=doc_text,
            doc_name=doc_name,
            top_entities=top_entities,
            entity_names_set=entity_names_set,
            repo_file_paths=repo_file_paths,
        )
        evaluations.append(evaluation)

    report = _aggregate_report(evaluations)
    _write_report(report, context_dir)
    return report


def _find_matching_module(filename_module: str, module_names: set[str]) -> str | None:
    """Fuzzy-match a filename-derived module name to a known module.

    Tries exact match first, then normalised comparison (lowercase,
    hyphens replaced with underscores and vice versa).
    """
    if filename_module in module_names:
        return filename_module

    # Normalise: lowercase, treat hyphens and underscores as equivalent
    normalised_target = filename_module.lower().replace("-", "_")
    for name in module_names:
        if name.lower().replace("-", "_") == normalised_target:
            return name

    return None


def _get_module_entities_from_analysis(
    module_name: str,
    analysis: GraphAnalysis,
) -> list[str]:
    """Fallback: pull top_entities from the GraphAnalysis ModuleInfo list."""
    normalised = module_name.lower().replace("-", "_")
    for mod in analysis.modules:
        if mod.name.lower().replace("-", "_") == normalised:
            return mod.top_entities[:10]
    return []


def _write_report(report: EvaluationReport, context_dir: Path) -> None:
    """Serialise the evaluation report to ``evaluation-report.json``."""
    report_path = context_dir / "evaluation-report.json"
    payload = report.model_dump()
    # Include composite_score per document (it's a @property, not in model_dump)
    for i, doc_eval in enumerate(report.documents):
        payload["documents"][i]["composite_score"] = round(doc_eval.composite_score, 4)
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
