"""Detect documentation coverage gaps after module docs are generated.

Compares generated documentation against the knowledge graph to find
uncovered files, missing high-PageRank entities, orphan configs, and
undocumented flows.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath

import networkx as nx
from pydantic import BaseModel, Field

from tribal_knowledge.models import FileExtraction, GraphAnalysis, ModuleInfo


# ── Models ───────────────────────────────────────────────────────────────────


class CoverageGap(BaseModel):
    """A single documentation coverage gap."""

    gap_type: str  # "uncovered_file" | "missing_entity" | "undocumented_flow" | "orphan_config"
    description: str
    severity: float  # 0.0–1.0
    affected_files: list[str] = Field(default_factory=list)
    suggested_module: str = ""


# ── Internal helpers ─────────────────────────────────────────────────────────


def _collect_file_paths_from_graph(G: nx.DiGraph) -> set[str]:
    """Return all unique file paths referenced by function/class nodes."""
    paths: set[str] = set()
    for _node_id, data in G.nodes(data=True):
        kind: str = data.get("kind", "")
        if kind in ("function", "class"):
            file_attr: str = data.get("file", "")
            if file_attr:
                paths.add(file_attr)
    return paths


def _max_pagerank_for_file(G: nx.DiGraph, file_path: str) -> float:
    """Return the maximum pagerank among entities belonging to *file_path*."""
    max_pr: float = 0.0
    for _node_id, data in G.nodes(data=True):
        if data.get("file") == file_path:
            pr: float = float(data.get("pagerank", 0.0))
            if pr > max_pr:
                max_pr = pr
    return max_pr


def _top_entities_globally(
    G: nx.DiGraph,
    limit: int = 20,
) -> list[tuple[str, str, float]]:
    """Return the top-N entities (function or class) sorted by pagerank.

    Returns list of (node_id, entity_name, pagerank_score).
    """
    candidates: list[tuple[str, str, float]] = []
    for node_id, data in G.nodes(data=True):
        kind: str = data.get("kind", "")
        if kind in ("function", "class"):
            name: str = data.get("name", node_id)
            pr: float = float(data.get("pagerank", 0.0))
            candidates.append((node_id, name, pr))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:limit]


def _text_mentions_path(text: str, file_path: str) -> bool:
    """Check whether *file_path* (or its basename) appears in *text*."""
    if file_path in text:
        return True
    # Also check the basename for references like "models.py"
    basename = PurePosixPath(file_path).name
    if basename and basename in text:
        return True
    return False


def _text_mentions_entity(text: str, entity_name: str) -> bool:
    """Check whether *entity_name* appears in *text*.

    Uses a simple substring search. Entity names are typically
    function/class identifiers so substring matching is adequate.
    """
    return entity_name in text


def _all_docs_text(module_docs: dict[str, str]) -> str:
    """Concatenate all module doc texts into one searchable string."""
    return "\n".join(module_docs.values())


def _longest_common_prefix_length(path_a: str, path_b: str) -> int:
    """Return the length of the longest common path-component prefix."""
    parts_a = PurePosixPath(path_a).parts
    parts_b = PurePosixPath(path_b).parts
    common = 0
    for a, b in zip(parts_a, parts_b):
        if a == b:
            common += 1
        else:
            break
    return common


# ── Public API ───────────────────────────────────────────────────────────────


def detect_gaps(
    module_docs: dict[str, str],
    knowledge_graph: nx.DiGraph,
    analysis: GraphAnalysis,
    extractions: list[FileExtraction],
) -> list[CoverageGap]:
    """Detect documentation coverage gaps.

    Checks three categories:

    1. **Uncovered files** -- files present in the graph (via function/class
       nodes) that are not mentioned in any generated module doc.
    2. **Missing high-PageRank entities** -- the top-20 entities globally
       (by pagerank) that do not appear in any module doc.
    3. **Orphan configs** -- config-type extractions with tribal signals
       whose file path is not referenced in any module doc.

    Returns a list of :class:`CoverageGap` instances sorted by severity
    descending.
    """
    gaps: list[CoverageGap] = []
    combined_text = _all_docs_text(module_docs)

    # ── (a) Uncovered files ──────────────────────────────────────────────

    file_paths = _collect_file_paths_from_graph(knowledge_graph)

    for fpath in sorted(file_paths):
        mentioned = False
        for _mod_name, doc_text in module_docs.items():
            if _text_mentions_path(doc_text, fpath):
                mentioned = True
                break
        if not mentioned:
            severity = _max_pagerank_for_file(knowledge_graph, fpath)
            suggested = suggest_module_for_file(fpath, module_docs, analysis)
            gaps.append(
                CoverageGap(
                    gap_type="uncovered_file",
                    description=f"File '{fpath}' is in the knowledge graph but not mentioned in any module doc.",
                    severity=min(severity, 1.0),
                    affected_files=[fpath],
                    suggested_module=suggested,
                )
            )

    # ── (b) Missing high-PageRank entities ───────────────────────────────

    top_entities = _top_entities_globally(knowledge_graph, limit=20)

    for node_id, entity_name, pr_score in top_entities:
        if not _text_mentions_entity(combined_text, entity_name):
            file_path: str = knowledge_graph.nodes[node_id].get("file", "")
            suggested = suggest_module_for_file(file_path, module_docs, analysis) if file_path else ""
            affected = [file_path] if file_path else []
            gaps.append(
                CoverageGap(
                    gap_type="missing_entity",
                    description=(
                        f"High-PageRank entity '{entity_name}' "
                        f"(score={pr_score:.6f}) is not mentioned in any module doc."
                    ),
                    severity=min(pr_score, 1.0),
                    affected_files=affected,
                    suggested_module=suggested,
                )
            )

    # ── (c) Orphan configs ───────────────────────────────────────────────

    for extraction in extractions:
        if extraction.file_type != "config":
            continue
        if not extraction.tribal_signals:
            continue
        mentioned = False
        for _mod_name, doc_text in module_docs.items():
            if _text_mentions_path(doc_text, extraction.file_path):
                mentioned = True
                break
        if not mentioned:
            signal_count = len(extraction.tribal_signals)
            # Severity scales with the number of tribal signals, capped at 1.0
            severity = min(signal_count * 0.2, 1.0)
            suggested = suggest_module_for_file(
                extraction.file_path, module_docs, analysis
            )
            gaps.append(
                CoverageGap(
                    gap_type="orphan_config",
                    description=(
                        f"Config file '{extraction.file_path}' has {signal_count} tribal "
                        f"signal(s) but is not referenced in any module doc."
                    ),
                    severity=severity,
                    affected_files=[extraction.file_path],
                    suggested_module=suggested,
                )
            )

    # Sort all gaps by severity descending
    gaps.sort(key=lambda g: g.severity, reverse=True)
    return gaps


def suggest_module_for_file(
    file_path: str,
    module_docs: dict[str, str],
    analysis: GraphAnalysis,
) -> str:
    """Suggest which module a file belongs to.

    First checks if any module in *analysis.modules* explicitly lists the
    file. If not, falls back to the module whose files share the longest
    common path prefix with *file_path*.
    """
    if not file_path:
        return ""

    # Direct lookup: is the file listed in a module?
    for module in analysis.modules:
        if file_path in module.files:
            return module.name

    # Fallback: longest common path-prefix heuristic
    best_module: str = ""
    best_prefix_len: int = 0

    for module in analysis.modules:
        for mf in module.files:
            prefix_len = _longest_common_prefix_length(file_path, mf)
            if prefix_len > best_prefix_len:
                best_prefix_len = prefix_len
                best_module = module.name

    return best_module


def format_gap_report(gaps: list[CoverageGap]) -> str:
    """Format gaps as a human-readable Markdown report.

    Gaps are grouped by ``gap_type`` and sorted by severity descending
    within each group. The groups themselves are ordered by the maximum
    severity of their members (highest first).
    """
    if not gaps:
        return "# Coverage Gap Report\n\nNo documentation gaps detected."

    # Group by gap_type
    grouped: dict[str, list[CoverageGap]] = defaultdict(list)
    for gap in gaps:
        grouped[gap.gap_type].append(gap)

    # Sort each group internally by severity descending
    for gap_list in grouped.values():
        gap_list.sort(key=lambda g: g.severity, reverse=True)

    # Order groups by max severity descending
    sorted_groups: list[tuple[str, list[CoverageGap]]] = sorted(
        grouped.items(),
        key=lambda item: item[1][0].severity,
        reverse=True,
    )

    # Human-friendly type labels
    type_labels: dict[str, str] = {
        "uncovered_file": "Uncovered Files",
        "missing_entity": "Missing High-PageRank Entities",
        "undocumented_flow": "Undocumented Flows",
        "orphan_config": "Orphan Configurations",
    }

    lines: list[str] = [
        "# Coverage Gap Report",
        "",
        f"**Total gaps found:** {len(gaps)}",
        "",
    ]

    for gap_type, gap_list in sorted_groups:
        heading = type_labels.get(gap_type, gap_type.replace("_", " ").title())
        lines.append(f"## {heading}")
        lines.append("")

        for gap in gap_list:
            severity_pct = f"{gap.severity:.0%}"
            lines.append(f"- **Severity {severity_pct}** -- {gap.description}")
            if gap.affected_files:
                files_str = ", ".join(f"`{f}`" for f in gap.affected_files)
                lines.append(f"  - Affected files: {files_str}")
            if gap.suggested_module:
                lines.append(f"  - Suggested module: *{gap.suggested_module}*")

        lines.append("")

    return "\n".join(lines)
