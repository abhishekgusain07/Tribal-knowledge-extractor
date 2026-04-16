"""Save and load knowledge graph, analysis, extractions, and manifest to JSON."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from tribal_knowledge.models import (
    FileExtraction,
    GraphAnalysis,
    ParseManifest,
)


def _ensure_dir(output_dir: Path) -> Path:
    """Create the output directory (and parents) if it does not exist."""
    tk_dir = output_dir / ".tribal-knowledge"
    tk_dir.mkdir(parents=True, exist_ok=True)
    return tk_dir


# ── Knowledge Graph ────────────────────────────────────────────────────────


def save_graph(G: nx.DiGraph, output_dir: Path) -> None:
    """Serialize a NetworkX DiGraph to ``{output_dir}/.tribal-knowledge/knowledge_graph.json``."""
    tk_dir = _ensure_dir(output_dir)
    data = json_graph.node_link_data(G)
    (tk_dir / "knowledge_graph.json").write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )


def load_graph(output_dir: Path) -> nx.DiGraph:
    """Load a NetworkX DiGraph from ``{output_dir}/.tribal-knowledge/knowledge_graph.json``."""
    tk_dir = output_dir / ".tribal-knowledge"
    raw = (tk_dir / "knowledge_graph.json").read_text(encoding="utf-8")
    data: dict[str, object] = json.loads(raw)
    graph: nx.DiGraph = json_graph.node_link_graph(data, directed=True)
    return graph


# ── Graph Analysis ─────────────────────────────────────────────────────────


def save_analysis(analysis: GraphAnalysis, output_dir: Path) -> None:
    """Serialize a ``GraphAnalysis`` to ``{output_dir}/.tribal-knowledge/analysis.json``."""
    tk_dir = _ensure_dir(output_dir)
    (tk_dir / "analysis.json").write_text(
        analysis.model_dump_json(indent=2),
        encoding="utf-8",
    )


def load_analysis(output_dir: Path) -> GraphAnalysis:
    """Load a ``GraphAnalysis`` from ``{output_dir}/.tribal-knowledge/analysis.json``."""
    tk_dir = output_dir / ".tribal-knowledge"
    raw = (tk_dir / "analysis.json").read_text(encoding="utf-8")
    return GraphAnalysis.model_validate_json(raw)


# ── Extractions ────────────────────────────────────────────────────────────


def save_extractions(extractions: list[FileExtraction], output_dir: Path) -> None:
    """Serialize all file extractions to ``{output_dir}/.tribal-knowledge/extractions.json``."""
    tk_dir = _ensure_dir(output_dir)
    data = [ext.model_dump() for ext in extractions]
    (tk_dir / "extractions.json").write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )


# ── Parse Manifest ─────────────────────────────────────────────────────────


def save_manifest(manifest: ParseManifest, output_dir: Path) -> None:
    """Serialize a ``ParseManifest`` to ``{output_dir}/.tribal-knowledge/manifest.json``."""
    tk_dir = _ensure_dir(output_dir)
    (tk_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )


def load_manifest(output_dir: Path) -> ParseManifest | None:
    """Load a ``ParseManifest`` from disk, or return ``None`` if it does not exist."""
    manifest_path = output_dir / ".tribal-knowledge" / "manifest.json"
    if not manifest_path.exists():
        return None
    raw = manifest_path.read_text(encoding="utf-8")
    return ParseManifest.model_validate_json(raw)
