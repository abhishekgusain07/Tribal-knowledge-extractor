"""Shared helper functions used by multiple pipeline agents.

These extract and format data from PipelineState so agent nodes stay concise.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import networkx as nx
from networkx.readwrite import json_graph

from tribal_knowledge.models import FileExtraction, GraphAnalysis
from tribal_knowledge.pipeline.pipeline_models import ModuleMapEntry


# ── Graph helpers ────────────────────────────────────────────────────


def load_graph_from_state(state: dict[str, object]) -> nx.DiGraph:
    """Reconstruct a NetworkX DiGraph from the serialised data in state."""
    data = state.get("knowledge_graph_data")
    if data is None:
        return nx.DiGraph()
    # json_graph.node_link_graph expects plain dicts
    if isinstance(data, str):
        data = json.loads(data)
    return json_graph.node_link_graph(data, directed=True)  # type: ignore[arg-type]


def get_all_entity_names(G: nx.DiGraph) -> list[str]:
    """Return a sorted list of every entity name in the graph."""
    names: list[str] = []
    for _, attrs in G.nodes(data=True):
        name = attrs.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return sorted(set(names))


def get_top_entities_for_module(
    G: nx.DiGraph,
    module_name: str,
    k: int = 10,
) -> list[str]:
    """Return the top-*k* entity names in *module_name* by PageRank."""
    candidates: list[tuple[str, float]] = []
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("module") == module_name:
            pr = attrs.get("pagerank", 0.0)
            name = attrs.get("name", "")
            if isinstance(pr, (int, float)) and isinstance(name, str) and name:
                candidates.append((name, float(pr)))
    candidates.sort(key=lambda t: t[1], reverse=True)
    return [name for name, _ in candidates[:k]]


def extract_module_subgraph_summary(
    G: nx.DiGraph,
    file_patterns: list[str],
) -> str:
    """Return a human-readable summary of the call-chains inside a module.

    *file_patterns* are the file paths (or globs) belonging to the module.
    """
    # Collect nodes whose file matches any pattern
    module_nodes: set[str] = set()
    for node_id, attrs in G.nodes(data=True):
        file_val = attrs.get("file", "")
        if isinstance(file_val, str) and any(
            file_val.startswith(p.rstrip("*").rstrip("/")) for p in file_patterns
        ):
            module_nodes.add(node_id)

    lines: list[str] = []
    for u, v, data in G.edges(data=True):
        if u in module_nodes or v in module_nodes:
            rel = data.get("relation", "?")
            if rel in ("CALLS", "IMPORTS"):
                u_name = G.nodes[u].get("name", u.split("::")[-1])
                v_name = G.nodes[v].get("name", v.split("::")[-1])
                direction = "→"
                label = f"{u_name} {direction} {v_name} [{rel}]"
                if label not in lines:
                    lines.append(label)
    return "\n".join(lines[:60]) if lines else "(no call/import edges found)"


def format_cross_module_calls(
    G: nx.DiGraph,
    module_name: str,
) -> str:
    """Return a summary of edges that cross from/to *module_name*."""
    lines: list[str] = []
    for u, v, data in G.edges(data=True):
        u_mod = G.nodes[u].get("module", "")
        v_mod = G.nodes[v].get("module", "")
        rel = data.get("relation", "")
        if rel not in ("CALLS", "IMPORTS"):
            continue
        if u_mod == module_name and v_mod != module_name and v_mod:
            u_name = G.nodes[u].get("name", u.split("::")[-1])
            v_name = G.nodes[v].get("name", v.split("::")[-1])
            lines.append(f"{u_name} → {v_name} (in {v_mod}) [{rel}]")
        elif v_mod == module_name and u_mod != module_name and u_mod:
            u_name = G.nodes[u].get("name", u.split("::")[-1])
            v_name = G.nodes[v].get("name", v.split("::")[-1])
            lines.append(f"{u_name} (in {u_mod}) → {v_name} [{rel}]")
    return "\n".join(lines[:40]) if lines else "(no cross-module edges)"


# ── ChromaDB helpers ─────────────────────────────────────────────────


def get_chromadb_collection(chromadb_path: str) -> chromadb.Collection:
    """Open the persistent ChromaDB collection created in Phase 1."""
    client = chromadb.PersistentClient(path=chromadb_path)
    return client.get_collection("code_chunks")


def query_chunks_for_module(
    collection: chromadb.Collection,
    module_description: str,
    module_name: str,
    n_results: int = 30,
) -> str:
    """Retrieve and format the most relevant code chunks for a module."""
    try:
        results = collection.query(
            query_texts=[module_description],
            n_results=n_results,
            where={"module": module_name},
        )
    except Exception:
        # Fallback: query without module filter if the module name doesn't match
        try:
            results = collection.query(
                query_texts=[module_description],
                n_results=n_results,
            )
        except Exception:
            return "(no code chunks available)"

    documents = results.get("documents", [[]])
    if not documents or not documents[0]:
        return "(no code chunks matched)"

    chunks: list[str] = []
    for doc in documents[0][:20]:
        if isinstance(doc, str):
            chunks.append(doc.strip())
    return "\n\n---\n\n".join(chunks) if chunks else "(no code chunks matched)"


# ── Extraction helpers ───────────────────────────────────────────────


def load_extractions_from_state(
    state: dict[str, object],
) -> list[FileExtraction]:
    """Deserialise extractions from state."""
    raw = state.get("extractions_data", [])
    if not isinstance(raw, list):
        return []
    results: list[FileExtraction] = []
    for item in raw:
        if isinstance(item, dict):
            try:
                results.append(FileExtraction.model_validate(item))
            except Exception:
                continue
    return results


def get_tribal_signals_for_module(
    extractions: list[FileExtraction],
    file_patterns: list[str],
) -> str:
    """Return formatted tribal signals for files matching *file_patterns*."""
    lines: list[str] = []
    for ext in extractions:
        if not any(
            ext.file_path.startswith(p.rstrip("*").rstrip("/"))
            for p in file_patterns
        ):
            continue
        for sig in ext.tribal_signals:
            lines.append(f"  [{ext.file_path}:{sig.line_number}] ({sig.category}) {sig.line_content.strip()}")
    return "\n".join(lines[:50]) if lines else "(no tribal signals found)"


# ── Module map helpers ───────────────────────────────────────────────


def get_module_entry(
    module_map: list[dict[str, object]],
    module_name: str,
) -> ModuleMapEntry:
    """Find the ModuleMapEntry for *module_name*."""
    for entry in module_map:
        if isinstance(entry, dict) and entry.get("name") == module_name:
            return ModuleMapEntry.model_validate(entry)
    return ModuleMapEntry(name=module_name, description="")


def format_module_list(module_map: list[dict[str, object]]) -> str:
    """Format a concise list of modules for prompts."""
    lines: list[str] = []
    for entry in module_map:
        if isinstance(entry, dict):
            name = entry.get("name", "?")
            desc = entry.get("description", "")
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


# ── Formatting helpers ───────────────────────────────────────────────


def format_top_entities(analysis: GraphAnalysis) -> str:
    """Format the top entities from analysis for the Explorer prompt."""
    lines: list[str] = []
    for item in analysis.top_functions[:20]:
        name = item.get("name", "?")
        file_val = item.get("file", "?")
        pr = item.get("pagerank", 0)
        lines.append(f"  func: {name} ({file_val}) [pagerank={pr:.4f}]")
    for item in analysis.top_classes[:10]:
        name = item.get("name", "?")
        file_val = item.get("file", "?")
        pr = item.get("pagerank", 0)
        lines.append(f"  class: {name} ({file_val}) [pagerank={pr:.4f}]")
    return "\n".join(lines) if lines else "(no top entities)"


def format_modules_for_prompt(modules: list[object]) -> str:
    """Format Phase-1 modules for the Explorer."""
    from tribal_knowledge.models import ModuleInfo

    lines: list[str] = []
    for m in modules:
        if isinstance(m, ModuleInfo):
            lines.append(
                f"- {m.name}: {len(m.files)} files, cohesion={m.cohesion:.2f}, "
                f"top entities: {', '.join(m.top_entities[:5])}"
            )
        elif isinstance(m, dict):
            info = ModuleInfo.model_validate(m)
            lines.append(
                f"- {info.name}: {len(info.files)} files, cohesion={info.cohesion:.2f}, "
                f"top entities: {', '.join(info.top_entities[:5])}"
            )
    return "\n".join(lines) if lines else "(no modules detected)"


def build_file_tree_summary(repo_path: str, max_depth: int = 4) -> str:
    """Build an indented file tree string from a repo directory."""
    root = Path(repo_path)
    if not root.is_dir():
        return "(not a directory)"

    skip_dirs = {
        "node_modules", ".git", "__pycache__", ".next", "dist", "build",
        ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache",
        "vendor", ".cargo", "target", ".tribal-knowledge",
    }

    lines: list[str] = []
    _walk_tree(root, root, skip_dirs, lines, depth=0, max_depth=max_depth)
    return "\n".join(lines[:300])


def _walk_tree(
    current: Path,
    root: Path,
    skip_dirs: set[str],
    lines: list[str],
    depth: int,
    max_depth: int,
) -> None:
    if depth > max_depth:
        return
    indent = "  " * depth
    try:
        entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except PermissionError:
        return
    for entry in entries:
        if entry.name.startswith(".") and entry.is_dir() and entry.name not in (".github",):
            continue
        if entry.is_dir():
            if entry.name in skip_dirs:
                continue
            lines.append(f"{indent}{entry.name}/")
            _walk_tree(entry, root, skip_dirs, lines, depth + 1, max_depth)
        else:
            lines.append(f"{indent}{entry.name}")


def format_package_configs(extractions: list[FileExtraction]) -> str:
    """Extract package config file names and constants from extractions."""
    lines: list[str] = []
    for ext in extractions:
        if ext.file_type == "config":
            lines.append(f"Config: {ext.file_path}")
        elif ext.file_type == "doc":
            lines.append(f"Doc: {ext.file_path}")
    return "\n".join(lines[:30]) if lines else "(no config files found)"
