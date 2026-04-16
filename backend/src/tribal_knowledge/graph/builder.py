"""Build a NetworkX DiGraph from FileExtractions and a DependencyGraph."""

from __future__ import annotations

import networkx as nx

from tribal_knowledge.models import (
    ClassDef,
    Constant,
    DependencyGraph,
    FileExtraction,
    FunctionDef,
    RouteHandler,
    TypeDef,
)


# ── Helper: add individual node types ──────────────────────────────────────


def _add_file_node(G: nx.DiGraph, ext: FileExtraction) -> None:
    """Add a file node with its metadata."""
    node_id = f"file::{ext.file_path}"
    G.add_node(
        node_id,
        kind="file",
        name=ext.file_path,
        file=ext.file_path,
        path=ext.file_path,
        language=ext.language,
        file_type=ext.file_type,
        line_count=ext.line_count,
    )


def _add_class_node(G: nx.DiGraph, file_path: str, cls: ClassDef) -> None:
    """Add a class node."""
    node_id = f"class::{file_path}::{cls.name}"
    G.add_node(
        node_id,
        kind="class",
        name=cls.name,
        file=file_path,
        bases=cls.bases,
        docstring=cls.docstring or "",
        decorators=cls.decorators,
        line_start=cls.line_start,
        line_end=cls.line_end,
    )


def _add_function_node(
    G: nx.DiGraph,
    file_path: str,
    func: FunctionDef,
    class_name: str = "",
) -> None:
    """Add a function/method node."""
    if class_name:
        node_id = f"func::{file_path}::{class_name}.{func.name}"
    else:
        node_id = f"func::{file_path}::{func.name}"
    G.add_node(
        node_id,
        kind="function",
        name=func.name,
        file=file_path,
        class_name=class_name,
        qualified_name=func.qualified_name,
        signature=func.signature,
        is_async=func.is_async,
        docstring=func.docstring or "",
        decorators=func.decorators,
        parameters=func.parameters,
        return_type=func.return_type or "",
        line_start=func.line_start,
        line_end=func.line_end,
    )


def _add_type_node(G: nx.DiGraph, file_path: str, type_def: TypeDef) -> None:
    """Add a type/interface/enum node."""
    node_id = f"type::{file_path}::{type_def.name}"
    G.add_node(
        node_id,
        kind="type",
        name=type_def.name,
        file=file_path,
        type_kind=type_def.kind,
        line_start=type_def.line_start,
        line_end=type_def.line_end,
    )


def _add_constant_node(G: nx.DiGraph, file_path: str, const: Constant) -> None:
    """Add a config/constant node."""
    node_id = f"config::{file_path}::{const.name}"
    G.add_node(
        node_id,
        kind="config",
        name=const.name,
        file=file_path,
        value=const.value or "",
        line=const.line,
    )


def _add_route_node(G: nx.DiGraph, file_path: str, route: RouteHandler) -> None:
    """Add a route node."""
    node_id = f"route::{route.path}::{route.method}"
    # Only add if not already present (same route from different files would collide,
    # but we keep the first one).
    if node_id not in G:
        G.add_node(
            node_id,
            kind="route",
            name=f"{route.method} {route.path}",
            file=file_path,
            path=route.path,
            method=route.method,
        )


# ── Lookup helpers ─────────────────────────────────────────────────────────


def _build_class_lookup(G: nx.DiGraph) -> dict[str, list[str]]:
    """Build a mapping from class name → list of node IDs for resolving bases."""
    lookup: dict[str, list[str]] = {}
    for node_id, data in G.nodes(data=True):
        if data.get("kind") == "class":
            name: str = data["name"]
            lookup.setdefault(name, []).append(node_id)
    return lookup


def _build_func_lookup(G: nx.DiGraph) -> dict[str, list[str]]:
    """Build a mapping from function/method name → list of node IDs."""
    lookup: dict[str, list[str]] = {}
    for node_id, data in G.nodes(data=True):
        if data.get("kind") == "function":
            name = data["name"]
            lookup.setdefault(name, []).append(node_id)
    return lookup


# ── Main builder ───────────────────────────────────────────────────────────


def build_knowledge_graph(
    extractions: list[FileExtraction],
    dependency_graph: DependencyGraph,
) -> nx.DiGraph:
    """Build a NetworkX DiGraph from parsed file extractions and dependency info.

    The graph contains typed nodes (file, class, function, type, config, route)
    and typed edges (CONTAINS, CALLS, IMPORTS, INHERITS, HANDLES).
    """
    G = nx.DiGraph()

    # ── Pass 1: Add all nodes from extractions ─────────────────────────────

    for ext in extractions:
        _add_file_node(G, ext)

        for cls in ext.classes:
            _add_class_node(G, ext.file_path, cls)
            for method in cls.methods:
                _add_function_node(G, ext.file_path, method, class_name=cls.name)

        for func in ext.functions:
            _add_function_node(G, ext.file_path, func)

        for type_def in ext.type_defs:
            _add_type_node(G, ext.file_path, type_def)

        for const in ext.constants:
            _add_constant_node(G, ext.file_path, const)

        for route in ext.route_handlers:
            _add_route_node(G, ext.file_path, route)

    # ── Pass 2: Add IMPORTS edges from DependencyGraph.resolved_imports ────

    for imp in dependency_graph.resolved_imports:
        source_id = f"file::{imp.source_file}"
        target_id = f"file::{imp.target_file}"
        # Only add edge if both nodes exist in the graph
        if source_id in G and target_id in G:
            G.add_edge(source_id, target_id, relation="IMPORTS")

    # ── Pass 3: Add CALLS edges from DependencyGraph.call_expressions ─────

    for call in dependency_graph.call_expressions:
        if call.callee_resolved:
            caller_id = f"func::{call.caller}"
            callee_id = f"func::{call.callee_resolved}"
            # Only add edge if both nodes exist
            if caller_id in G and callee_id in G:
                G.add_edge(caller_id, callee_id, relation="CALLS")

    # ── Pass 4: Add CONTAINS edges (file→class, class→method, file→func) ──

    for ext in extractions:
        file_id = f"file::{ext.file_path}"

        for cls in ext.classes:
            class_id = f"class::{ext.file_path}::{cls.name}"
            G.add_edge(file_id, class_id, relation="CONTAINS")

            for method in cls.methods:
                method_id = f"func::{ext.file_path}::{cls.name}.{method.name}"
                G.add_edge(class_id, method_id, relation="CONTAINS")

        for func in ext.functions:
            func_id = f"func::{ext.file_path}::{func.name}"
            G.add_edge(file_id, func_id, relation="CONTAINS")

        for type_def in ext.type_defs:
            type_id = f"type::{ext.file_path}::{type_def.name}"
            G.add_edge(file_id, type_id, relation="CONTAINS")

        for const in ext.constants:
            const_id = f"config::{ext.file_path}::{const.name}"
            G.add_edge(file_id, const_id, relation="CONTAINS")

    # ── Pass 5: Add INHERITS edges (class.bases → resolve against classes) ─

    class_lookup = _build_class_lookup(G)

    for ext in extractions:
        for cls in ext.classes:
            class_id = f"class::{ext.file_path}::{cls.name}"
            for base_name in cls.bases:
                # Try to resolve the base class by name
                candidates = class_lookup.get(base_name, [])
                for candidate_id in candidates:
                    if candidate_id != class_id:
                        G.add_edge(class_id, candidate_id, relation="INHERITS")

    # ── Pass 6: Add HANDLES edges (function → route) from route_handlers ───

    func_lookup = _build_func_lookup(G)

    for ext in extractions:
        for route in ext.route_handlers:
            route_id = f"route::{route.path}::{route.method}"
            if route_id not in G:
                continue
            handler_name = route.handler_function
            if not handler_name:
                continue
            # Try to find the handler function in the same file first
            file_candidates = [
                nid
                for nid in func_lookup.get(handler_name, [])
                if G.nodes[nid].get("file") == ext.file_path
            ]
            if file_candidates:
                for fid in file_candidates:
                    G.add_edge(fid, route_id, relation="HANDLES")
            else:
                # Fall back to any function with that name
                for fid in func_lookup.get(handler_name, []):
                    G.add_edge(fid, route_id, relation="HANDLES")

    return G
