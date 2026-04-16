"""PageRank scoring, Louvain community detection, and cross-module analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import PurePosixPath

import networkx as nx
from networkx.algorithms.community import louvain_communities

from tribal_knowledge.models import GraphAnalysis, ModuleDependency, ModuleInfo


# ── Helpers ────────────────────────────────────────────────────────────────


def _dominant_directory(files: list[str]) -> str:
    """Name a module by the most common top-level directory among its files.

    Falls back to "root" if all files live at the repo root.
    """
    dirs: list[str] = []
    for f in files:
        parts = PurePosixPath(f).parts
        if len(parts) > 1:
            dirs.append(parts[0])
        else:
            dirs.append("root")
    if not dirs:
        return "unnamed"
    counter = Counter(dirs)
    dominant, _count = counter.most_common(1)[0]
    return dominant


def _compute_cohesion(
    G_undirected: nx.Graph,
    community_nodes: list[str],
) -> tuple[int, int, float]:
    """Compute internal edges, external edges, and cohesion for a community.

    Cohesion = internal_edges / (internal_edges + external_edges).
    Returns (internal_edges, external_edges, cohesion).
    """
    node_set = set(community_nodes)
    internal = 0
    external = 0
    for node in community_nodes:
        if node not in G_undirected:
            continue
        for neighbor in G_undirected.neighbors(node):
            if neighbor in node_set:
                internal += 1
            else:
                external += 1
    # Each internal edge is counted twice (once from each endpoint)
    internal = internal // 2
    total = internal + external
    cohesion = internal / total if total > 0 else 0.0
    return internal, external, cohesion


def _node_counts(G: nx.DiGraph) -> dict[str, int]:
    """Count nodes by kind."""
    counts: dict[str, int] = defaultdict(int)
    for _node, data in G.nodes(data=True):
        kind: str = data.get("kind", "unknown")
        counts[kind] += 1
    return dict(counts)


def _edge_counts(G: nx.DiGraph) -> dict[str, int]:
    """Count edges by relation."""
    counts: dict[str, int] = defaultdict(int)
    for _u, _v, data in G.edges(data=True):
        relation: str = data.get("relation", "unknown")
        counts[relation] += 1
    return dict(counts)


def _top_by_pagerank(
    G: nx.DiGraph,
    kind: str,
    limit: int,
) -> list[dict[str, object]]:
    """Return the top-N nodes of a given kind, sorted by pagerank descending."""
    candidates: list[tuple[str, dict[str, object]]] = []
    for node_id, data in G.nodes(data=True):
        if data.get("kind") == kind:
            candidates.append((node_id, dict(data)))
    candidates.sort(key=lambda x: x[1].get("pagerank", 0.0), reverse=True)  # type: ignore[arg-type]
    result: list[dict[str, object]] = []
    for node_id, attrs in candidates[:limit]:
        entry: dict[str, object] = {"id": node_id}
        entry.update(attrs)
        result.append(entry)
    return result


# ── Main analysis ──────────────────────────────────────────────────────────


def analyze_graph(G: nx.DiGraph) -> GraphAnalysis:
    """Run PageRank and Louvain community detection on the knowledge graph.

    Mutates *G* in place (adds ``pagerank`` and ``module`` attributes to nodes).
    Returns a :class:`GraphAnalysis` summarising the results.
    """

    # ── 1. PageRank ────────────────────────────────────────────────────────

    pagerank_scores: dict[str, float] = nx.pagerank(G, alpha=0.85)
    for node_id, score in pagerank_scores.items():
        G.nodes[node_id]["pagerank"] = score

    # ── 2. Top entities ────────────────────────────────────────────────────

    top_functions = _top_by_pagerank(G, "function", 50)
    top_files = _top_by_pagerank(G, "file", 30)
    top_classes = _top_by_pagerank(G, "class", 20)

    # ── 3. Louvain community detection on file-level subgraph ──────────────

    file_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "file"]

    modules: list[ModuleInfo] = []
    communities: list[set[str]] = []
    modularity_score = 0.0

    if file_nodes:
        file_subgraph = G.to_undirected().subgraph(file_nodes)

        # Only run if there are edges; otherwise Louvain has nothing to partition
        if file_subgraph.number_of_edges() > 0:
            frozen_communities: list[frozenset[str]] = louvain_communities(
                file_subgraph, resolution=1.0, seed=42
            )
            communities = [set(c) for c in frozen_communities]

            # Compute modularity
            partition_for_modularity: list[set[str]] = communities
            try:
                modularity_score = nx.community.modularity(
                    file_subgraph, partition_for_modularity
                )
            except (nx.NetworkXError, ZeroDivisionError):
                modularity_score = 0.0
        else:
            # No edges between files — each file is its own community
            communities = [{n} for n in file_nodes]

        # Build ModuleInfo objects
        g_undirected = G.to_undirected()
        for idx, community_set in enumerate(communities):
            community_node_ids = sorted(community_set)
            # Strip the "file::" prefix to get plain file paths for naming/storage
            plain_paths = [
                nid.removeprefix("file::") for nid in community_node_ids
            ]
            module_name = _dominant_directory(plain_paths)

            # Deduplicate module names by appending an index if necessary
            existing_names = {m.name for m in modules}
            if module_name in existing_names:
                module_name = f"{module_name}-{idx}"

            internal, external, cohesion = _compute_cohesion(
                g_undirected, community_node_ids
            )

            # Top entities: functions/classes in this module's files, by pagerank
            child_entities: list[tuple[str, float]] = []
            for file_node_id in community_node_ids:
                for _u, v, edata in G.edges(file_node_id, data=True):
                    if edata.get("relation") == "CONTAINS":
                        pr: float = G.nodes[v].get("pagerank", 0.0)
                        child_entities.append((v, pr))
            child_entities.sort(key=lambda x: x[1], reverse=True)
            top_entity_ids = [eid for eid, _pr in child_entities[:5]]

            modules.append(
                ModuleInfo(
                    name=module_name,
                    files=plain_paths,
                    top_entities=top_entity_ids,
                    internal_edges=internal,
                    external_edges=external,
                    cohesion=cohesion,
                )
            )

            # Store module assignment on file nodes (using node IDs)
            for file_node_id in community_node_ids:
                G.nodes[file_node_id]["module"] = module_name

    # ── 4. Propagate module to child nodes ─────────────────────────────────

    for node_id, data in G.nodes(data=True):
        if data.get("kind") in ("class", "function", "type", "config", "route"):
            file_path: str = data.get("file", "")
            file_node_id = f"file::{file_path}"
            if file_node_id in G:
                parent_module: str = G.nodes[file_node_id].get("module", "")
                if parent_module:
                    G.nodes[node_id]["module"] = parent_module

    # ── 5. Cross-module dependencies ───────────────────────────────────────

    cross_module_deps = extract_cross_module_deps(G)

    # ── 6. Assemble result ─────────────────────────────────────────────────

    return GraphAnalysis(
        total_nodes=G.number_of_nodes(),
        total_edges=G.number_of_edges(),
        node_counts=_node_counts(G),
        edge_counts=_edge_counts(G),
        top_functions=top_functions,
        top_files=top_files,
        top_classes=top_classes,
        modules=modules,
        cross_module_deps=cross_module_deps,
        modularity_score=modularity_score,
    )


# ── Cross-module dependency extraction ─────────────────────────────────────


def extract_cross_module_deps(G: nx.DiGraph) -> list[ModuleDependency]:
    """Find all CALLS/IMPORTS edges that cross module boundaries.

    Aggregates by (source_module, target_module) pair, includes top 3
    key connections ranked by callee pagerank, and marks direction as
    ``"mutual"`` if both directions have edges.
    """
    cross_edges: dict[tuple[str, str], list[tuple[str, str, dict[str, str]]]] = (
        defaultdict(list)
    )

    for u, v, data in G.edges(data=True):
        relation: str = data.get("relation", "")
        if relation not in ("CALLS", "IMPORTS"):
            continue
        u_module: str = G.nodes[u].get("module", "")
        v_module: str = G.nodes[v].get("module", "")
        if u_module and v_module and u_module != v_module:
            cross_edges[(u_module, v_module)].append((u, v, data))

    deps: list[ModuleDependency] = []
    for (src, tgt), edges in cross_edges.items():
        # Rank connections by callee pagerank
        sorted_edges = sorted(
            edges,
            key=lambda e: G.nodes[e[1]].get("pagerank", 0.0),
            reverse=True,
        )
        key_conns: list[str] = [
            f"{e[0].split('::')[-1]} \u2192 {e[1].split('::')[-1]}"
            for e in sorted_edges[:3]
        ]
        is_mutual = (tgt, src) in cross_edges
        deps.append(
            ModuleDependency(
                source_module=src,
                target_module=tgt,
                edge_count=len(edges),
                key_connections=key_conns,
                direction="mutual" if is_mutual else "uses",
            )
        )

    return deps
