"""Create code chunks with context envelopes for embedding."""

from __future__ import annotations

import tiktoken
import networkx as nx

from tribal_knowledge.models import (
    ClassDef,
    CodeChunk,
    FileExtraction,
    FunctionDef,
    GraphAnalysis,
    Import,
)

# ── Token counting ─────────────────────────────────────────────────────────

_ENCODING = tiktoken.get_encoding("cl100k_base")

TOKEN_MIN = 500
TOKEN_MAX = 2000
TOKEN_TRUNCATE = 1500


def _count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(_ENCODING.encode(text))


def _truncate_to_tokens(text: str, max_tokens: int = TOKEN_TRUNCATE) -> str:
    """Truncate text to *max_tokens*, appending a truncation notice."""
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    remaining = len(tokens) - max_tokens
    truncated = _ENCODING.decode(tokens[:max_tokens])
    return truncated + f"\n[... truncated, {remaining} tokens remaining]"


# ── Helpers for context envelope ───────────────────────────────────────────


def _format_imports(imports: list[Import]) -> str:
    """Format import list as a compact string."""
    parts: list[str] = []
    for imp in imports[:10]:  # cap at 10 to keep envelope small
        if imp.names:
            parts.append(f"from {imp.module} import {', '.join(imp.names[:5])}")
        else:
            parts.append(f"import {imp.module}")
    return "; ".join(parts) if parts else "none"


def _get_callers(G: nx.DiGraph, node_id: str, limit: int = 3) -> list[str]:
    """Return up to *limit* callers of a node, sorted by pagerank descending."""
    if node_id not in G:
        return []
    callers: list[tuple[str, float]] = []
    for pred in G.predecessors(node_id):
        edge_data = G.edges[pred, node_id]
        if edge_data.get("relation") == "CALLS":
            pr: float = G.nodes[pred].get("pagerank", 0.0)
            callers.append((pred, pr))
    callers.sort(key=lambda x: x[1], reverse=True)
    return [c[0].split("::")[-1] for c in callers[:limit]]


def _get_callees(G: nx.DiGraph, node_id: str) -> list[str]:
    """Return callees of a node."""
    if node_id not in G:
        return []
    callees: list[str] = []
    for succ in G.successors(node_id):
        edge_data = G.edges[node_id, succ]
        if edge_data.get("relation") == "CALLS":
            callees.append(succ.split("::")[-1])
    return callees[:10]


def _get_pagerank(G: nx.DiGraph, node_id: str) -> float:
    """Return pagerank of a node, or 0.0 if not found."""
    if node_id not in G:
        return 0.0
    score: float = G.nodes[node_id].get("pagerank", 0.0)
    return score


def _get_pagerank_rank(score: float, analysis: GraphAnalysis) -> int:
    """Return the rank (1-based) of a pagerank score among all top functions."""
    # Build a sorted list of all known pagerank scores from the analysis
    all_scores: list[float] = []
    for entry in analysis.top_functions:
        pr = entry.get("pagerank")
        if isinstance(pr, (int, float)):
            all_scores.append(float(pr))
    for entry in analysis.top_files:
        pr = entry.get("pagerank")
        if isinstance(pr, (int, float)):
            all_scores.append(float(pr))
    for entry in analysis.top_classes:
        pr = entry.get("pagerank")
        if isinstance(pr, (int, float)):
            all_scores.append(float(pr))

    all_scores.sort(reverse=True)
    for idx, s in enumerate(all_scores):
        if score >= s:
            return idx + 1
    return len(all_scores) + 1


def _get_module_for_file(file_path: str, analysis: GraphAnalysis) -> str:
    """Find the module name for a file from the analysis modules."""
    for mod in analysis.modules:
        if file_path in mod.files:
            return mod.name
    return "unknown"


def _get_source_lines(file_path: str) -> list[str]:
    """Read source lines from a file, returning empty list on error."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except OSError:
        return []


# ── Chunk builders ─────────────────────────────────────────────────────────


def _build_function_chunk(
    func: FunctionDef,
    extraction: FileExtraction,
    G: nx.DiGraph,
    analysis: GraphAnalysis,
    class_name: str | None = None,
    source_lines: list[str] | None = None,
) -> CodeChunk:
    """Build a function chunk with context envelope."""
    entity_name = f"{class_name}.{func.name}" if class_name else func.name
    chunk_id = f"function::{extraction.file_path}::{entity_name}"

    # Build the node_id for graph lookups (matches builder convention)
    node_id = f"func::{extraction.file_path}::{entity_name}"

    module = _get_module_for_file(extraction.file_path, analysis)
    pagerank = _get_pagerank(G, node_id)
    rank = _get_pagerank_rank(pagerank, analysis)
    callers = _get_callers(G, node_id)
    callees = _get_callees(G, node_id)

    # Extract function body from source lines
    code = ""
    if source_lines and func.line_start > 0 and func.line_end > 0:
        body_lines = source_lines[func.line_start - 1 : func.line_end]
        code = "".join(body_lines)
    elif func.signature:
        code = func.signature

    # Build class context line
    class_line = ""
    if class_name:
        # Find class docstring
        cls_doc = ""
        for cls in extraction.classes:
            if cls.name == class_name and cls.docstring:
                cls_doc = cls.docstring.split("\n")[0][:80]
                break
        class_line = f"Class: {class_name}" + (f" ({cls_doc})" if cls_doc else "") + "\n"

    # Build context envelope
    envelope = (
        f"[{extraction.language}] File: {extraction.file_path}\n"
        f"Module: {module}\n"
        f"Imports: {_format_imports(extraction.imports)}\n"
        f"{class_line}"
        f"Callers: {', '.join(callers) if callers else 'none'}\n"
        f"Calls: {', '.join(callees) if callees else 'none'}\n"
        f"PageRank: {pagerank:.6f} (#{rank})\n"
        f"\n"
        f"{code}"
    )

    # Truncate if over budget
    envelope = _truncate_to_tokens(envelope, TOKEN_TRUNCATE)
    token_count = _count_tokens(envelope)

    return CodeChunk(
        chunk_id=chunk_id,
        entity_type="function",
        entity_name=entity_name,
        file_path=extraction.file_path,
        language=extraction.language,
        module=module,
        pagerank=pagerank,
        content=envelope,
        token_count=token_count,
    )


def _build_class_chunk(
    cls: ClassDef,
    extraction: FileExtraction,
    G: nx.DiGraph,
    analysis: GraphAnalysis,
) -> CodeChunk:
    """Build a class chunk: signature + method signatures + docstring."""
    chunk_id = f"class::{extraction.file_path}::{cls.name}"
    node_id = f"class::{extraction.file_path}::{cls.name}"

    module = _get_module_for_file(extraction.file_path, analysis)
    pagerank = _get_pagerank(G, node_id)
    rank = _get_pagerank_rank(pagerank, analysis)

    # Build class signature
    bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
    decorators_str = "\n".join(f"@{d}" for d in cls.decorators) + "\n" if cls.decorators else ""
    class_sig = f"{decorators_str}class {cls.name}{bases_str}:"

    # Add docstring
    docstring_section = ""
    if cls.docstring:
        short_doc = cls.docstring.split("\n")[0][:200]
        docstring_section = f'    """{short_doc}"""\n'

    # Add method signatures (not bodies)
    method_sigs: list[str] = []
    for method in cls.methods:
        sig = method.signature if method.signature else f"def {method.name}(...)"
        prefix = "    async " if method.is_async else "    "
        method_sigs.append(f"{prefix}{sig}")

    methods_section = "\n".join(method_sigs) if method_sigs else "    pass"

    code = f"{class_sig}\n{docstring_section}{methods_section}"

    envelope = (
        f"[{extraction.language}] File: {extraction.file_path}\n"
        f"Module: {module}\n"
        f"Imports: {_format_imports(extraction.imports)}\n"
        f"PageRank: {pagerank:.6f} (#{rank})\n"
        f"\n"
        f"{code}"
    )

    envelope = _truncate_to_tokens(envelope, TOKEN_TRUNCATE)
    token_count = _count_tokens(envelope)

    return CodeChunk(
        chunk_id=chunk_id,
        entity_type="class",
        entity_name=cls.name,
        file_path=extraction.file_path,
        language=extraction.language,
        module=module,
        pagerank=pagerank,
        content=envelope,
        token_count=token_count,
    )


def _build_file_summary_chunk(
    extraction: FileExtraction,
    G: nx.DiGraph,
    analysis: GraphAnalysis,
    source_lines: list[str] | None = None,
) -> CodeChunk:
    """Build a file summary chunk: first 50 lines + imports + exports + entity names."""
    chunk_id = f"file_summary::{extraction.file_path}::{extraction.file_path}"
    node_id = f"file::{extraction.file_path}"

    module = _get_module_for_file(extraction.file_path, analysis)
    pagerank = _get_pagerank(G, node_id)
    rank = _get_pagerank_rank(pagerank, analysis)

    # First 50 lines
    first_50 = ""
    if source_lines:
        first_50 = "".join(source_lines[:50])

    # Imports summary
    imports_str = _format_imports(extraction.imports)

    # Exports summary
    exports_list = [e.name for e in extraction.exports[:20]]
    exports_str = ", ".join(exports_list) if exports_list else "none"

    # Class and function names
    class_names = [c.name for c in extraction.classes]
    func_names = [f.name for f in extraction.functions]

    classes_str = ", ".join(class_names) if class_names else "none"
    functions_str = ", ".join(func_names[:20]) if func_names else "none"

    envelope = (
        f"[{extraction.language}] File: {extraction.file_path}\n"
        f"Module: {module}\n"
        f"Imports: {imports_str}\n"
        f"Exports: {exports_str}\n"
        f"Classes: {classes_str}\n"
        f"Functions: {functions_str}\n"
        f"PageRank: {pagerank:.6f} (#{rank})\n"
        f"\n"
        f"{first_50}"
    )

    envelope = _truncate_to_tokens(envelope, TOKEN_TRUNCATE)
    token_count = _count_tokens(envelope)

    return CodeChunk(
        chunk_id=chunk_id,
        entity_type="file_summary",
        entity_name=extraction.file_path,
        file_path=extraction.file_path,
        language=extraction.language,
        module=module,
        pagerank=pagerank,
        content=envelope,
        token_count=token_count,
    )


# ── Public API ─────────────────────────────────────────────────────────────


def create_chunks(
    extractions: list[FileExtraction],
    G: nx.DiGraph,
    analysis: GraphAnalysis,
) -> list[CodeChunk]:
    """Create code chunks with context envelopes from all file extractions.

    Three chunk types are produced:
    1. Function chunks: function body + context envelope
    2. Class chunks: class signature + method signatures + docstring
    3. File summary chunks: first 50 lines + imports + exports + names

    Parameters
    ----------
    extractions:
        Parsed file extractions from the AST parsing step.
    G:
        The knowledge graph (NetworkX DiGraph) with pagerank scores.
    analysis:
        The graph analysis output containing modules and rankings.

    Returns
    -------
    list[CodeChunk]
        All chunks, ready for embedding.
    """
    chunks: list[CodeChunk] = []

    for extraction in extractions:
        # Read source lines once per file
        source_lines = _get_source_lines(extraction.file_path)

        # 1. Function chunks (top-level functions)
        for func in extraction.functions:
            chunk = _build_function_chunk(
                func, extraction, G, analysis,
                class_name=None,
                source_lines=source_lines,
            )
            chunks.append(chunk)

        # 2. Method chunks (class methods)
        for cls in extraction.classes:
            # Class chunk
            class_chunk = _build_class_chunk(cls, extraction, G, analysis)
            chunks.append(class_chunk)

            # Individual method chunks
            for method in cls.methods:
                chunk = _build_function_chunk(
                    method, extraction, G, analysis,
                    class_name=cls.name,
                    source_lines=source_lines,
                )
                chunks.append(chunk)

        # 3. File summary chunk
        file_chunk = _build_file_summary_chunk(
            extraction, G, analysis, source_lines=source_lines,
        )
        chunks.append(file_chunk)

    return chunks
