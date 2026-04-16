"""Shared Pydantic models used across all Phase 1 modules."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# ── Repo Ingestion ──────────────────────────────────────────────────────────


class RepoSource(BaseModel):
    """Input: where the repo comes from."""

    path_or_url: str
    branch: str | None = None


class RepoInfo(BaseModel):
    """Output: metadata about the ingested repo."""

    local_path: Path
    name: str
    owner: str | None = None
    is_remote: bool = False
    default_branch: str | None = None
    total_files: int = 0
    languages_detected: dict[str, int] = Field(default_factory=dict)
    head_commit: str | None = None
    file_change_frequency: dict[str, int] = Field(default_factory=dict)


# ── File Discovery ──────────────────────────────────────────────────────────

FileType = Literal["code", "test", "config", "doc"]

SUPPORTED_LANGUAGES = {"python", "typescript", "javascript", "go", "rust", "java"}

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}


class DiscoveredFile(BaseModel):
    """A file found in the repo with its classification."""

    path: str  # relative to repo root
    absolute_path: str
    language: str | None = None
    file_type: FileType = "code"
    size_bytes: int = 0
    line_count: int = 0
    content_hash: str = ""
    last_modified: str | None = None


# ── AST Extraction ──────────────────────────────────────────────────────────


class FunctionDef(BaseModel):
    """Extracted function/method definition."""

    name: str
    qualified_name: str = ""
    signature: str = ""
    line_start: int = 0
    line_end: int = 0
    docstring: str | None = None
    decorators: list[str] = Field(default_factory=list)
    is_async: bool = False
    parameters: list[str] = Field(default_factory=list)
    return_type: str | None = None


class ClassDef(BaseModel):
    """Extracted class definition."""

    name: str
    line_start: int = 0
    line_end: int = 0
    bases: list[str] = Field(default_factory=list)
    docstring: str | None = None
    decorators: list[str] = Field(default_factory=list)
    methods: list[FunctionDef] = Field(default_factory=list)


class Import(BaseModel):
    """Extracted import statement."""

    module: str
    names: list[str] = Field(default_factory=list)
    is_relative: bool = False
    alias: str | None = None


class TypeDef(BaseModel):
    """Extracted type definition (interface, type alias, enum, protocol)."""

    name: str
    kind: str = ""  # "interface" | "type_alias" | "enum" | "protocol"
    line_start: int = 0
    line_end: int = 0


class Export(BaseModel):
    """Extracted export."""

    name: str
    kind: str = ""  # "function" | "class" | "type" | "variable" | "default"


class Constant(BaseModel):
    """Extracted constant assignment."""

    name: str
    value: str | None = None
    line: int = 0


class RouteHandler(BaseModel):
    """Extracted HTTP route handler."""

    path: str
    method: str = "*"
    handler_function: str = ""


class DBModel(BaseModel):
    """Extracted database model."""

    name: str
    fields: list[str] = Field(default_factory=list)


class TribalSignal(BaseModel):
    """A line in the code that signals tribal knowledge."""

    line_number: int
    line_content: str
    category: str = ""  # "hack", "workaround", "deprecated", "ordering", "compatibility", etc.


class FileExtraction(BaseModel):
    """Complete extraction result for a single source file."""

    file_path: str
    language: str
    file_type: FileType = "code"

    functions: list[FunctionDef] = Field(default_factory=list)
    classes: list[ClassDef] = Field(default_factory=list)
    imports: list[Import] = Field(default_factory=list)
    type_defs: list[TypeDef] = Field(default_factory=list)
    constants: list[Constant] = Field(default_factory=list)
    exports: list[Export] = Field(default_factory=list)

    route_handlers: list[RouteHandler] = Field(default_factory=list)
    db_models: list[DBModel] = Field(default_factory=list)
    env_vars: list[str] = Field(default_factory=list)

    line_count: int = 0
    has_module_docstring: bool = False
    tribal_signals: list[TribalSignal] = Field(default_factory=list)


# ── Dependency Resolution ───────────────────────────────────────────────────


class SymbolEntry(BaseModel):
    """A symbol in the global symbol table."""

    file_path: str
    line: int = 0
    kind: str = ""  # "function" | "class" | "type" | "variable"
    qualified_name: str = ""


class ResolvedImport(BaseModel):
    """An import statement resolved to its target file."""

    source_file: str
    target_file: str
    imported_names: list[str] = Field(default_factory=list)
    resolution: str = "unresolved"  # "exact" | "fuzzy" | "external" | "unresolved"


class CallExpression(BaseModel):
    """A function call extracted from source code."""

    caller: str
    callee_raw: str
    callee_resolved: str | None = None
    file: str
    line: int = 0


class DependencyGraph(BaseModel):
    """The raw resolved relationships before building the NetworkX graph."""

    resolved_imports: list[ResolvedImport] = Field(default_factory=list)
    call_expressions: list[CallExpression] = Field(default_factory=list)
    symbol_table: dict[str, SymbolEntry] = Field(default_factory=dict)

    total_calls: int = 0
    resolved_calls: int = 0
    resolution_rate: float = 0.0


# ── Graph Analysis ──────────────────────────────────────────────────────────


class ModuleDependency(BaseModel):
    """Cross-module dependency for dependency-index.md generation."""

    source_module: str
    target_module: str
    edge_count: int = 0
    key_connections: list[str] = Field(default_factory=list)
    direction: str = "uses"  # "uses" | "mutual"


class ModuleInfo(BaseModel):
    """A detected logical module (community) in the codebase."""

    name: str
    files: list[str] = Field(default_factory=list)
    top_entities: list[str] = Field(default_factory=list)
    internal_edges: int = 0
    external_edges: int = 0
    cohesion: float = 0.0


class GraphAnalysis(BaseModel):
    """Output of PageRank + community detection analysis."""

    total_nodes: int = 0
    total_edges: int = 0
    node_counts: dict[str, int] = Field(default_factory=dict)
    edge_counts: dict[str, int] = Field(default_factory=dict)

    top_functions: list[dict[str, object]] = Field(default_factory=list)
    top_files: list[dict[str, object]] = Field(default_factory=list)
    top_classes: list[dict[str, object]] = Field(default_factory=list)

    modules: list[ModuleInfo] = Field(default_factory=list)
    cross_module_deps: list[ModuleDependency] = Field(default_factory=list)
    modularity_score: float = 0.0


# ── Chunking & Embedding ───────────────────────────────────────────────────


class CodeChunk(BaseModel):
    """A code chunk with context envelope, ready for embedding."""

    chunk_id: str
    entity_type: str = ""  # "function" | "class" | "file_summary"
    entity_name: str = ""
    file_path: str = ""
    language: str = ""
    module: str | None = None
    pagerank: float = 0.0

    content: str = ""
    token_count: int = 0


# ── Serialization / Manifest ───────────────────────────────────────────────


class ParseManifest(BaseModel):
    """Snapshot of a parse run — self-refresh compares against current state."""

    generated_at: str
    head_commit: str | None = None
    repo_name: str = ""
    total_files_parsed: int = 0
    file_hashes: dict[str, str] = Field(default_factory=dict)
    module_assignments: dict[str, str] = Field(default_factory=dict)
    resolution_rate: float = 0.0
    embedding_model: str = "voyage-code-3"


# ── Utilities ───────────────────────────────────────────────────────────────


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
