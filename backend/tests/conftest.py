"""Shared test fixtures for Phase 2 pipeline tests.

All fixtures produce data without requiring API keys or external services.
"""

from __future__ import annotations

import pytest
import networkx as nx
from networkx.readwrite import json_graph

from tribal_knowledge.models import (
    FileExtraction,
    FunctionDef,
    ClassDef,
    Import,
    GraphAnalysis,
    ModuleInfo,
    ModuleDependency,
    TribalSignal,
)
from tribal_knowledge.pipeline.pipeline_models import (
    AnalystFindings,
    CritiqueResult,
    ExplorerOutput,
    ModuleMapEntry,
    PatternFinding,
)
from tribal_knowledge.pipeline.state import PipelineState


# ---------------------------------------------------------------------------
# Knowledge graph fixture
# ---------------------------------------------------------------------------


def _build_test_graph() -> nx.DiGraph:
    """Build a small but realistic knowledge graph for testing."""
    G = nx.DiGraph()

    # Module: auth
    G.add_node(
        "auth::login",
        name="login",
        kind="function",
        file="src/auth/login.py",
        module="auth",
        pagerank=0.15,
    )
    G.add_node(
        "auth::verify_token",
        name="verify_token",
        kind="function",
        file="src/auth/token.py",
        module="auth",
        pagerank=0.12,
    )
    G.add_node(
        "auth::User",
        name="User",
        kind="class",
        file="src/auth/models.py",
        module="auth",
        pagerank=0.10,
    )

    # Module: api
    G.add_node(
        "api::handle_request",
        name="handle_request",
        kind="function",
        file="src/api/routes.py",
        module="api",
        pagerank=0.20,
    )
    G.add_node(
        "api::validate_input",
        name="validate_input",
        kind="function",
        file="src/api/validation.py",
        module="api",
        pagerank=0.08,
    )

    # Module: db
    G.add_node(
        "db::get_connection",
        name="get_connection",
        kind="function",
        file="src/db/pool.py",
        module="db",
        pagerank=0.18,
    )
    G.add_node(
        "db::run_query",
        name="run_query",
        kind="function",
        file="src/db/query.py",
        module="db",
        pagerank=0.09,
    )

    # Edges
    G.add_edge("api::handle_request", "auth::verify_token", relation="CALLS")
    G.add_edge("api::handle_request", "api::validate_input", relation="CALLS")
    G.add_edge("auth::login", "db::get_connection", relation="CALLS")
    G.add_edge("auth::login", "auth::User", relation="CALLS")
    G.add_edge("db::run_query", "db::get_connection", relation="CALLS")
    G.add_edge("api::handle_request", "db::run_query", relation="CALLS")
    G.add_edge("auth::login", "auth::verify_token", relation="CALLS")
    G.add_edge("src/api/routes.py", "src/auth/login.py", relation="IMPORTS")

    return G


@pytest.fixture()
def test_graph() -> nx.DiGraph:
    return _build_test_graph()


@pytest.fixture()
def test_graph_data() -> dict:
    """Serialized graph in node_link_data format, as stored in PipelineState."""
    G = _build_test_graph()
    return json_graph.node_link_data(G)


# ---------------------------------------------------------------------------
# Analysis fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_analysis() -> GraphAnalysis:
    return GraphAnalysis(
        total_nodes=7,
        total_edges=8,
        node_counts={"function": 5, "class": 1},
        edge_counts={"CALLS": 7, "IMPORTS": 1},
        top_functions=[
            {"name": "handle_request", "file": "src/api/routes.py", "pagerank": 0.20},
            {"name": "get_connection", "file": "src/db/pool.py", "pagerank": 0.18},
            {"name": "login", "file": "src/auth/login.py", "pagerank": 0.15},
            {"name": "verify_token", "file": "src/auth/token.py", "pagerank": 0.12},
            {"name": "run_query", "file": "src/db/query.py", "pagerank": 0.09},
        ],
        top_files=[],
        top_classes=[
            {"name": "User", "file": "src/auth/models.py", "pagerank": 0.10},
        ],
        modules=[
            ModuleInfo(
                name="auth",
                files=["src/auth/login.py", "src/auth/token.py", "src/auth/models.py"],
                top_entities=["login", "verify_token", "User"],
                internal_edges=2,
                external_edges=3,
                cohesion=0.65,
            ),
            ModuleInfo(
                name="api",
                files=["src/api/routes.py", "src/api/validation.py"],
                top_entities=["handle_request", "validate_input"],
                internal_edges=1,
                external_edges=3,
                cohesion=0.55,
            ),
            ModuleInfo(
                name="db",
                files=["src/db/pool.py", "src/db/query.py"],
                top_entities=["get_connection", "run_query"],
                internal_edges=1,
                external_edges=2,
                cohesion=0.70,
            ),
        ],
        cross_module_deps=[
            ModuleDependency(
                source_module="api",
                target_module="auth",
                edge_count=1,
                key_connections=["handle_request -> verify_token"],
            ),
            ModuleDependency(
                source_module="auth",
                target_module="db",
                edge_count=1,
                key_connections=["login -> get_connection"],
            ),
            ModuleDependency(
                source_module="api",
                target_module="db",
                edge_count=1,
                key_connections=["handle_request -> run_query"],
            ),
        ],
        modularity_score=0.42,
    )


@pytest.fixture()
def test_analysis_data(test_analysis: GraphAnalysis) -> dict:
    return test_analysis.model_dump()


# ---------------------------------------------------------------------------
# Extractions fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_extractions() -> list[FileExtraction]:
    return [
        FileExtraction(
            file_path="src/auth/login.py",
            language="python",
            file_type="code",
            functions=[
                FunctionDef(name="login", qualified_name="auth.login.login", signature="def login(username, password)"),
            ],
            classes=[],
            imports=[Import(module="db.pool", names=["get_connection"])],
            tribal_signals=[
                TribalSignal(line_number=42, line_content="# HACK: bcrypt rounds must stay at 12 — perf regression at 14", category="hack"),
            ],
            line_count=80,
        ),
        FileExtraction(
            file_path="src/api/routes.py",
            language="python",
            file_type="code",
            functions=[
                FunctionDef(name="handle_request", qualified_name="api.routes.handle_request", signature="def handle_request(req)"),
            ],
            classes=[],
            imports=[
                Import(module="auth.login", names=["login"]),
                Import(module="auth.token", names=["verify_token"]),
            ],
            tribal_signals=[
                TribalSignal(line_number=15, line_content="# WARNING: validate_input must run BEFORE auth check", category="ordering"),
            ],
            line_count=120,
        ),
        FileExtraction(
            file_path="src/db/pool.py",
            language="python",
            file_type="code",
            functions=[
                FunctionDef(name="get_connection", qualified_name="db.pool.get_connection", signature="def get_connection()"),
            ],
            classes=[],
            imports=[],
            tribal_signals=[],
            line_count=50,
        ),
    ]


@pytest.fixture()
def test_extractions_data(test_extractions: list[FileExtraction]) -> list[dict]:
    return [e.model_dump() for e in test_extractions]


# ---------------------------------------------------------------------------
# Module map fixture (Explorer output format)
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_module_map() -> list[dict]:
    entries = [
        ModuleMapEntry(
            name="auth",
            description="Authentication and authorization logic",
            files=["src/auth/login.py", "src/auth/token.py", "src/auth/models.py"],
            key_entities=["login", "verify_token", "User"],
            importance=0.8,
        ),
        ModuleMapEntry(
            name="api",
            description="REST API request handling and validation",
            files=["src/api/routes.py", "src/api/validation.py"],
            key_entities=["handle_request", "validate_input"],
            importance=0.9,
        ),
        ModuleMapEntry(
            name="db",
            description="Database connection pooling and query execution",
            files=["src/db/pool.py", "src/db/query.py"],
            key_entities=["get_connection", "run_query"],
            importance=0.7,
        ),
    ]
    return [e.model_dump() for e in entries]


# ---------------------------------------------------------------------------
# Analyst findings fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_analyst_findings() -> AnalystFindings:
    return AnalystFindings(
        module="auth",
        overview="Handles user login, token verification, and session management.",
        data_flow="Request → login() → bcrypt verify → get_connection() → User lookup → JWT token → response",
        patterns=[
            PatternFinding(
                name="Repository Pattern",
                where="src/auth/models.py",
                why="Abstracts DB access behind User model methods",
            ),
        ],
        tribal_knowledge=[
            "bcrypt rounds must stay at 12 — performance regression at 14",
            "verify_token checks expiry BEFORE signature because expired tokens are cheaper to reject",
        ],
        cross_module_dependencies=["db (get_connection)", "api (verify_token called by handle_request)"],
        conventions=["All auth functions raise AuthError, never return None", "Token format is JWT with Ed25519"],
        modification_patterns="To add a new auth method: add handler in login.py, register in AUTH_METHODS dict in config.py",
        failure_patterns=[
            "Changing bcrypt rounds breaks existing password hashes silently",
            "Token cache TTL must match JWT expiry or stale tokens pass verification",
        ],
    )


@pytest.fixture()
def test_analyst_findings_data(test_analyst_findings: AnalystFindings) -> dict:
    return test_analyst_findings.model_dump()


# ---------------------------------------------------------------------------
# Sample draft document fixture
# ---------------------------------------------------------------------------


SAMPLE_DRAFT = """\
# Auth

## Quick Commands
- To add a new auth method: add handler in `login.py`, register in `AUTH_METHODS` dict
- To debug token issues: check `verify_token()` in `src/auth/token.py`

## Key Files
- `src/auth/login.py` — Entry point for all authentication flows
- `src/auth/token.py` — JWT creation and verification with Ed25519
- `src/auth/models.py` — User model with repository pattern for DB access

## Non-Obvious Patterns
- bcrypt rounds must stay at 12 — performance regression at 14 breaks response SLAs
- verify_token checks expiry BEFORE signature because expired tokens are cheaper to reject
- All auth functions raise AuthError, never return None — callers must handle exceptions
- Token cache TTL must match JWT expiry otherwise stale tokens pass verification

## See Also
- [db.md](db.md) — get_connection() is called during every login
- [api.md](api.md) — handle_request() calls verify_token() on every request
"""


@pytest.fixture()
def sample_draft() -> str:
    return SAMPLE_DRAFT


# ---------------------------------------------------------------------------
# Full PipelineState fixture (for integration / multi-agent tests)
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_pipeline_state(
    test_graph_data: dict,
    test_analysis_data: dict,
    test_extractions_data: list[dict],
    test_module_map: list[dict],
) -> PipelineState:
    """A PipelineState populated through the Explorer stage."""
    return PipelineState(
        repo_path="/tmp/test-repo",
        output_dir="/tmp/test-output",
        knowledge_graph_data=test_graph_data,
        analysis_data=test_analysis_data,
        extractions_data=test_extractions_data,
        chromadb_path="",
        module_map=test_module_map,
        framework="Python + FastAPI",
        file_tree_summary="src/\n  auth/\n    login.py\n    token.py\n    models.py\n  api/\n    routes.py\n    validation.py\n  db/\n    pool.py\n    query.py",
        current_module="auth",
        module_queue=["api", "db"],
        analyst_output={},
        current_draft="",
        critique={},
        score=0.0,
        iteration=0,
        previous_score=0.0,
        module_docs={},
        all_analyst_outputs=[],
        overview_doc="",
        conventions_doc="",
        agents_doc="",
        dependency_index_doc="",
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost=0.0,
        generation_log=[],
        errors=[],
    )
