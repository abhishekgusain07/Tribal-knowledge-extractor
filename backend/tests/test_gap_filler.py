"""Tests for tribal_knowledge.evaluation.gap_filler — coverage gap detection."""

from __future__ import annotations

import networkx as nx
import pytest

from tribal_knowledge.evaluation.gap_filler import (
    CoverageGap,
    detect_gaps,
    suggest_module_for_file,
)
from tribal_knowledge.models import (
    FileExtraction,
    GraphAnalysis,
    ModuleInfo,
    TribalSignal,
)


# ---------------------------------------------------------------------------
# detect_gaps
# ---------------------------------------------------------------------------


class TestDetectGaps:
    """Tests for detect_gaps."""

    def test_all_files_covered_no_gaps(
        self,
        test_graph: nx.DiGraph,
        test_analysis: GraphAnalysis,
        test_extractions: list[FileExtraction],
    ) -> None:
        """All files and entities mentioned in docs -> no gaps."""
        # Build module_docs that mention every file and entity
        module_docs: dict[str, str] = {
            "auth": (
                "# Auth\n"
                "src/auth/login.py - login function\n"
                "src/auth/token.py - verify_token function\n"
                "src/auth/models.py - User class\n"
            ),
            "api": (
                "# API\n"
                "src/api/routes.py - handle_request\n"
                "src/api/validation.py - validate_input\n"
            ),
            "db": (
                "# DB\n"
                "src/db/pool.py - get_connection\n"
                "src/db/query.py - run_query\n"
            ),
        }

        gaps = detect_gaps(module_docs, test_graph, test_analysis, test_extractions)

        # No uncovered files or missing entities
        uncovered = [g for g in gaps if g.gap_type == "uncovered_file"]
        missing = [g for g in gaps if g.gap_type == "missing_entity"]
        assert uncovered == []
        assert missing == []

    def test_uncovered_file_detected(
        self,
        test_graph: nx.DiGraph,
        test_analysis: GraphAnalysis,
        test_extractions: list[FileExtraction],
    ) -> None:
        """File in graph but not in any doc -> uncovered_file gap."""
        # Only mention auth files, omit api and db
        module_docs: dict[str, str] = {
            "auth": (
                "# Auth\n"
                "src/auth/login.py - login\n"
                "src/auth/token.py - verify_token\n"
                "src/auth/models.py - User\n"
            ),
        }

        gaps = detect_gaps(module_docs, test_graph, test_analysis, test_extractions)

        uncovered = [g for g in gaps if g.gap_type == "uncovered_file"]
        uncovered_paths = [g.affected_files[0] for g in uncovered]
        assert "src/api/routes.py" in uncovered_paths
        assert "src/db/pool.py" in uncovered_paths

    def test_missing_entity_detected(
        self,
        test_graph: nx.DiGraph,
        test_analysis: GraphAnalysis,
        test_extractions: list[FileExtraction],
    ) -> None:
        """Top-pagerank entity not in any doc -> missing_entity gap."""
        # Mention files but not certain entity names
        module_docs: dict[str, str] = {
            "auth": "# Auth\nsrc/auth/login.py\nsrc/auth/token.py\nsrc/auth/models.py\n",
            "api": "# API\nsrc/api/routes.py\nsrc/api/validation.py\n",
            "db": "# DB\nsrc/db/pool.py\nsrc/db/query.py\n",
        }

        gaps = detect_gaps(module_docs, test_graph, test_analysis, test_extractions)

        missing = [g for g in gaps if g.gap_type == "missing_entity"]
        missing_names = [g.description for g in missing]
        # The entities are not mentioned by name, so some should be flagged
        # Check that at least some missing entities are found
        assert len(missing) > 0

    def test_orphan_config_detected(
        self,
        test_graph: nx.DiGraph,
        test_analysis: GraphAnalysis,
    ) -> None:
        """Config file with tribal signals not referenced -> orphan_config gap."""
        # Module docs that mention nothing about config files
        module_docs: dict[str, str] = {
            "auth": (
                "# Auth\nsrc/auth/login.py - login\n"
                "src/auth/token.py - verify_token\n"
                "src/auth/models.py - User\n"
            ),
            "api": "# API\nsrc/api/routes.py - handle_request\nsrc/api/validation.py - validate_input\n",
            "db": "# DB\nsrc/db/pool.py - get_connection\nsrc/db/query.py - run_query\n",
        }

        # Create a config extraction with tribal signals
        config_extraction = FileExtraction(
            file_path="config/settings.yaml",
            language="yaml",
            file_type="config",
            tribal_signals=[
                TribalSignal(
                    line_number=5,
                    line_content="# HACK: timeout must be exactly 30s for legacy API",
                    category="hack",
                ),
            ],
            line_count=20,
        )

        extractions: list[FileExtraction] = [config_extraction]

        gaps = detect_gaps(module_docs, test_graph, test_analysis, extractions)

        orphan_gaps = [g for g in gaps if g.gap_type == "orphan_config"]
        assert len(orphan_gaps) == 1
        assert "config/settings.yaml" in orphan_gaps[0].affected_files

    def test_config_without_tribal_signals_not_flagged(
        self,
        test_graph: nx.DiGraph,
        test_analysis: GraphAnalysis,
    ) -> None:
        """Config file without tribal signals -> not an orphan_config gap."""
        module_docs: dict[str, str] = {
            "auth": "# Auth\nsrc/auth/login.py\n",
            "api": "# API\nsrc/api/routes.py\n",
            "db": "# DB\nsrc/db/pool.py\n",
        }

        config_extraction = FileExtraction(
            file_path="config/clean.yaml",
            language="yaml",
            file_type="config",
            tribal_signals=[],  # No signals
            line_count=10,
        )

        gaps = detect_gaps(
            module_docs, test_graph, test_analysis, [config_extraction]
        )

        orphan_gaps = [g for g in gaps if g.gap_type == "orphan_config"]
        assert orphan_gaps == []

    def test_gaps_sorted_by_severity_descending(
        self,
        test_graph: nx.DiGraph,
        test_analysis: GraphAnalysis,
        test_extractions: list[FileExtraction],
    ) -> None:
        """Returned gaps are sorted by severity descending."""
        # Empty docs -> many gaps
        module_docs: dict[str, str] = {}

        gaps = detect_gaps(module_docs, test_graph, test_analysis, test_extractions)

        if len(gaps) >= 2:
            severities = [g.severity for g in gaps]
            assert severities == sorted(severities, reverse=True)

    def test_empty_graph_no_file_or_entity_gaps(
        self,
        test_analysis: GraphAnalysis,
        test_extractions: list[FileExtraction],
    ) -> None:
        """Empty knowledge graph -> no uncovered_file or missing_entity gaps."""
        empty_graph: nx.DiGraph = nx.DiGraph()
        module_docs: dict[str, str] = {}

        gaps = detect_gaps(module_docs, empty_graph, test_analysis, test_extractions)

        uncovered = [g for g in gaps if g.gap_type == "uncovered_file"]
        missing = [g for g in gaps if g.gap_type == "missing_entity"]
        assert uncovered == []
        assert missing == []


# ---------------------------------------------------------------------------
# suggest_module_for_file
# ---------------------------------------------------------------------------


class TestSuggestModuleForFile:
    """Tests for suggest_module_for_file."""

    def test_file_in_module_returns_module_name(
        self,
        test_analysis: GraphAnalysis,
    ) -> None:
        """File listed in a module -> returns that module name."""
        module_docs: dict[str, str] = {"auth": "# Auth"}
        result = suggest_module_for_file(
            "src/auth/login.py", module_docs, test_analysis
        )
        assert result == "auth"

    def test_file_not_in_any_module_uses_prefix_heuristic(
        self,
        test_analysis: GraphAnalysis,
    ) -> None:
        """File not listed in any module -> fallback to longest-prefix match."""
        module_docs: dict[str, str] = {"auth": "# Auth"}
        # File shares "src/auth/" prefix with auth module files
        result = suggest_module_for_file(
            "src/auth/middleware.py", module_docs, test_analysis
        )
        assert result == "auth"

    def test_empty_file_path_returns_empty(
        self,
        test_analysis: GraphAnalysis,
    ) -> None:
        """Empty file path -> returns empty string."""
        module_docs: dict[str, str] = {"auth": "# Auth"}
        result = suggest_module_for_file("", module_docs, test_analysis)
        assert result == ""

    def test_completely_unrelated_file(
        self,
        test_analysis: GraphAnalysis,
    ) -> None:
        """File with no common prefix -> returns best available match."""
        module_docs: dict[str, str] = {}
        result = suggest_module_for_file(
            "totally/different/path.py", module_docs, test_analysis
        )
        # Should return some module (or empty if no prefix overlap at all)
        assert isinstance(result, str)

    def test_prefers_direct_membership_over_prefix(
        self,
        test_analysis: GraphAnalysis,
    ) -> None:
        """Direct module membership takes priority over prefix heuristic."""
        module_docs: dict[str, str] = {}
        # src/api/routes.py is in the "api" module
        result = suggest_module_for_file(
            "src/api/routes.py", module_docs, test_analysis
        )
        assert result == "api"


# ---------------------------------------------------------------------------
# CoverageGap model
# ---------------------------------------------------------------------------


class TestCoverageGapModel:
    """Tests for CoverageGap Pydantic model."""

    def test_default_values(self) -> None:
        """CoverageGap initializes with correct defaults."""
        gap = CoverageGap(
            gap_type="uncovered_file",
            description="File not covered",
            severity=0.5,
        )
        assert gap.affected_files == []
        assert gap.suggested_module == ""

    def test_full_initialization(self) -> None:
        """CoverageGap can be fully populated."""
        gap = CoverageGap(
            gap_type="missing_entity",
            description="Entity 'login' not in docs",
            severity=0.8,
            affected_files=["src/auth/login.py"],
            suggested_module="auth",
        )
        assert gap.gap_type == "missing_entity"
        assert gap.severity == 0.8
        assert gap.affected_files == ["src/auth/login.py"]
        assert gap.suggested_module == "auth"
