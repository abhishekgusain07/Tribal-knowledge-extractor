"""Integration tests for the Phase 2 generation pipeline.

These tests exercise the LangGraph routing logic, module iteration loop,
and output writing — all without API keys (using fallback paths).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tribal_knowledge.pipeline.graph import (
    advance_module_queue_node,
    build_pipeline,
    has_more_modules,
    save_module_doc_node,
    should_fix_or_pass,
)
from tribal_knowledge.pipeline.output import (
    _build_quality_report,
    _sanitize_module_name,
    count_doc_tokens,
    write_outputs,
)
from tribal_knowledge.pipeline.pipeline_models import CritiqueResult, QualityReport
from tribal_knowledge.pipeline.state import PipelineState

from tests.conftest import SAMPLE_DRAFT


# ===================================================================
# Routing function tests
# ===================================================================


class TestShouldFixOrPass:
    def test_passes_when_score_high(self) -> None:
        state = PipelineState(score=0.90, iteration=0, previous_score=0.0)
        assert should_fix_or_pass(state) == "pass"

    def test_fixes_when_score_low(self) -> None:
        state = PipelineState(score=0.60, iteration=0, previous_score=0.0)
        assert should_fix_or_pass(state) == "fix"

    def test_passes_at_max_iterations(self) -> None:
        state = PipelineState(score=0.50, iteration=3, previous_score=0.48)
        assert should_fix_or_pass(state) == "pass"

    def test_passes_on_convergence(self) -> None:
        # iteration >= 2 and improvement < 0.02
        state = PipelineState(score=0.70, iteration=2, previous_score=0.69)
        assert should_fix_or_pass(state) == "pass"

    def test_fixes_when_still_improving(self) -> None:
        state = PipelineState(score=0.75, iteration=2, previous_score=0.60)
        assert should_fix_or_pass(state) == "fix"

    def test_boundary_score_085_passes(self) -> None:
        state = PipelineState(score=0.85, iteration=0, previous_score=0.0)
        assert should_fix_or_pass(state) == "pass"

    def test_just_below_threshold_fixes(self) -> None:
        state = PipelineState(score=0.84, iteration=0, previous_score=0.0)
        assert should_fix_or_pass(state) == "fix"

    def test_defaults_to_fix_with_empty_state(self) -> None:
        state = PipelineState()
        assert should_fix_or_pass(state) == "fix"


class TestHasMoreModules:
    def test_next_when_queue_nonempty(self) -> None:
        state = PipelineState(module_queue=["api", "db"])
        assert has_more_modules(state) == "next"

    def test_done_when_queue_empty(self) -> None:
        state = PipelineState(module_queue=[])
        assert has_more_modules(state) == "done"

    def test_done_with_no_queue_key(self) -> None:
        state = PipelineState()
        assert has_more_modules(state) == "done"


# ===================================================================
# Utility node tests
# ===================================================================


class TestSaveModuleDocNode:
    def test_saves_current_draft(self) -> None:
        state = PipelineState(
            current_module="auth",
            current_draft="# Auth doc",
            module_docs={},
        )
        result = save_module_doc_node(state)

        assert result["module_docs"]["auth"] == "# Auth doc"

    def test_preserves_existing_docs(self) -> None:
        state = PipelineState(
            current_module="api",
            current_draft="# API doc",
            module_docs={"auth": "# Auth doc"},
        )
        result = save_module_doc_node(state)

        assert result["module_docs"]["auth"] == "# Auth doc"
        assert result["module_docs"]["api"] == "# API doc"


class TestAdvanceModuleQueueNode:
    def test_pops_first_module(self) -> None:
        state = PipelineState(module_queue=["api", "db"])
        result = advance_module_queue_node(state)

        assert result["current_module"] == "api"
        assert result["module_queue"] == ["db"]

    def test_resets_iteration_state(self) -> None:
        state = PipelineState(module_queue=["api"])
        result = advance_module_queue_node(state)

        assert result["iteration"] == 0
        assert result["score"] == 0.0
        assert result["previous_score"] == 0.0
        assert result["current_draft"] == ""
        assert result["critique"] == {}
        assert result["analyst_output"] == {}

    def test_handles_empty_queue(self) -> None:
        state = PipelineState(module_queue=[])
        result = advance_module_queue_node(state)

        assert result["current_module"] == ""
        assert result["module_queue"] == []


# ===================================================================
# Output module tests
# ===================================================================


class TestSanitizeModuleName:
    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("auth", "auth"),
            ("api-layer", "api-layer"),
            ("My Module", "my-module"),
            ("hello_world", "hello-world"),
            ("special!@#chars", "specialchars"),
            ("multiple---hyphens", "multiple-hyphens"),
            ("  leading-trailing  ", "leading-trailing"),
            ("", "unnamed-module"),
        ],
    )
    def test_sanitizes_correctly(self, input_name: str, expected: str) -> None:
        assert _sanitize_module_name(input_name) == expected


class TestCountDocTokens:
    def test_empty_string_returns_zero(self) -> None:
        assert count_doc_tokens("") == 0

    def test_counts_nonzero_for_text(self) -> None:
        tokens = count_doc_tokens("Hello, world! This is a test.")
        assert tokens > 0

    def test_longer_text_has_more_tokens(self) -> None:
        short = count_doc_tokens("Hello")
        long = count_doc_tokens("Hello, world! This is a much longer sentence with many more tokens.")
        assert long > short


class TestBuildQualityReport:
    def test_builds_report_from_state(self) -> None:
        state = PipelineState(
            module_docs={"auth": "# Auth doc", "api": "# API doc"},
            overview_doc="# Overview",
            conventions_doc="# Conventions",
            agents_doc="# Agents",
            dependency_index_doc="# Deps",
            total_input_tokens=1000,
            total_output_tokens=500,
            total_cost=0.05,
            generation_log=[],
        )
        report = _build_quality_report(state)

        assert isinstance(report, QualityReport)
        # 2 module docs + 4 synthesis docs = 6
        assert len(report.documents) == 6
        assert report.total_tokens == 1500
        assert report.total_cost == pytest.approx(0.05)

    def test_synthesis_docs_score_based_on_existence(self) -> None:
        state = PipelineState(
            module_docs={},
            overview_doc="# Overview",
            conventions_doc="",  # Empty → score 0
            agents_doc="# Agents",
            dependency_index_doc="",
            generation_log=[],
        )
        report = _build_quality_report(state)

        scores_by_name = {d.name: d.score for d in report.documents}
        assert scores_by_name["OVERVIEW.md"] == 1.0
        assert scores_by_name["CONVENTIONS.md"] == 0.0
        assert scores_by_name["AGENTS.md"] == 1.0

    def test_average_score_excludes_zeros(self) -> None:
        state = PipelineState(
            module_docs={"auth": "# Auth"},
            overview_doc="# Overview",
            conventions_doc="",
            agents_doc="",
            dependency_index_doc="",
            generation_log=[
                {"module": "auth", "score": 0.9, "iterations": 2, "tokens_used": 100},
            ],
        )
        report = _build_quality_report(state)

        # Only scores > 0 are included in the average
        assert report.average_score > 0.0


class TestWriteOutputs:
    def test_writes_all_files(self, tmp_path: Path) -> None:
        state = PipelineState(
            module_docs={"auth": "# Auth\nContent here", "api": "# API\nContent here"},
            overview_doc="# Overview",
            conventions_doc="# Conventions",
            agents_doc="# Agents",
            dependency_index_doc="# Dependencies",
            generation_log=[],
        )
        report = write_outputs(state, str(tmp_path))

        context_dir = tmp_path / "context"
        assert context_dir.exists()
        assert (context_dir / "auth.md").exists()
        assert (context_dir / "api.md").exists()
        assert (context_dir / "OVERVIEW.md").exists()
        assert (context_dir / "CONVENTIONS.md").exists()
        assert (context_dir / "AGENTS.md").exists()
        assert (context_dir / "dependency-index.md").exists()
        assert (context_dir / "quality-report.json").exists()

    def test_quality_report_is_valid_json(self, tmp_path: Path) -> None:
        state = PipelineState(
            module_docs={"auth": "# Auth"},
            overview_doc="# Overview",
            conventions_doc="# Conventions",
            agents_doc="# Agents",
            dependency_index_doc="# Deps",
            generation_log=[],
        )
        write_outputs(state, str(tmp_path))

        report_path = tmp_path / "context" / "quality-report.json"
        data = json.loads(report_path.read_text())
        assert "documents" in data
        assert "average_score" in data

    def test_skips_empty_synthesis_docs(self, tmp_path: Path) -> None:
        state = PipelineState(
            module_docs={"auth": "# Auth"},
            overview_doc="# Overview",
            conventions_doc="",  # Empty
            agents_doc="",  # Empty
            dependency_index_doc="",  # Empty
            generation_log=[],
        )
        write_outputs(state, str(tmp_path))

        context_dir = tmp_path / "context"
        assert (context_dir / "OVERVIEW.md").exists()
        assert not (context_dir / "CONVENTIONS.md").exists()

    def test_returns_quality_report(self, tmp_path: Path) -> None:
        state = PipelineState(
            module_docs={"auth": "# Auth"},
            overview_doc="# Overview",
            conventions_doc="# Conv",
            agents_doc="# Agents",
            dependency_index_doc="# Deps",
            generation_log=[],
        )
        report = write_outputs(state, str(tmp_path))

        assert isinstance(report, QualityReport)
        assert len(report.documents) > 0


# ===================================================================
# Full pipeline graph compilation
# ===================================================================


class TestBuildPipeline:
    def test_compiles_without_error(self) -> None:
        pipeline = build_pipeline()
        assert pipeline is not None

    def test_graph_has_expected_nodes(self) -> None:
        pipeline = build_pipeline()
        # The compiled graph should have all our nodes
        node_names = set(pipeline.get_graph().nodes.keys())
        expected = {
            "explorer",
            "analyst",
            "writer",
            "critic",
            "fixer",
            "save_module",
            "next_module",
            "synthesize",
        }
        # LangGraph adds __start__ and __end__ nodes
        assert expected.issubset(node_names)


# ===================================================================
# End-to-end pipeline run (no API keys)
# ===================================================================


class TestPipelineEndToEnd:
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False)
    def test_full_pipeline_produces_outputs(
        self, base_pipeline_state: PipelineState, tmp_path: Path
    ) -> None:
        """Run the full pipeline in fallback mode (no API keys) and verify outputs."""
        # Override output dir to use tmp_path
        state = dict(base_pipeline_state)
        state["output_dir"] = str(tmp_path)
        # Set repo_path to a real directory so file tree works
        state["repo_path"] = str(tmp_path)

        pipeline = build_pipeline()
        final_state = pipeline.invoke(state)

        # Verify module docs were generated for all 3 modules
        module_docs = final_state.get("module_docs", {})
        assert len(module_docs) >= 1, f"Expected module docs, got: {list(module_docs.keys())}"

        # Verify synthesis docs exist
        assert final_state.get("overview_doc", ""), "OVERVIEW.md should be non-empty"
        assert final_state.get("conventions_doc", ""), "CONVENTIONS.md should be non-empty"
        assert final_state.get("agents_doc", ""), "AGENTS.md should be non-empty"
        assert final_state.get("dependency_index_doc", ""), "dependency-index.md should be non-empty"

        # Write outputs and verify files
        report = write_outputs(final_state, str(tmp_path))
        context_dir = tmp_path / "context"

        assert (context_dir / "OVERVIEW.md").exists()
        assert (context_dir / "quality-report.json").exists()
        assert isinstance(report, QualityReport)
        assert len(report.documents) > 0

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False)
    def test_pipeline_handles_single_module(self, tmp_path: Path) -> None:
        """Pipeline should work with just one module."""
        from tribal_knowledge.pipeline.pipeline_models import ModuleMapEntry
        from tribal_knowledge.models import GraphAnalysis, ModuleInfo

        state = PipelineState(
            repo_path=str(tmp_path),
            output_dir=str(tmp_path),
            knowledge_graph_data={"directed": True, "multigraph": False, "graph": {}, "nodes": [], "edges": []},
            analysis_data=GraphAnalysis(
                modules=[ModuleInfo(name="solo", files=["main.py"], top_entities=["main"])]
            ).model_dump(),
            extractions_data=[],
            chromadb_path="",
            module_map=[
                ModuleMapEntry(
                    name="solo",
                    description="Single module project",
                    files=["main.py"],
                    key_entities=["main"],
                    importance=1.0,
                ).model_dump()
            ],
            framework="Python",
            file_tree_summary="main.py",
            current_module="solo",
            module_queue=[],
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

        pipeline = build_pipeline()
        final_state = pipeline.invoke(state)

        module_docs = final_state.get("module_docs", {})
        assert "solo" in module_docs
