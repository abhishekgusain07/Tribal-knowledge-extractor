"""Tests for the Explorer agent node."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tribal_knowledge.models import GraphAnalysis, ModuleInfo
from tribal_knowledge.pipeline.agents.explorer import (
    _fallback_module_map,
    _full_codebase_fallback,
    explorer_node,
)
from tribal_knowledge.pipeline.pipeline_models import ExplorerOutput, ModuleMapEntry
from tribal_knowledge.pipeline.state import PipelineState


# ---------------------------------------------------------------------------
# _fallback_module_map
# ---------------------------------------------------------------------------


class TestFallbackModuleMap:
    def test_converts_phase1_modules(self, test_analysis: GraphAnalysis) -> None:
        entries = _fallback_module_map(test_analysis)

        assert len(entries) == 3
        names = {e.name for e in entries}
        assert names == {"auth", "api", "db"}

    def test_preserves_files_and_entities(self, test_analysis: GraphAnalysis) -> None:
        entries = _fallback_module_map(test_analysis)
        auth = next(e for e in entries if e.name == "auth")

        assert "src/auth/login.py" in auth.files
        assert "login" in auth.key_entities

    def test_uses_cohesion_as_importance(self, test_analysis: GraphAnalysis) -> None:
        entries = _fallback_module_map(test_analysis)
        auth = next(e for e in entries if e.name == "auth")
        assert auth.importance == pytest.approx(0.65)

    def test_empty_modules(self) -> None:
        analysis = GraphAnalysis(modules=[])
        entries = _fallback_module_map(analysis)
        assert entries == []


# ---------------------------------------------------------------------------
# _full_codebase_fallback
# ---------------------------------------------------------------------------


class TestFullCodebaseFallback:
    def test_returns_single_module(self, test_analysis: GraphAnalysis) -> None:
        entries = _full_codebase_fallback(test_analysis)

        assert len(entries) == 1
        assert entries[0].name == "full-codebase"

    def test_collects_all_files(self, test_analysis: GraphAnalysis) -> None:
        entries = _full_codebase_fallback(test_analysis)
        files = entries[0].files

        assert "src/auth/login.py" in files
        assert "src/api/routes.py" in files
        assert "src/db/pool.py" in files

    def test_deduplicates_files(self) -> None:
        analysis = GraphAnalysis(
            modules=[
                ModuleInfo(name="a", files=["f1.py", "f2.py"]),
                ModuleInfo(name="b", files=["f2.py", "f3.py"]),
            ]
        )
        entries = _full_codebase_fallback(analysis)
        assert len(entries[0].files) == 3

    def test_includes_top_entities(self, test_analysis: GraphAnalysis) -> None:
        entries = _full_codebase_fallback(test_analysis)
        assert "handle_request" in entries[0].key_entities


# ---------------------------------------------------------------------------
# explorer_node — with mocked Gemini
# ---------------------------------------------------------------------------


class TestExplorerNodeWithLLM:
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_returns_llm_modules_on_success(
        self,
        mock_gemini_cls: MagicMock,
        base_pipeline_state: PipelineState,
    ) -> None:
        fake_output = ExplorerOutput(
            framework="FastAPI + SQLAlchemy",
            modules=[
                ModuleMapEntry(
                    name="auth",
                    description="Auth module",
                    files=["src/auth/login.py"],
                    key_entities=["login"],
                    importance=0.9,
                ),
                ModuleMapEntry(
                    name="api",
                    description="API layer",
                    files=["src/api/routes.py"],
                    key_entities=["handle_request"],
                    importance=0.8,
                ),
            ],
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.invoke.return_value = fake_output
        mock_gemini_cls.return_value = mock_llm

        result = explorer_node(base_pipeline_state)

        assert result["framework"] == "FastAPI + SQLAlchemy"
        assert len(result["module_map"]) == 2
        assert result["module_queue"] == ["auth", "api"]
        assert result["current_module"] == "auth"

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_falls_back_on_llm_error(
        self,
        mock_gemini_cls: MagicMock,
        base_pipeline_state: PipelineState,
    ) -> None:
        mock_gemini_cls.side_effect = Exception("API unavailable")

        result = explorer_node(base_pipeline_state)

        # Should use Phase-1 fallback modules
        assert len(result["module_map"]) == 3
        names = [m["name"] for m in result["module_map"]]
        assert "auth" in names
        assert result["framework"] == ""

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_falls_back_when_llm_returns_empty(
        self,
        mock_gemini_cls: MagicMock,
        base_pipeline_state: PipelineState,
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.invoke.return_value = ExplorerOutput(framework="", modules=[])
        mock_gemini_cls.return_value = mock_llm

        result = explorer_node(base_pipeline_state)

        # Falls back to Phase-1 modules
        assert len(result["module_map"]) >= 1

    def test_includes_file_tree_summary(
        self, base_pipeline_state: PipelineState
    ) -> None:
        # Patch to avoid real Gemini call
        with patch(
            "langchain_google_genai.ChatGoogleGenerativeAI",
            side_effect=Exception("no key"),
        ):
            result = explorer_node(base_pipeline_state)

        assert "file_tree_summary" in result
        assert isinstance(result["file_tree_summary"], str)
