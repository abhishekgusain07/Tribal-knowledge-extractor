"""Tests for the Writer agent node."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tribal_knowledge.pipeline.agents.writer import _build_fallback_doc, writer_node
from tribal_knowledge.pipeline.pipeline_models import AnalystFindings, PatternFinding
from tribal_knowledge.pipeline.state import PipelineState


# ---------------------------------------------------------------------------
# _build_fallback_doc
# ---------------------------------------------------------------------------


class TestBuildFallbackDoc:
    def test_has_four_sections(self, test_analyst_findings: AnalystFindings) -> None:
        doc = _build_fallback_doc(test_analyst_findings)

        assert "## Quick Commands" in doc
        assert "## Key Files" in doc
        assert "## Non-Obvious Patterns" in doc
        assert "## See Also" in doc

    def test_starts_with_module_heading(
        self, test_analyst_findings: AnalystFindings
    ) -> None:
        doc = _build_fallback_doc(test_analyst_findings)
        assert doc.startswith("# auth\n") or doc.startswith("# Auth\n") or doc.startswith("# auth")

    def test_includes_patterns_and_tribal_knowledge(
        self, test_analyst_findings: AnalystFindings
    ) -> None:
        doc = _build_fallback_doc(test_analyst_findings)

        assert "Repository Pattern" in doc
        assert "bcrypt" in doc

    def test_includes_failure_patterns(
        self, test_analyst_findings: AnalystFindings
    ) -> None:
        doc = _build_fallback_doc(test_analyst_findings)

        assert "bcrypt rounds" in doc.lower() or "token cache" in doc.lower()

    def test_handles_empty_findings(self) -> None:
        findings = AnalystFindings(module="empty")
        doc = _build_fallback_doc(findings)

        assert "# empty" in doc
        assert "## Quick Commands" in doc
        # Should have placeholder text for empty sections
        assert "(no" in doc.lower()

    def test_line_count_reasonable(
        self, test_analyst_findings: AnalystFindings
    ) -> None:
        doc = _build_fallback_doc(test_analyst_findings)
        lines = doc.strip().splitlines()

        # Fallback should produce a reasonable number of lines
        assert len(lines) >= 10
        assert len(lines) <= 50


# ---------------------------------------------------------------------------
# writer_node — without API key (fallback path)
# ---------------------------------------------------------------------------


class TestWriterNodeFallback:
    def test_produces_draft_without_api_key(
        self, base_pipeline_state: PipelineState, test_analyst_findings_data: dict
    ) -> None:
        state = dict(base_pipeline_state)
        state["analyst_output"] = test_analyst_findings_data

        # Patch to force LLM failure
        with patch(
            "langchain_anthropic.ChatAnthropic",
            side_effect=Exception("no key"),
        ):
            result = writer_node(state)

        assert "current_draft" in result
        assert isinstance(result["current_draft"], str)
        assert len(result["current_draft"]) > 0

    def test_resets_iteration_state(
        self, base_pipeline_state: PipelineState, test_analyst_findings_data: dict
    ) -> None:
        state = dict(base_pipeline_state)
        state["analyst_output"] = test_analyst_findings_data
        state["iteration"] = 5  # simulate leftover state
        state["score"] = 0.9

        with patch(
            "langchain_anthropic.ChatAnthropic",
            side_effect=Exception("no key"),
        ):
            result = writer_node(state)

        assert result["iteration"] == 0
        assert result["score"] == 0.0
        assert result["previous_score"] == 0.0

    def test_fallback_has_required_sections(
        self, base_pipeline_state: PipelineState, test_analyst_findings_data: dict
    ) -> None:
        state = dict(base_pipeline_state)
        state["analyst_output"] = test_analyst_findings_data

        with patch(
            "langchain_anthropic.ChatAnthropic",
            side_effect=Exception("no key"),
        ):
            result = writer_node(state)

        draft = result["current_draft"]
        assert "## Quick Commands" in draft
        assert "## Key Files" in draft
        assert "## Non-Obvious Patterns" in draft
        assert "## See Also" in draft


# ---------------------------------------------------------------------------
# writer_node — with mocked LLM
# ---------------------------------------------------------------------------


class TestWriterNodeWithLLM:
    @patch("langchain_anthropic.ChatAnthropic")
    def test_returns_llm_draft(
        self,
        mock_anthropic_cls: MagicMock,
        base_pipeline_state: PipelineState,
        test_analyst_findings_data: dict,
    ) -> None:
        state = dict(base_pipeline_state)
        state["analyst_output"] = test_analyst_findings_data

        mock_response = MagicMock()
        mock_response.content = "# Auth\n\n## Quick Commands\n- Add auth method\n\n## Key Files\n- login.py\n\n## Non-Obvious Patterns\n- bcrypt 12 rounds\n\n## See Also\n- db.md"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_anthropic_cls.return_value = mock_llm

        result = writer_node(state)

        assert "# Auth" in result["current_draft"]
        assert result["iteration"] == 0

    @patch("langchain_anthropic.ChatAnthropic")
    def test_handles_empty_analyst_output(
        self,
        mock_anthropic_cls: MagicMock,
        base_pipeline_state: PipelineState,
    ) -> None:
        state = dict(base_pipeline_state)
        state["analyst_output"] = {"module": "unknown"}

        mock_response = MagicMock()
        mock_response.content = "# unknown\n\n## Quick Commands\n- N/A"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_anthropic_cls.return_value = mock_llm

        result = writer_node(state)

        assert isinstance(result["current_draft"], str)
