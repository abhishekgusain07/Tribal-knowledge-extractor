"""Tests for the Fixer agent node."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tribal_knowledge.pipeline.agents.fixer import fixer_node
from tribal_knowledge.pipeline.pipeline_models import AnalystFindings, CritiqueResult
from tribal_knowledge.pipeline.state import PipelineState

from tests.conftest import SAMPLE_DRAFT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fixer_state(
    base_pipeline_state: PipelineState,
    test_analyst_findings_data: dict,
    draft: str = SAMPLE_DRAFT,
    iteration: int = 1,
    score: float = 0.65,
) -> dict:
    """Build a state dict ready for fixer_node."""
    state = dict(base_pipeline_state)
    state["current_draft"] = draft
    state["analyst_output"] = test_analyst_findings_data
    state["iteration"] = iteration
    state["critique"] = CritiqueResult(
        overall_score=score,
        dimension_scores={
            "accuracy": 0.9,
            "coverage": 0.5,
            "conciseness": 0.8,
            "usefulness": 0.6,
        },
        critique=[
            "COVERAGE: Missing mention of User class in Key Files",
            "USEFULNESS: Add explanation of why bcrypt rounds matter",
        ],
        passed=False,
    ).model_dump()
    return state


# ---------------------------------------------------------------------------
# fixer_node — fallback (LLM error)
# ---------------------------------------------------------------------------


class TestFixerNodeFallback:
    def test_returns_original_on_llm_error(
        self,
        base_pipeline_state: PipelineState,
        test_analyst_findings_data: dict,
    ) -> None:
        state = _make_fixer_state(base_pipeline_state, test_analyst_findings_data)

        with patch(
            "langchain_anthropic.ChatAnthropic",
            side_effect=Exception("no key"),
        ):
            result = fixer_node(state)

        # Should return original draft unchanged
        assert result["current_draft"] == SAMPLE_DRAFT

    def test_increments_iteration(
        self,
        base_pipeline_state: PipelineState,
        test_analyst_findings_data: dict,
    ) -> None:
        state = _make_fixer_state(
            base_pipeline_state, test_analyst_findings_data, iteration=2
        )

        with patch(
            "langchain_anthropic.ChatAnthropic",
            side_effect=Exception("no key"),
        ):
            result = fixer_node(state)

        assert result["iteration"] == 3


# ---------------------------------------------------------------------------
# fixer_node — with mocked LLM
# ---------------------------------------------------------------------------


class TestFixerNodeWithLLM:
    @patch("langchain_anthropic.ChatAnthropic")
    def test_returns_revised_draft(
        self,
        mock_anthropic_cls: MagicMock,
        base_pipeline_state: PipelineState,
        test_analyst_findings_data: dict,
    ) -> None:
        revised = SAMPLE_DRAFT.replace(
            "## Key Files",
            "## Key Files\n- `src/auth/models.py` — User model with repository pattern",
        )

        mock_response = MagicMock()
        mock_response.content = revised
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_anthropic_cls.return_value = mock_llm

        state = _make_fixer_state(base_pipeline_state, test_analyst_findings_data)
        result = fixer_node(state)

        assert "User model" in result["current_draft"]
        assert result["iteration"] == 2  # was 1, now 2

    @patch("langchain_anthropic.ChatAnthropic")
    def test_surgical_edit_not_full_rewrite(
        self,
        mock_anthropic_cls: MagicMock,
        base_pipeline_state: PipelineState,
        test_analyst_findings_data: dict,
    ) -> None:
        """Fixer should change <30% of the original draft."""
        # Small surgical addition
        revised = SAMPLE_DRAFT.replace(
            "## See Also",
            "- User class in models.py abstracts all DB access\n\n## See Also",
        )

        mock_response = MagicMock()
        mock_response.content = revised
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_anthropic_cls.return_value = mock_llm

        state = _make_fixer_state(base_pipeline_state, test_analyst_findings_data)
        result = fixer_node(state)

        original_lines = set(SAMPLE_DRAFT.strip().splitlines())
        revised_lines = set(result["current_draft"].strip().splitlines())

        # Symmetric difference should be small relative to original
        diff_lines = original_lines.symmetric_difference(revised_lines)
        diff_ratio = len(diff_lines) / max(len(original_lines), 1)

        assert diff_ratio < 0.30, (
            f"Fixer changed {diff_ratio:.0%} of lines — expected surgical edit (<30%)"
        )

    @patch("langchain_anthropic.ChatAnthropic")
    def test_handles_empty_critique(
        self,
        mock_anthropic_cls: MagicMock,
        base_pipeline_state: PipelineState,
        test_analyst_findings_data: dict,
    ) -> None:
        state = dict(base_pipeline_state)
        state["current_draft"] = SAMPLE_DRAFT
        state["analyst_output"] = test_analyst_findings_data
        state["iteration"] = 0
        state["critique"] = CritiqueResult(
            overall_score=0.80,
            dimension_scores={},
            critique=[],
            passed=False,
        ).model_dump()

        mock_response = MagicMock()
        mock_response.content = SAMPLE_DRAFT  # No changes needed
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_anthropic_cls.return_value = mock_llm

        result = fixer_node(state)

        assert result["current_draft"] == SAMPLE_DRAFT
        assert result["iteration"] == 1

    @patch("langchain_anthropic.ChatAnthropic")
    def test_handles_empty_state_gracefully(
        self,
        mock_anthropic_cls: MagicMock,
        base_pipeline_state: PipelineState,
    ) -> None:
        """Fixer should not crash with empty/default state values."""
        state = dict(base_pipeline_state)
        state["current_draft"] = ""
        state["analyst_output"] = {"module": "unknown"}
        state["critique"] = CritiqueResult().model_dump()
        state["iteration"] = 0

        mock_response = MagicMock()
        mock_response.content = "# Fixed\n\n## Quick Commands\n- N/A"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_anthropic_cls.return_value = mock_llm

        result = fixer_node(state)

        assert isinstance(result["current_draft"], str)
        assert result["iteration"] == 1
