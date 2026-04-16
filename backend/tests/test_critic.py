"""Tests for the Critic agent node."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tribal_knowledge.pipeline.agents.critic import (
    _MAX_LINES,
    _REQUIRED_SECTIONS,
    _programmatic_critique,
    critic_node,
)
from tribal_knowledge.pipeline.pipeline_models import AnalystFindings, CritiqueResult
from tribal_knowledge.pipeline.state import PipelineState

from tests.conftest import SAMPLE_DRAFT


# ---------------------------------------------------------------------------
# _programmatic_critique
# ---------------------------------------------------------------------------


class TestProgrammaticCritique:
    def test_returns_valid_critique_result(
        self, test_analyst_findings: AnalystFindings
    ) -> None:
        result = _programmatic_critique(
            SAMPLE_DRAFT,
            test_analyst_findings,
            top_entities=["login", "verify_token", "User"],
        )

        assert isinstance(result, CritiqueResult)
        assert 0.0 <= result.overall_score <= 1.0
        assert set(result.dimension_scores.keys()) == {
            "accuracy",
            "coverage",
            "conciseness",
            "usefulness",
        }

    def test_all_scores_in_range(
        self, test_analyst_findings: AnalystFindings
    ) -> None:
        result = _programmatic_critique(
            SAMPLE_DRAFT,
            test_analyst_findings,
            top_entities=["login", "verify_token"],
        )

        for dim, score in result.dimension_scores.items():
            assert 0.0 <= score <= 1.0, f"{dim} score {score} out of range"

    def test_good_draft_scores_high(
        self, test_analyst_findings: AnalystFindings
    ) -> None:
        result = _programmatic_critique(
            SAMPLE_DRAFT,
            test_analyst_findings,
            top_entities=["login", "verify_token", "User"],
        )

        # The sample draft is well-structured — should score reasonably well
        assert result.overall_score >= 0.6

    def test_penalizes_missing_sections(
        self, test_analyst_findings: AnalystFindings
    ) -> None:
        bad_draft = "# Auth\n\nJust some text without proper sections."

        result = _programmatic_critique(
            bad_draft,
            test_analyst_findings,
            top_entities=["login"],
        )

        assert len(result.critique) > 0
        assert any("section" in c.lower() for c in result.critique)

    def test_penalizes_over_line_limit(
        self, test_analyst_findings: AnalystFindings
    ) -> None:
        long_draft = "# Auth\n\n## Quick Commands\n- cmd\n\n## Key Files\n- file\n\n## Non-Obvious Patterns\n- pattern\n\n## See Also\n- ref\n"
        # Add many extra lines
        long_draft += "\n".join(f"- extra line {i}" for i in range(40))

        result = _programmatic_critique(
            long_draft,
            test_analyst_findings,
            top_entities=["login"],
        )

        assert result.dimension_scores["conciseness"] < 1.0
        assert any("CONCISENESS" in c for c in result.critique)

    def test_penalizes_missing_entities(
        self, test_analyst_findings: AnalystFindings
    ) -> None:
        sparse_draft = "# Auth\n\n## Quick Commands\n- do stuff\n\n## Key Files\n- file.py\n\n## Non-Obvious Patterns\n- something\n\n## See Also\n- other"

        result = _programmatic_critique(
            sparse_draft,
            test_analyst_findings,
            top_entities=["login", "verify_token", "User", "nonexistent_func"],
        )

        assert any("COVERAGE" in c for c in result.critique)

    def test_empty_draft_scores_low(
        self, test_analyst_findings: AnalystFindings
    ) -> None:
        result = _programmatic_critique(
            "",
            test_analyst_findings,
            top_entities=["login"],
        )

        assert result.overall_score < 0.7

    def test_passed_flag_matches_threshold(
        self, test_analyst_findings: AnalystFindings
    ) -> None:
        result = _programmatic_critique(
            SAMPLE_DRAFT,
            test_analyst_findings,
            top_entities=["login", "verify_token", "User"],
        )

        assert result.passed == (result.overall_score >= 0.85)

    def test_no_top_entities_gives_default_coverage(self) -> None:
        findings = AnalystFindings(module="test")
        result = _programmatic_critique(
            "# Test\n\n## Quick Commands\n- cmd\n\n## Key Files\n- f\n\n## Non-Obvious Patterns\n- p\n\n## See Also\n- s",
            findings,
            top_entities=[],
        )

        assert result.dimension_scores["coverage"] == pytest.approx(0.8)

    def test_why_language_boosts_usefulness(
        self, test_analyst_findings: AnalystFindings
    ) -> None:
        draft_with_why = (
            "# Auth\n\n## Quick Commands\n- cmd\n\n## Key Files\n- f.py\n\n"
            "## Non-Obvious Patterns\n"
            "- This ensures tokens are validated because expired tokens are cheaper\n"
            "- Otherwise the cache becomes stale since TTL mismatches\n\n"
            "## See Also\n- db.md"
        )

        result = _programmatic_critique(
            draft_with_why,
            test_analyst_findings,
            top_entities=[],
        )

        assert result.dimension_scores["usefulness"] >= 0.7


# ---------------------------------------------------------------------------
# critic_node — without API key (programmatic fallback)
# ---------------------------------------------------------------------------


class TestCriticNodeFallback:
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False)
    def test_produces_critique_without_api_key(
        self, base_pipeline_state: PipelineState, test_analyst_findings_data: dict
    ) -> None:
        state = dict(base_pipeline_state)
        state["analyst_output"] = test_analyst_findings_data
        state["current_draft"] = SAMPLE_DRAFT

        result = critic_node(state)

        assert "critique" in result
        assert "score" in result
        assert "previous_score" in result

        critique = CritiqueResult.model_validate(result["critique"])
        assert 0.0 <= critique.overall_score <= 1.0

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False)
    def test_captures_previous_score(
        self, base_pipeline_state: PipelineState, test_analyst_findings_data: dict
    ) -> None:
        state = dict(base_pipeline_state)
        state["analyst_output"] = test_analyst_findings_data
        state["current_draft"] = SAMPLE_DRAFT
        state["score"] = 0.72

        result = critic_node(state)

        assert result["previous_score"] == pytest.approx(0.72)


# ---------------------------------------------------------------------------
# critic_node — with mocked LLM
# ---------------------------------------------------------------------------


class TestCriticNodeWithLLM:
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}, clear=False)
    @patch("langchain_anthropic.ChatAnthropic")
    def test_uses_llm_when_key_present(
        self,
        mock_anthropic_cls: MagicMock,
        base_pipeline_state: PipelineState,
        test_analyst_findings_data: dict,
    ) -> None:
        fake_critique = CritiqueResult(
            overall_score=0.88,
            dimension_scores={
                "accuracy": 0.9,
                "coverage": 0.85,
                "conciseness": 0.9,
                "usefulness": 0.87,
            },
            critique=["COVERAGE: Missing mention of User class"],
            passed=True,
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.invoke.return_value = fake_critique
        mock_anthropic_cls.return_value = mock_llm

        state = dict(base_pipeline_state)
        state["analyst_output"] = test_analyst_findings_data
        state["current_draft"] = SAMPLE_DRAFT

        result = critic_node(state)

        assert result["score"] == pytest.approx(0.88)
        critique = CritiqueResult.model_validate(result["critique"])
        assert critique.passed is True

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}, clear=False)
    @patch("langchain_anthropic.ChatAnthropic")
    def test_falls_back_on_llm_error(
        self,
        mock_anthropic_cls: MagicMock,
        base_pipeline_state: PipelineState,
        test_analyst_findings_data: dict,
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("API error")
        mock_anthropic_cls.return_value = mock_llm

        state = dict(base_pipeline_state)
        state["analyst_output"] = test_analyst_findings_data
        state["current_draft"] = SAMPLE_DRAFT

        result = critic_node(state)

        # Should still produce a valid result via programmatic fallback
        assert "score" in result
        critique = CritiqueResult.model_validate(result["critique"])
        assert 0.0 <= critique.overall_score <= 1.0
