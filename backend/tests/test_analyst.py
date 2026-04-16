"""Tests for the Analyst agent node."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tribal_knowledge.pipeline.agents.analyst import (
    _build_user_prompt,
    _fallback_findings,
    analyst_node,
)
from tribal_knowledge.pipeline.pipeline_models import AnalystFindings
from tribal_knowledge.pipeline.state import PipelineState


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_includes_module_name(self) -> None:
        prompt = _build_user_prompt(
            module_name="auth",
            module_description="Authentication",
            files=["src/auth/login.py"],
            key_entities=["login"],
            top_entities=["login", "verify_token"],
            code_chunks="def login(): ...",
            subgraph_summary="login → verify_token [CALLS]",
            cross_module="login → get_connection (in db) [CALLS]",
            tribal_signals="[src/auth/login.py:42] (hack) bcrypt rounds",
        )

        assert "**auth**" in prompt
        assert "Authentication" in prompt
        assert "src/auth/login.py" in prompt
        assert "login → verify_token" in prompt
        assert "bcrypt rounds" in prompt

    def test_handles_empty_inputs(self) -> None:
        prompt = _build_user_prompt(
            module_name="empty",
            module_description="",
            files=[],
            key_entities=[],
            top_entities=[],
            code_chunks="",
            subgraph_summary="",
            cross_module="",
            tribal_signals="",
        )

        assert "**empty**" in prompt
        assert "(none)" in prompt


# ---------------------------------------------------------------------------
# _fallback_findings
# ---------------------------------------------------------------------------


class TestFallbackFindings:
    def test_returns_valid_analyst_findings(self) -> None:
        findings = _fallback_findings(
            module_name="auth",
            module_description="Authentication module",
            top_entities=["login", "verify_token"],
            tribal_signals="  [src/auth/login.py:42] (hack) bcrypt rounds must stay at 12",
            cross_module="login → get_connection (in db) [CALLS]",
        )

        assert isinstance(findings, AnalystFindings)
        assert findings.module == "auth"
        assert findings.overview == "Authentication module"

    def test_populates_patterns_from_top_entities(self) -> None:
        findings = _fallback_findings(
            module_name="auth",
            module_description="Auth",
            top_entities=["login", "verify_token", "User"],
            tribal_signals="(no tribal signals found)",
            cross_module="(no cross-module edges)",
        )

        pattern_names = [p.name for p in findings.patterns]
        assert "login" in pattern_names
        assert "verify_token" in pattern_names

    def test_parses_tribal_signals(self) -> None:
        signals = "  [f1.py:10] (hack) first signal\n  [f2.py:20] (workaround) second signal"
        findings = _fallback_findings(
            module_name="test",
            module_description="Test",
            top_entities=[],
            tribal_signals=signals,
            cross_module="(no cross-module edges)",
        )

        assert len(findings.tribal_knowledge) == 2

    def test_parses_cross_module_deps(self) -> None:
        cross_module = "login → get_connection (in db) [CALLS]\nhandle_request → verify_token (in auth) [CALLS]"
        findings = _fallback_findings(
            module_name="api",
            module_description="API",
            top_entities=[],
            tribal_signals="(no tribal signals found)",
            cross_module=cross_module,
        )

        assert len(findings.cross_module_dependencies) == 2

    def test_handles_no_signals_placeholder(self) -> None:
        findings = _fallback_findings(
            module_name="empty",
            module_description="",
            top_entities=[],
            tribal_signals="(no tribal signals found)",
            cross_module="(no cross-module edges)",
        )

        assert findings.tribal_knowledge == []
        assert findings.cross_module_dependencies == []


# ---------------------------------------------------------------------------
# analyst_node — without API key (fallback path)
# ---------------------------------------------------------------------------


class TestAnalystNodeFallback:
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False)
    def test_produces_findings_without_api_key(
        self, base_pipeline_state: PipelineState
    ) -> None:
        result = analyst_node(base_pipeline_state)

        assert "analyst_output" in result
        findings = AnalystFindings.model_validate(result["analyst_output"])
        assert findings.module == "auth"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False)
    def test_appends_to_all_analyst_outputs(
        self, base_pipeline_state: PipelineState
    ) -> None:
        result = analyst_node(base_pipeline_state)

        assert "all_analyst_outputs" in result
        assert len(result["all_analyst_outputs"]) >= 1

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False)
    def test_accumulates_across_modules(
        self, base_pipeline_state: PipelineState
    ) -> None:
        # Simulate existing analyst outputs
        base_pipeline_state["all_analyst_outputs"] = [
            {"module": "db", "overview": "Database layer"}
        ]

        result = analyst_node(base_pipeline_state)

        assert len(result["all_analyst_outputs"]) == 2


# ---------------------------------------------------------------------------
# analyst_node — with mocked LLM
# ---------------------------------------------------------------------------


class TestAnalystNodeWithLLM:
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}, clear=False)
    @patch("langchain_anthropic.ChatAnthropic")
    def test_calls_claude_with_structured_output(
        self,
        mock_anthropic_cls: MagicMock,
        base_pipeline_state: PipelineState,
    ) -> None:
        fake_findings = AnalystFindings(
            module="auth",
            overview="Auth module handles login and token verification.",
            data_flow="Request → login → bcrypt → DB → JWT",
            tribal_knowledge=["bcrypt rounds must be 12"],
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.invoke.return_value = fake_findings
        mock_anthropic_cls.return_value = mock_llm

        result = analyst_node(base_pipeline_state)

        findings = AnalystFindings.model_validate(result["analyst_output"])
        assert findings.module == "auth"
        assert "bcrypt rounds must be 12" in findings.tribal_knowledge

        # Verify model was called
        mock_anthropic_cls.assert_called_once()
        mock_llm.with_structured_output.assert_called_once_with(AnalystFindings)
