"""Tests for tribal_knowledge.evaluation.scorer — automated scoring functions."""

from __future__ import annotations

import pytest

from tests.conftest import SAMPLE_DRAFT
from tribal_knowledge.evaluation.scorer import (
    conciseness_score,
    entity_coverage_score,
    factual_accuracy_score,
    structural_completeness_score,
)


# ---------------------------------------------------------------------------
# entity_coverage_score
# ---------------------------------------------------------------------------


class TestEntityCoverageScore:
    """Tests for entity_coverage_score."""

    def test_perfect_coverage_all_entities_mentioned(self) -> None:
        """Doc mentions every entity -> score 1.0, no missing."""
        entities = ["login", "verify_token", "User"]
        score, missing = entity_coverage_score(SAMPLE_DRAFT, entities)
        assert score == 1.0
        assert missing == []

    def test_no_coverage_none_mentioned(self) -> None:
        """Doc mentions no entities -> score 0.0, all missing."""
        entities = ["FooBar", "BazQux", "NeverMentioned"]
        score, missing = entity_coverage_score(SAMPLE_DRAFT, entities)
        assert score == 0.0
        assert set(missing) == set(entities)

    def test_partial_coverage_proportional(self) -> None:
        """Doc mentions some entities -> proportional score."""
        # "login" is in SAMPLE_DRAFT, "FooBar" is not
        entities = ["login", "FooBar"]
        score, missing = entity_coverage_score(SAMPLE_DRAFT, entities)
        assert score == pytest.approx(0.5)
        assert missing == ["FooBar"]

    def test_empty_entity_list_returns_perfect(self) -> None:
        """Empty entity list -> score 1.0 by convention."""
        score, missing = entity_coverage_score(SAMPLE_DRAFT, [])
        assert score == 1.0
        assert missing == []

    def test_missing_entities_list_correct(self) -> None:
        """Returned missing list contains exactly the unmentioned entities."""
        entities = ["login", "verify_token", "NonExistent1", "NonExistent2"]
        score, missing = entity_coverage_score(SAMPLE_DRAFT, entities)
        assert score == pytest.approx(0.5)
        assert set(missing) == {"NonExistent1", "NonExistent2"}

    def test_case_insensitive_matching(self) -> None:
        """Entity matching is case-insensitive."""
        entities = ["LOGIN", "VERIFY_TOKEN"]
        score, missing = entity_coverage_score(SAMPLE_DRAFT, entities)
        assert score == 1.0
        assert missing == []

    def test_single_entity_found(self) -> None:
        """Single entity that exists -> score 1.0."""
        score, missing = entity_coverage_score(SAMPLE_DRAFT, ["bcrypt"])
        assert score == 1.0
        assert missing == []

    def test_single_entity_missing(self) -> None:
        """Single entity that doesn't exist -> score 0.0."""
        score, missing = entity_coverage_score(SAMPLE_DRAFT, ["ZZZNotHere"])
        assert score == 0.0
        assert missing == ["ZZZNotHere"]


# ---------------------------------------------------------------------------
# factual_accuracy_score
# ---------------------------------------------------------------------------


class TestFactualAccuracyScore:
    """Tests for factual_accuracy_score."""

    def test_all_refs_valid_high_score(self) -> None:
        """Doc with all valid backtick refs -> score 1.0."""
        doc = "Use `login` from `src/auth/login.py` and `verify_token`."
        entity_names = {"login", "verify_token"}
        file_paths = {"src/auth/login.py"}
        score, invalid = factual_accuracy_score(doc, entity_names, file_paths)
        assert score == 1.0
        assert invalid == []

    def test_all_refs_invalid_low_score(self) -> None:
        """Doc with all invalid backtick refs -> score 0.0."""
        doc = "Use `nonexistent_func` from `src/missing/file.py`."
        entity_names: set[str] = set()
        file_paths: set[str] = set()
        score, invalid = factual_accuracy_score(doc, entity_names, file_paths)
        assert score == 0.0
        assert set(invalid) == {"nonexistent_func", "src/missing/file.py"}

    def test_no_refs_returns_perfect(self) -> None:
        """Doc with no backtick refs -> score 1.0."""
        doc = "This document has no backtick references at all."
        score, invalid = factual_accuracy_score(doc, set(), set())
        assert score == 1.0
        assert invalid == []

    def test_mixed_valid_invalid_proportional(self) -> None:
        """Mixed valid and invalid refs -> proportional score."""
        doc = "See `login` and `ghost_function`."
        entity_names = {"login"}
        file_paths: set[str] = set()
        score, invalid = factual_accuracy_score(doc, entity_names, file_paths)
        assert score == pytest.approx(0.5)
        assert invalid == ["ghost_function"]

    def test_file_path_validation(self) -> None:
        """File path references validated against known file paths."""
        doc = "Check `src/auth/login.py` for details."
        entity_names: set[str] = set()
        file_paths = {"src/auth/login.py"}
        score, invalid = factual_accuracy_score(doc, entity_names, file_paths)
        assert score == 1.0
        assert invalid == []

    def test_invalid_file_path_detected(self) -> None:
        """Invalid file path ref appears in invalid list."""
        doc = "Check `src/missing/nope.py` for details."
        entity_names: set[str] = set()
        file_paths = {"src/auth/login.py"}
        score, invalid = factual_accuracy_score(doc, entity_names, file_paths)
        assert score == 0.0
        assert invalid == ["src/missing/nope.py"]

    def test_sample_draft_accuracy(self) -> None:
        """SAMPLE_DRAFT with known entities and files -> high accuracy."""
        entity_names = {
            "login",
            "verify_token",
            "User",
            "get_connection",
            "handle_request",
            "AUTH_METHODS",
            "login.py",
        }
        file_paths = {
            "src/auth/login.py",
            "src/auth/token.py",
            "src/auth/models.py",
        }
        score, invalid = factual_accuracy_score(SAMPLE_DRAFT, entity_names, file_paths)
        # Most refs should be valid
        assert score > 0.5
        # invalid list should be a list (could have some refs not matched)
        assert isinstance(invalid, list)

    def test_deduplicated_refs(self) -> None:
        """Duplicate refs in doc are deduplicated before scoring."""
        doc = "Use `login` and then `login` again."
        entity_names = {"login"}
        file_paths: set[str] = set()
        score, invalid = factual_accuracy_score(doc, entity_names, file_paths)
        assert score == 1.0
        assert invalid == []


# ---------------------------------------------------------------------------
# conciseness_score
# ---------------------------------------------------------------------------


class TestConcisenessScore:
    """Tests for conciseness_score."""

    def test_short_doc_high_score(self) -> None:
        """Doc under 35 lines -> high score (close to 1.0)."""
        short_doc = "\n".join([f"Line {i}" for i in range(20)])
        score, issues = conciseness_score(short_doc)
        assert score >= 0.9
        assert issues == []

    def test_exactly_at_limit(self) -> None:
        """Doc exactly at max_lines -> score 1.0."""
        doc = "\n".join([f"Line {i}" for i in range(35)])
        score, issues = conciseness_score(doc, max_lines=35)
        assert score >= 0.9
        assert issues == []

    def test_long_doc_penalized(self) -> None:
        """Doc over 35 lines -> penalized score."""
        long_doc = "\n".join([f"Line {i}" for i in range(60)])
        score, issues = conciseness_score(long_doc, max_lines=35)
        assert score < 1.0
        assert len(issues) >= 1

    def test_very_long_doc_heavily_penalized(self) -> None:
        """Doc over 2x max_lines -> heavily penalized."""
        very_long_doc = "\n".join([f"Line {i}" for i in range(120)])
        score, issues = conciseness_score(very_long_doc, max_lines=35)
        assert score < 0.5
        assert len(issues) >= 1

    def test_code_blocks_penalized(self) -> None:
        """Doc with many code blocks -> penalized as encyclopedia pattern."""
        lines: list[str] = []
        for i in range(5):
            lines.append(f"Some text {i}")
            lines.append("```python")
            lines.append(f"def func_{i}(): pass")
            lines.append("```")
        doc = "\n".join(lines)
        score, issues = conciseness_score(doc, max_lines=35)
        # Should detect code_block pattern
        code_block_issues = [i for i in issues if "code_block" in i]
        assert len(code_block_issues) >= 1

    def test_function_signatures_penalized(self) -> None:
        """Doc with many function signatures -> penalized."""
        lines: list[str] = []
        for i in range(5):
            lines.append(f"    def function_{i}(arg1, arg2):")
        doc = "\n".join(lines)
        score, issues = conciseness_score(doc, max_lines=35)
        sig_issues = [i for i in issues if "function_signature" in i]
        assert len(sig_issues) >= 1

    def test_score_clamped_to_zero(self) -> None:
        """Score never goes below 0.0."""
        # Extremely long doc with many patterns
        lines: list[str] = []
        for i in range(200):
            lines.append(f"    def func_{i}(a, b, c):")
            lines.append("```")
        doc = "\n".join(lines)
        score, _issues = conciseness_score(doc, max_lines=10)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# structural_completeness_score
# ---------------------------------------------------------------------------


class TestStructuralCompletenessScore:
    """Tests for structural_completeness_score."""

    def test_all_sections_present(self) -> None:
        """Doc with all 4 required sections -> score 1.0."""
        score = structural_completeness_score(SAMPLE_DRAFT)
        assert score == 1.0

    def test_two_sections_present(self) -> None:
        """Doc with 2 of 4 sections -> score 0.5."""
        doc = "## Quick Commands\nDo this.\n## Key Files\nThese files.\n"
        score = structural_completeness_score(doc)
        assert score == pytest.approx(0.5)

    def test_no_sections_present(self) -> None:
        """Doc with 0 required sections -> score 0.0."""
        doc = "# My Document\nSome random content here.\n"
        score = structural_completeness_score(doc)
        assert score == 0.0

    def test_one_section_present(self) -> None:
        """Doc with 1 of 4 sections -> score 0.25."""
        doc = "## See Also\nLink to other docs.\n"
        score = structural_completeness_score(doc)
        assert score == pytest.approx(0.25)

    def test_three_sections_present(self) -> None:
        """Doc with 3 of 4 sections -> score 0.75."""
        doc = (
            "## Quick Commands\nDo this.\n"
            "## Key Files\nThese files.\n"
            "## Non-Obvious Patterns\nWatch out.\n"
        )
        score = structural_completeness_score(doc)
        assert score == pytest.approx(0.75)

    def test_case_insensitive_section_detection(self) -> None:
        """Section detection is case-insensitive."""
        doc = (
            "## quick commands\nDo this.\n"
            "## key files\nThese files.\n"
            "## non-obvious patterns\nWatch out.\n"
            "## see also\nLinks.\n"
        )
        score = structural_completeness_score(doc)
        assert score == 1.0
