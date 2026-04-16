"""Tests for tribal_knowledge.parser.java_parser — Java language parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from tribal_knowledge.models import FileExtraction
from tribal_knowledge.parser.java_parser import JavaParser


@pytest.fixture()
def java_source() -> bytes:
    """Read the sample Java fixture file as bytes."""
    fixture = Path(__file__).parent / "fixtures" / "sample_java" / "UserService.java"
    return fixture.read_bytes()


@pytest.fixture()
def java_parser() -> JavaParser:
    """Instantiate the Java parser."""
    return JavaParser()


@pytest.fixture()
def java_extraction(java_parser: JavaParser, java_source: bytes) -> FileExtraction:
    """Parse the sample Java file and return the extraction."""
    return java_parser.parse_file("UserService.java", java_source)


# ---------------------------------------------------------------------------
# Function / method extraction
# ---------------------------------------------------------------------------


class TestJavaFunctionExtraction:
    """Tests for Java method extraction."""

    def test_extracts_get_users(self, java_extraction: FileExtraction) -> None:
        """Should extract the getUsers public method."""
        func_names = [f.name for f in java_extraction.functions]
        assert "getUsers" in func_names

    def test_extracts_validate_user(self, java_extraction: FileExtraction) -> None:
        """Should extract the validateUser private method."""
        func_names = [f.name for f in java_extraction.functions]
        assert "validateUser" in func_names

    def test_function_count(self, java_extraction: FileExtraction) -> None:
        """Should extract at least 2 methods."""
        assert len(java_extraction.functions) >= 2


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------


class TestJavaClassExtraction:
    """Tests for Java class extraction."""

    def test_extracts_user_service(self, java_extraction: FileExtraction) -> None:
        """Should extract the UserService class."""
        class_names = [c.name for c in java_extraction.classes]
        assert "UserService" in class_names

    def test_extracts_user(self, java_extraction: FileExtraction) -> None:
        """Should extract the User class."""
        class_names = [c.name for c in java_extraction.classes]
        assert "User" in class_names

    def test_class_count(self, java_extraction: FileExtraction) -> None:
        """Should extract at least 2 classes."""
        assert len(java_extraction.classes) >= 2


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------


class TestJavaImportExtraction:
    """Tests for Java import extraction."""

    def test_extracts_java_util_list(self, java_extraction: FileExtraction) -> None:
        """Should extract java.util.List import."""
        import_modules = [i.module for i in java_extraction.imports]
        assert any("java.util" in m for m in import_modules)

    def test_extracts_spring_import(self, java_extraction: FileExtraction) -> None:
        """Should extract Spring stereotype.Service import."""
        import_modules = [i.module for i in java_extraction.imports]
        assert any("springframework" in m or "spring" in m.lower() for m in import_modules)

    def test_extracts_persistence_import(self, java_extraction: FileExtraction) -> None:
        """Should extract javax.persistence import."""
        import_modules = [i.module for i in java_extraction.imports]
        assert any("persistence" in m for m in import_modules)

    def test_import_count(self, java_extraction: FileExtraction) -> None:
        """Should extract at least 3 imports."""
        assert len(java_extraction.imports) >= 3


# ---------------------------------------------------------------------------
# Constant extraction
# ---------------------------------------------------------------------------


class TestJavaConstantExtraction:
    """Tests for Java constant extraction."""

    def test_extracts_default_role(self, java_extraction: FileExtraction) -> None:
        """Should extract the DEFAULT_ROLE constant."""
        const_names = [c.name for c in java_extraction.constants]
        assert "DEFAULT_ROLE" in const_names

    def test_constant_count(self, java_extraction: FileExtraction) -> None:
        """Should extract at least 1 constant."""
        assert len(java_extraction.constants) >= 1


# ---------------------------------------------------------------------------
# Env var extraction
# ---------------------------------------------------------------------------


class TestJavaEnvVarExtraction:
    """Tests for Java environment variable extraction."""

    def test_extracts_database_url(self, java_extraction: FileExtraction) -> None:
        """Should extract DATABASE_URL from System.getenv."""
        assert "DATABASE_URL" in java_extraction.env_vars

    def test_env_var_count(self, java_extraction: FileExtraction) -> None:
        """Should extract at least 1 env var."""
        assert len(java_extraction.env_vars) >= 1


# ---------------------------------------------------------------------------
# DB model detection
# ---------------------------------------------------------------------------


class TestJavaDBModelExtraction:
    """Tests for Java DB model / @Entity detection."""

    def test_detects_user_entity(self, java_extraction: FileExtraction) -> None:
        """User class with @Entity annotation should be detected as DB model."""
        db_model_names = [m.name for m in java_extraction.db_models]
        assert "User" in db_model_names

    def test_db_model_count(self, java_extraction: FileExtraction) -> None:
        """Should detect at least 1 DB model."""
        assert len(java_extraction.db_models) >= 1


# ---------------------------------------------------------------------------
# Decorator / annotation extraction
# ---------------------------------------------------------------------------


class TestJavaAnnotationExtraction:
    """Tests for Java annotation detection (mapped to decorators)."""

    def test_detects_service_annotation(self, java_extraction: FileExtraction) -> None:
        """@Service annotation should appear in class decorators."""
        user_service_classes = [
            c for c in java_extraction.classes if c.name == "UserService"
        ]
        assert len(user_service_classes) == 1
        decorators = user_service_classes[0].decorators
        assert any("Service" in d for d in decorators)

    def test_detects_entity_annotation(self, java_extraction: FileExtraction) -> None:
        """@Entity annotation should appear in User class decorators."""
        user_classes = [c for c in java_extraction.classes if c.name == "User"]
        assert len(user_classes) == 1
        decorators = user_classes[0].decorators
        assert any("Entity" in d for d in decorators)


# ---------------------------------------------------------------------------
# Tribal signal extraction
# ---------------------------------------------------------------------------


class TestJavaTribalSignals:
    """Tests for Java tribal signal extraction."""

    def test_extracts_important_signal(self, java_extraction: FileExtraction) -> None:
        """Should detect IMPORTANT comment as tribal signal."""
        categories = [s.category for s in java_extraction.tribal_signals]
        assert "important" in categories

    def test_important_signal_content(self, java_extraction: FileExtraction) -> None:
        """IMPORTANT signal should contain the relevant comment text."""
        important_signals = [
            s for s in java_extraction.tribal_signals if s.category == "important"
        ]
        assert len(important_signals) >= 1
        assert "singleton" in important_signals[0].line_content.lower()


# ---------------------------------------------------------------------------
# parse_file integration
# ---------------------------------------------------------------------------


class TestJavaParseFile:
    """Integration tests for JavaParser.parse_file."""

    def test_returns_file_extraction(self, java_extraction: FileExtraction) -> None:
        """parse_file returns a FileExtraction instance."""
        assert isinstance(java_extraction, FileExtraction)

    def test_language_is_java(self, java_extraction: FileExtraction) -> None:
        """Language field should be 'java'."""
        assert java_extraction.language == "java"

    def test_file_path_set(self, java_extraction: FileExtraction) -> None:
        """file_path should be set to the passed value."""
        assert java_extraction.file_path == "UserService.java"

    def test_line_count_positive(self, java_extraction: FileExtraction) -> None:
        """line_count should be positive for non-empty source."""
        assert java_extraction.line_count > 0
