"""Tests for tribal_knowledge.parser.go_parser — Go language parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from tribal_knowledge.models import FileExtraction
from tribal_knowledge.parser.go_parser import GoParser


@pytest.fixture()
def go_source() -> bytes:
    """Read the sample Go fixture file as bytes."""
    fixture = Path(__file__).parent / "fixtures" / "sample_go" / "main.go"
    return fixture.read_bytes()


@pytest.fixture()
def go_parser() -> GoParser:
    """Instantiate the Go parser."""
    return GoParser()


@pytest.fixture()
def go_extraction(go_parser: GoParser, go_source: bytes) -> FileExtraction:
    """Parse the sample Go file and return the extraction."""
    return go_parser.parse_file("main.go", go_source)


# ---------------------------------------------------------------------------
# Function extraction
# ---------------------------------------------------------------------------


class TestGoFunctionExtraction:
    """Tests for Go function/method extraction."""

    def test_extracts_main_function(self, go_extraction: FileExtraction) -> None:
        """Should extract the main function."""
        func_names = [f.name for f in go_extraction.functions]
        assert "main" in func_names

    def test_extracts_handle_health(self, go_extraction: FileExtraction) -> None:
        """Should extract the handleHealth function."""
        func_names = [f.name for f in go_extraction.functions]
        assert "handleHealth" in func_names

    def test_extracts_method_start(self, go_extraction: FileExtraction) -> None:
        """Should extract the Start method on Server."""
        func_names = [f.name for f in go_extraction.functions]
        assert "Start" in func_names

    def test_function_count(self, go_extraction: FileExtraction) -> None:
        """Should extract at least 3 functions/methods."""
        assert len(go_extraction.functions) >= 3


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------


class TestGoImportExtraction:
    """Tests for Go import extraction."""

    def test_extracts_fmt_import(self, go_extraction: FileExtraction) -> None:
        """Should extract 'fmt' import."""
        import_modules = [i.module for i in go_extraction.imports]
        assert "fmt" in import_modules

    def test_extracts_net_http_import(self, go_extraction: FileExtraction) -> None:
        """Should extract 'net/http' import."""
        import_modules = [i.module for i in go_extraction.imports]
        assert "net/http" in import_modules

    def test_extracts_os_import(self, go_extraction: FileExtraction) -> None:
        """Should extract 'os' import."""
        import_modules = [i.module for i in go_extraction.imports]
        assert "os" in import_modules

    def test_import_count(self, go_extraction: FileExtraction) -> None:
        """Should extract exactly 3 imports."""
        assert len(go_extraction.imports) == 3


# ---------------------------------------------------------------------------
# Type / struct extraction
# ---------------------------------------------------------------------------


class TestGoTypeExtraction:
    """Tests for Go struct/type extraction."""

    def test_extracts_server_struct(self, go_extraction: FileExtraction) -> None:
        """Should extract the Server struct."""
        class_names = [c.name for c in go_extraction.classes]
        assert "Server" in class_names

    def test_struct_count(self, go_extraction: FileExtraction) -> None:
        """Should extract at least 1 struct."""
        assert len(go_extraction.classes) >= 1


# ---------------------------------------------------------------------------
# Constant extraction
# ---------------------------------------------------------------------------


class TestGoConstantExtraction:
    """Tests for Go constant extraction."""

    def test_extracts_default_timeout(self, go_extraction: FileExtraction) -> None:
        """Should extract the DefaultTimeout constant."""
        const_names = [c.name for c in go_extraction.constants]
        assert "DefaultTimeout" in const_names

    def test_constant_count(self, go_extraction: FileExtraction) -> None:
        """Should extract at least 1 constant."""
        assert len(go_extraction.constants) >= 1


# ---------------------------------------------------------------------------
# Env var extraction
# ---------------------------------------------------------------------------


class TestGoEnvVarExtraction:
    """Tests for Go environment variable extraction."""

    def test_extracts_api_key_env(self, go_extraction: FileExtraction) -> None:
        """Should extract API_KEY from os.Getenv."""
        assert "API_KEY" in go_extraction.env_vars

    def test_env_var_count(self, go_extraction: FileExtraction) -> None:
        """Should extract at least 1 env var."""
        assert len(go_extraction.env_vars) >= 1


# ---------------------------------------------------------------------------
# Route handler extraction
# ---------------------------------------------------------------------------


class TestGoRouteExtraction:
    """Tests for Go route handler extraction."""

    def test_extracts_health_route(self, go_extraction: FileExtraction) -> None:
        """Should extract /health route."""
        route_paths = [r.path for r in go_extraction.route_handlers]
        assert "/health" in route_paths

    def test_route_count(self, go_extraction: FileExtraction) -> None:
        """Should extract at least 1 route."""
        assert len(go_extraction.route_handlers) >= 1


# ---------------------------------------------------------------------------
# Tribal signal extraction
# ---------------------------------------------------------------------------


class TestGoTribalSignals:
    """Tests for Go tribal signal extraction."""

    def test_extracts_hack_signal(self, go_extraction: FileExtraction) -> None:
        """Should detect HACK comment as tribal signal."""
        categories = [s.category for s in go_extraction.tribal_signals]
        assert "hack" in categories

    def test_hack_signal_content(self, go_extraction: FileExtraction) -> None:
        """HACK signal should contain the relevant comment text."""
        hack_signals = [
            s for s in go_extraction.tribal_signals if s.category == "hack"
        ]
        assert len(hack_signals) >= 1
        assert "timeout" in hack_signals[0].line_content.lower()


# ---------------------------------------------------------------------------
# Export extraction
# ---------------------------------------------------------------------------


class TestGoExportExtraction:
    """Tests for Go exported items detection."""

    def test_server_is_exported(self, go_extraction: FileExtraction) -> None:
        """Server (capitalized) should be detected as exported."""
        export_names = [e.name for e in go_extraction.exports]
        assert "Server" in export_names

    def test_start_is_exported(self, go_extraction: FileExtraction) -> None:
        """Start (capitalized method) should be detected as exported."""
        export_names = [e.name for e in go_extraction.exports]
        assert "Start" in export_names

    def test_default_timeout_is_exported(self, go_extraction: FileExtraction) -> None:
        """DefaultTimeout (capitalized constant) should be detected as exported."""
        export_names = [e.name for e in go_extraction.exports]
        assert "DefaultTimeout" in export_names

    def test_handle_health_not_exported(self, go_extraction: FileExtraction) -> None:
        """handleHealth (lowercase) should NOT be exported."""
        export_names = [e.name for e in go_extraction.exports]
        assert "handleHealth" not in export_names


# ---------------------------------------------------------------------------
# parse_file integration
# ---------------------------------------------------------------------------


class TestGoParseFile:
    """Integration tests for GoParser.parse_file."""

    def test_returns_file_extraction(self, go_extraction: FileExtraction) -> None:
        """parse_file returns a FileExtraction instance."""
        assert isinstance(go_extraction, FileExtraction)

    def test_language_is_go(self, go_extraction: FileExtraction) -> None:
        """Language field should be 'go'."""
        assert go_extraction.language == "go"

    def test_file_path_set(self, go_extraction: FileExtraction) -> None:
        """file_path should be set to the passed value."""
        assert go_extraction.file_path == "main.go"

    def test_line_count_positive(self, go_extraction: FileExtraction) -> None:
        """line_count should be positive for non-empty source."""
        assert go_extraction.line_count > 0
