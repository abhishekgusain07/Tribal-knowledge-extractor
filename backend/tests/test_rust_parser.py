"""Tests for tribal_knowledge.parser.rust_parser — Rust language parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from tribal_knowledge.models import FileExtraction
from tribal_knowledge.parser.rust_parser import RustParser


@pytest.fixture()
def rust_source() -> bytes:
    """Read the sample Rust fixture file as bytes."""
    fixture = Path(__file__).parent / "fixtures" / "sample_rust" / "lib.rs"
    return fixture.read_bytes()


@pytest.fixture()
def rust_parser() -> RustParser:
    """Instantiate the Rust parser."""
    return RustParser()


@pytest.fixture()
def rust_extraction(rust_parser: RustParser, rust_source: bytes) -> FileExtraction:
    """Parse the sample Rust file and return the extraction."""
    return rust_parser.parse_file("lib.rs", rust_source)


# ---------------------------------------------------------------------------
# Function extraction
# ---------------------------------------------------------------------------


class TestRustFunctionExtraction:
    """Tests for Rust function/method extraction."""

    def test_extracts_from_env_method(self, rust_extraction: FileExtraction) -> None:
        """Should extract from_env method on Config."""
        func_names = [f.name for f in rust_extraction.functions]
        assert "from_env" in func_names

    def test_extracts_handle_method(self, rust_extraction: FileExtraction) -> None:
        """Should extract handle method (trait impl)."""
        func_names = [f.name for f in rust_extraction.functions]
        assert "handle" in func_names

    def test_extracts_internal_helper(self, rust_extraction: FileExtraction) -> None:
        """Should extract internal_helper function."""
        func_names = [f.name for f in rust_extraction.functions]
        assert "internal_helper" in func_names

    def test_function_count(self, rust_extraction: FileExtraction) -> None:
        """Should extract at least 3 functions/methods."""
        assert len(rust_extraction.functions) >= 3


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------


class TestRustImportExtraction:
    """Tests for Rust import (use) extraction."""

    def test_extracts_hashmap_import(self, rust_extraction: FileExtraction) -> None:
        """Should extract std::collections::HashMap use."""
        import_modules = [i.module for i in rust_extraction.imports]
        hashmap_found = any("collections" in m and "HashMap" in m for m in import_modules)
        # Also check if it's in names
        if not hashmap_found:
            for imp in rust_extraction.imports:
                if "collections" in imp.module and "HashMap" in imp.names:
                    hashmap_found = True
                    break
                if "HashMap" in imp.module:
                    hashmap_found = True
                    break
        assert hashmap_found

    def test_extracts_env_import(self, rust_extraction: FileExtraction) -> None:
        """Should extract std::env use."""
        # The parser splits `use std::env;` into module="std", names=["env"]
        env_found = False
        for imp in rust_extraction.imports:
            if "env" in imp.module:
                env_found = True
                break
            if "env" in imp.names:
                env_found = True
                break
        assert env_found

    def test_import_count(self, rust_extraction: FileExtraction) -> None:
        """Should extract at least 2 imports."""
        assert len(rust_extraction.imports) >= 2


# ---------------------------------------------------------------------------
# Type extraction (struct / trait)
# ---------------------------------------------------------------------------


class TestRustTypeExtraction:
    """Tests for Rust struct and trait extraction."""

    def test_extracts_config_struct(self, rust_extraction: FileExtraction) -> None:
        """Should extract the Config struct."""
        class_names = [c.name for c in rust_extraction.classes]
        type_names = [t.name for t in rust_extraction.type_defs]
        assert "Config" in class_names or "Config" in type_names

    def test_extracts_handler_trait(self, rust_extraction: FileExtraction) -> None:
        """Should extract the Handler trait."""
        class_names = [c.name for c in rust_extraction.classes]
        type_names = [t.name for t in rust_extraction.type_defs]
        assert "Handler" in class_names or "Handler" in type_names

    def test_type_count(self, rust_extraction: FileExtraction) -> None:
        """Should extract at least 2 types (struct + trait)."""
        total = len(rust_extraction.classes) + len(rust_extraction.type_defs)
        assert total >= 2


# ---------------------------------------------------------------------------
# Constant extraction
# ---------------------------------------------------------------------------


class TestRustConstantExtraction:
    """Tests for Rust constant extraction."""

    def test_extracts_max_retries(self, rust_extraction: FileExtraction) -> None:
        """Should extract the MAX_RETRIES constant."""
        const_names = [c.name for c in rust_extraction.constants]
        assert "MAX_RETRIES" in const_names

    def test_constant_count(self, rust_extraction: FileExtraction) -> None:
        """Should extract at least 1 constant."""
        assert len(rust_extraction.constants) >= 1


# ---------------------------------------------------------------------------
# Export (pub) detection
# ---------------------------------------------------------------------------


class TestRustExportExtraction:
    """Tests for Rust pub visibility detection."""

    def test_config_is_exported(self, rust_extraction: FileExtraction) -> None:
        """pub struct Config should be detected as exported."""
        export_names = [e.name for e in rust_extraction.exports]
        assert "Config" in export_names

    def test_handler_is_exported(self, rust_extraction: FileExtraction) -> None:
        """pub trait Handler should be detected as exported."""
        export_names = [e.name for e in rust_extraction.exports]
        assert "Handler" in export_names

    def test_from_env_is_pub_method(self, rust_extraction: FileExtraction) -> None:
        """pub fn from_env should be extracted as a function with pub in signature."""
        from_env_fns = [f for f in rust_extraction.functions if f.name == "from_env"]
        assert len(from_env_fns) == 1
        assert "pub" in from_env_fns[0].signature

    def test_handle_is_pub_method(self, rust_extraction: FileExtraction) -> None:
        """fn handle in trait impl should be extracted as a function."""
        handle_fns = [f for f in rust_extraction.functions if f.name == "handle"]
        assert len(handle_fns) == 1

    def test_internal_helper_not_exported(self, rust_extraction: FileExtraction) -> None:
        """internal_helper (no pub) should NOT be exported."""
        export_names = [e.name for e in rust_extraction.exports]
        assert "internal_helper" not in export_names


# ---------------------------------------------------------------------------
# Env var extraction
# ---------------------------------------------------------------------------


class TestRustEnvVarExtraction:
    """Tests for Rust environment variable extraction."""

    def test_extracts_host_env(self, rust_extraction: FileExtraction) -> None:
        """Should extract HOST from env::var."""
        assert "HOST" in rust_extraction.env_vars

    def test_extracts_port_env(self, rust_extraction: FileExtraction) -> None:
        """Should extract PORT from env::var."""
        assert "PORT" in rust_extraction.env_vars

    def test_env_var_count(self, rust_extraction: FileExtraction) -> None:
        """Should extract at least 2 env vars."""
        assert len(rust_extraction.env_vars) >= 2


# ---------------------------------------------------------------------------
# Tribal signal extraction
# ---------------------------------------------------------------------------


class TestRustTribalSignals:
    """Tests for Rust tribal signal extraction."""

    def test_extracts_todo_signal(self, rust_extraction: FileExtraction) -> None:
        """Should detect TODO comment as tribal signal."""
        categories = [s.category for s in rust_extraction.tribal_signals]
        assert "todo" in categories

    def test_todo_signal_content(self, rust_extraction: FileExtraction) -> None:
        """TODO signal should contain the relevant comment text."""
        todo_signals = [
            s for s in rust_extraction.tribal_signals if s.category == "todo"
        ]
        assert len(todo_signals) >= 1
        assert "error handling" in todo_signals[0].line_content.lower()


# ---------------------------------------------------------------------------
# parse_file integration
# ---------------------------------------------------------------------------


class TestRustParseFile:
    """Integration tests for RustParser.parse_file."""

    def test_returns_file_extraction(self, rust_extraction: FileExtraction) -> None:
        """parse_file returns a FileExtraction instance."""
        assert isinstance(rust_extraction, FileExtraction)

    def test_language_is_rust(self, rust_extraction: FileExtraction) -> None:
        """Language field should be 'rust'."""
        assert rust_extraction.language == "rust"

    def test_file_path_set(self, rust_extraction: FileExtraction) -> None:
        """file_path should be set to the passed value."""
        assert rust_extraction.file_path == "lib.rs"

    def test_line_count_positive(self, rust_extraction: FileExtraction) -> None:
        """line_count should be positive for non-empty source."""
        assert rust_extraction.line_count > 0
