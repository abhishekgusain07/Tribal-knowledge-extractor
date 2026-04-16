"""Abstract base class for all language parsers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from tribal_knowledge.models import (
    ClassDef,
    Constant,
    DBModel,
    Export,
    FileExtraction,
    FunctionDef,
    Import,
    RouteHandler,
    TribalSignal,
    TypeDef,
)


class BaseParser(ABC):
    """Abstract base for tree-sitter based language parsers.

    Subclasses must implement every ``extract_*`` method.
    The concrete :meth:`parse_file` orchestrates them all and returns a
    :class:`FileExtraction`.
    """

    language: str = ""

    # ── abstract extraction hooks ─────────────────────────────────────

    @abstractmethod
    def extract_functions(self, file_path: str, source: bytes) -> list[FunctionDef]:
        ...

    @abstractmethod
    def extract_classes(self, file_path: str, source: bytes) -> list[ClassDef]:
        ...

    @abstractmethod
    def extract_imports(self, file_path: str, source: bytes) -> list[Import]:
        ...

    @abstractmethod
    def extract_type_defs(self, file_path: str, source: bytes) -> list[TypeDef]:
        ...

    @abstractmethod
    def extract_exports(self, file_path: str, source: bytes) -> list[Export]:
        ...

    @abstractmethod
    def extract_constants(self, file_path: str, source: bytes) -> list[Constant]:
        ...

    @abstractmethod
    def extract_routes(self, file_path: str, source: bytes) -> list[RouteHandler]:
        ...

    @abstractmethod
    def extract_db_models(self, file_path: str, source: bytes) -> list[DBModel]:
        ...

    @abstractmethod
    def extract_env_vars(self, file_path: str, source: bytes) -> list[str]:
        ...

    # ── concrete helpers ──────────────────────────────────────────────

    # Tribal-signal patterns compiled once at class level.
    _STANDARD_MARKERS: dict[str, str] = {
        "HACK": "hack",
        "WORKAROUND": "workaround",
        "XXX": "hack",
        "FIXME": "fixme",
        "TODO": "todo",
        "NOTE": "note",
        "IMPORTANT": "important",
        "WARNING": "warning",
    }

    _META_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bDEPRECATED\b", re.IGNORECASE), "deprecated"),
        (re.compile(r"\bDO\s+NOT\s+REMOVE\b", re.IGNORECASE), "important"),
        (re.compile(r"\bDO\s+NOT\s+DELETE\b", re.IGNORECASE), "important"),
        (re.compile(r"\bDO\s+NOT\s+CHANGE\b", re.IGNORECASE), "important"),
        (re.compile(r"\bbackward\s+compat", re.IGNORECASE), "compatibility"),
        (re.compile(r"\bserialization\b", re.IGNORECASE), "compatibility"),
        (re.compile(r"\border\s+matters\b", re.IGNORECASE), "ordering"),
        (re.compile(r"\bmust\s+be\s+before\b", re.IGNORECASE), "ordering"),
        (re.compile(r"\bmust\s+be\s+after\b", re.IGNORECASE), "ordering"),
    ]

    _RULE_BREAKING_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"eslint-disable"), "rule_override"),
        (re.compile(r"type:\s*ignore"), "rule_override"),
        (re.compile(r"\bnoqa\b"), "rule_override"),
        (re.compile(r"pragma:\s*no\s*cover"), "rule_override"),
    ]

    def extract_tribal_signals(self, source: bytes) -> list[TribalSignal]:
        """Scan source for tribal knowledge signals (comments, markers, overrides)."""
        signals: list[TribalSignal] = []
        lines = source.decode("utf-8", errors="replace").splitlines()

        for line_idx, line in enumerate(lines):
            line_number = line_idx + 1

            # 1. Standard markers (HACK, TODO, FIXME, etc.)
            for marker, category in self._STANDARD_MARKERS.items():
                if re.search(rf"\b{marker}\b", line):
                    signals.append(
                        TribalSignal(
                            line_number=line_number,
                            line_content=line.strip(),
                            category=category,
                        )
                    )
                    break  # one signal per line for standard markers

            # 2. Meta-blog patterns
            for pattern, category in self._META_PATTERNS:
                if pattern.search(line):
                    signals.append(
                        TribalSignal(
                            line_number=line_number,
                            line_content=line.strip(),
                            category=category,
                        )
                    )
                    break  # one signal per meta group per line

            # 3. Rule-breaking overrides
            for pattern, category in self._RULE_BREAKING_PATTERNS:
                if pattern.search(line):
                    signals.append(
                        TribalSignal(
                            line_number=line_number,
                            line_content=line.strip(),
                            category=category,
                        )
                    )
                    break

        return signals

    def detect_module_docstring(self, source: bytes) -> bool:
        """Detect whether the source starts with a module-level docstring.

        Checks for a triple-quoted string at the beginning of the file
        (ignoring leading comments and blank lines).
        """
        text = source.decode("utf-8", errors="replace").lstrip()
        return (
            text.startswith('"""')
            or text.startswith("'''")
            or text.startswith('r"""')
            or text.startswith("r'''")
        )

    # ── concrete orchestrator ─────────────────────────────────────────

    def parse_file(self, file_path: str, source: bytes) -> FileExtraction:
        """Parse *source* and return a complete :class:`FileExtraction`."""
        line_count = source.count(b"\n") + (1 if source and not source.endswith(b"\n") else 0)

        return FileExtraction(
            file_path=file_path,
            language=self.language,
            functions=self.extract_functions(file_path, source),
            classes=self.extract_classes(file_path, source),
            imports=self.extract_imports(file_path, source),
            type_defs=self.extract_type_defs(file_path, source),
            constants=self.extract_constants(file_path, source),
            exports=self.extract_exports(file_path, source),
            route_handlers=self.extract_routes(file_path, source),
            db_models=self.extract_db_models(file_path, source),
            env_vars=self.extract_env_vars(file_path, source),
            line_count=line_count,
            has_module_docstring=self.detect_module_docstring(source),
            tribal_signals=self.extract_tribal_signals(source),
        )


def get_parser_for_language(language: str) -> BaseParser:
    """Return an instantiated parser for the given language name."""
    if language == "python":
        from tribal_knowledge.parser.python_parser import PythonParser

        return PythonParser()
    if language == "typescript":
        from tribal_knowledge.parser.typescript_parser import TypeScriptParser

        return TypeScriptParser()
    if language == "javascript":
        from tribal_knowledge.parser.javascript_parser import JavaScriptParser

        return JavaScriptParser()
    if language == "go":
        from tribal_knowledge.parser.go_parser import GoParser

        return GoParser()
    if language == "rust":
        from tribal_knowledge.parser.rust_parser import RustParser

        return RustParser()
    if language == "java":
        from tribal_knowledge.parser.java_parser import JavaParser

        return JavaParser()
    raise ValueError(f"Unsupported language: {language!r}")
