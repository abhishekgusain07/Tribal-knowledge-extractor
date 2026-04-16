"""JavaScript / JSX parser — thin wrapper around the TypeScript parser."""

from __future__ import annotations

from tribal_knowledge.models import TypeDef
from tribal_knowledge.parser.typescript_parser import TypeScriptParser


class JavaScriptParser(TypeScriptParser):
    """Extract structural information from JavaScript / JSX source files.

    JavaScript is a subset of TypeScript for our extraction purposes, so
    this class inherits nearly everything. The only differences are:

    1. The tree-sitter language is ``"javascript"`` instead of ``"typescript"``.
    2. :meth:`extract_type_defs` always returns an empty list because
       plain JS has no ``interface``, ``type``, or ``enum`` declarations.
    """

    language: str = "javascript"

    def __init__(self) -> None:
        super().__init__(lang_override="javascript")

    # ── Overrides ─────────────────────────────────────────────────────

    def extract_type_defs(self, file_path: str, source: bytes) -> list[TypeDef]:
        """JavaScript has no type definitions — always returns ``[]``."""
        return []
