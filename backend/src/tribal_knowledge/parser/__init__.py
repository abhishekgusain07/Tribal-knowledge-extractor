"""Multi-language Tree-sitter AST parsers."""

from tribal_knowledge.parser.base import BaseParser, get_parser_for_language
from tribal_knowledge.parser.go_parser import GoParser
from tribal_knowledge.parser.java_parser import JavaParser
from tribal_knowledge.parser.python_parser import PythonParser
from tribal_knowledge.parser.rust_parser import RustParser

__all__ = [
    "BaseParser",
    "GoParser",
    "JavaParser",
    "PythonParser",
    "RustParser",
    "get_parser_for_language",
]
