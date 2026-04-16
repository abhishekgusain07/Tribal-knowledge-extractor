"""Rust-specific AST parser using tree-sitter."""

from __future__ import annotations

import re

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from tribal_knowledge.models import (
    ClassDef,
    Constant,
    DBModel,
    Export,
    FunctionDef,
    Import,
    RouteHandler,
    TypeDef,
)
from tribal_knowledge.parser.base import BaseParser


# ── Helpers ────────────────────────────────────────────────────────────────


def _node_text(node: Node) -> str:
    """Decode a tree-sitter node's text to a Python string."""
    return node.text.decode("utf-8") if node.text is not None else ""


def _child_of_type(node: Node, type_name: str) -> Node | None:
    """Return the first child whose ``type`` matches *type_name*."""
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _children_of_type(node: Node, type_name: str) -> list[Node]:
    """Return all children whose ``type`` matches *type_name*."""
    return [c for c in node.children if c.type == type_name]


def _has_visibility_pub(node: Node) -> bool:
    """Check if a node has ``pub`` visibility modifier."""
    vis = _child_of_type(node, "visibility_modifier")
    if vis is not None:
        text = _node_text(vis)
        return text.startswith("pub")
    return False


def _is_async_fn(node: Node) -> bool:
    """Check whether a function_item node has the ``async`` keyword."""
    for child in node.children:
        if not child.is_named and _node_text(child) == "async":
            return True
        if child.type == "fn":
            break
    return False


def _collect_doc_comment_above(node: Node) -> str | None:
    """Collect ``///`` doc-comment lines above a node."""
    lines: list[str] = []
    prev = node.prev_named_sibling
    while prev is not None and prev.type == "line_comment":
        text = _node_text(prev)
        if text.startswith("///"):
            lines.insert(0, text[3:].strip())
        elif text.startswith("//!"):
            lines.insert(0, text[3:].strip())
        else:
            break
        prev = prev.prev_named_sibling
    if lines:
        return "\n".join(lines)
    return None


def _extract_fn_name(node: Node) -> str:
    """Extract the function name from a function_item node."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(name_node)
    ident = _child_of_type(node, "identifier")
    if ident is not None:
        return _node_text(ident)
    return ""


def _extract_parameters(node: Node) -> list[str]:
    """Extract parameter names from a function_item's parameters node."""
    params: list[str] = []
    param_node = _child_of_type(node, "parameters")
    if param_node is None:
        return params
    for child in param_node.children:
        if child.type == "parameter":
            pattern = child.child_by_field_name("pattern")
            if pattern is not None:
                params.append(_node_text(pattern))
            else:
                ident = _child_of_type(child, "identifier")
                if ident is not None:
                    params.append(_node_text(ident))
        elif child.type == "self_parameter":
            params.append(_node_text(child))
    return params


def _extract_return_type(node: Node) -> str | None:
    """Extract the return type from a function_item node."""
    # Look for -> type pattern among children
    found_arrow = False
    for child in node.children:
        if not child.is_named and _node_text(child) == "->":
            found_arrow = True
            continue
        if found_arrow and child.type == "block":
            break
        if found_arrow and child.is_named:
            return _node_text(child)
    return None


def _build_fn_signature(
    name: str,
    params: list[str],
    return_type: str | None,
    is_pub: bool,
    is_async: bool,
) -> str:
    """Build a concise Rust function signature string."""
    parts: list[str] = []
    if is_pub:
        parts.append("pub ")
    if is_async:
        parts.append("async ")
    parts.append("fn ")
    parts.append(f"{name}(")
    parts.append(", ".join(params))
    parts.append(")")
    if return_type:
        parts.append(f" -> {return_type}")
    return "".join(parts)


def _extract_struct_name(node: Node) -> str:
    """Extract the name from a struct_item node."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(name_node)
    ident = _child_of_type(node, "type_identifier")
    if ident is not None:
        return _node_text(ident)
    return ""


def _extract_struct_fields(node: Node) -> list[str]:
    """Extract field names from a struct_item's field_declaration_list."""
    fields: list[str] = []
    field_list = _child_of_type(node, "field_declaration_list")
    if field_list is None:
        return fields
    for child in field_list.children:
        if child.type == "field_declaration":
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                fields.append(_node_text(name_node))
            else:
                ident = _child_of_type(child, "field_identifier")
                if ident is not None:
                    fields.append(_node_text(ident))
    return fields


def _extract_impl_type_name(impl_node: Node) -> str:
    """Extract the type name from an impl_item node."""
    type_node = _child_of_type(impl_node, "type_identifier")
    if type_node is not None:
        return _node_text(type_node)
    # Could be a generic type
    generic = _child_of_type(impl_node, "generic_type")
    if generic is not None:
        inner = _child_of_type(generic, "type_identifier")
        if inner is not None:
            return _node_text(inner)
    return ""


def _get_attribute_macros(node: Node) -> list[str]:
    """Extract attribute macro strings from siblings above the node."""
    attrs: list[str] = []
    prev = node.prev_named_sibling
    while prev is not None and prev.type == "attribute_item":
        attrs.insert(0, _node_text(prev))
        prev = prev.prev_named_sibling
    return attrs


def _extract_derive_names(node: Node) -> list[str]:
    """Extract derive trait names from attribute_item nodes above a struct/enum."""
    names: list[str] = []
    prev = node.prev_named_sibling
    while prev is not None and prev.type == "attribute_item":
        text = _node_text(prev)
        derive_match = re.search(r"derive\s*\(([^)]+)\)", text)
        if derive_match:
            inner = derive_match.group(1)
            for part in inner.split(","):
                stripped = part.strip()
                if stripped:
                    names.append(stripped)
        prev = prev.prev_named_sibling
    return names


# ── Route detection patterns ──────────────────────────────────────────────

_ROUTE_ATTR_RE = re.compile(
    r'#\[(get|post|put|delete|patch|head|options)\s*\(\s*"([^"]+)"',
    re.IGNORECASE,
)

# ── Env var patterns ──────────────────────────────────────────────────────

_ENV_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'(?:std::)?env::var\s*\(\s*"([^"]+)"\s*\)'),
    re.compile(r'(?:std::)?env::var_os\s*\(\s*"([^"]+)"\s*\)'),
]

# ── DB model derive traits ───────────────────────────────────────────────

_DB_DERIVE_TRAITS: frozenset[str] = frozenset({
    "sqlx::FromRow",
    "FromRow",
    "diesel::Queryable",
    "Queryable",
    "diesel::Insertable",
    "Insertable",
    "Selectable",
    "diesel::Selectable",
    "AsChangeset",
    "diesel::AsChangeset",
})


# ── Main parser class ────────────────────────────────────────────────────


class RustParser(BaseParser):
    """Tree-sitter based parser for Rust source files."""

    language: str = "rust"

    def __init__(self) -> None:
        self._parser = get_parser("rust")

    # ── helpers ────────────────────────────────────────────────────────

    def _parse_tree(self, source: bytes) -> Node:
        """Parse source bytes and return the root node."""
        tree = self._parser.parse(source)
        return tree.root_node

    def detect_module_docstring(self, source: bytes) -> bool:
        """Detect whether the Rust file starts with a ``//!`` module doc-comment."""
        text = source.decode("utf-8", errors="replace").lstrip()
        return text.startswith("//!")

    # ── extraction methods ─────────────────────────────────────────────

    def extract_functions(self, file_path: str, source: bytes) -> list[FunctionDef]:
        """Extract top-level functions and methods inside impl blocks."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        functions: list[FunctionDef] = []

        for child in root.children:
            if child.type == "function_item":
                func = self._extract_function_item(child, class_name=None)
                if func is not None:
                    functions.append(func)

            elif child.type == "impl_item":
                type_name = _extract_impl_type_name(child)
                decl_list = _child_of_type(child, "declaration_list")
                if decl_list is not None:
                    for item in decl_list.children:
                        if item.type == "function_item":
                            func = self._extract_function_item(
                                item, class_name=type_name or None
                            )
                            if func is not None:
                                functions.append(func)

        return functions

    def _extract_function_item(
        self, node: Node, class_name: str | None = None
    ) -> FunctionDef | None:
        """Build a FunctionDef from a function_item node."""
        name = _extract_fn_name(node)
        if not name:
            return None

        is_pub = _has_visibility_pub(node)
        is_async = _is_async_fn(node)
        params = _extract_parameters(node)
        return_type = _extract_return_type(node)
        docstring = _collect_doc_comment_above(node)

        qualified_name = f"{class_name}.{name}" if class_name else name
        signature = _build_fn_signature(name, params, return_type, is_pub, is_async)

        # Collect attribute decorators
        decorators = _get_attribute_macros(node)

        return FunctionDef(
            name=name,
            qualified_name=qualified_name,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            docstring=docstring,
            decorators=decorators,
            is_async=is_async,
            parameters=params,
            return_type=return_type,
        )

    def extract_classes(self, file_path: str, source: bytes) -> list[ClassDef]:
        """Extract struct definitions as ClassDef.

        Methods from corresponding impl blocks are associated with their struct.
        """
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        structs: dict[str, ClassDef] = {}

        for child in root.children:
            if child.type == "struct_item":
                name = _extract_struct_name(child)
                if not name:
                    continue
                docstring = _collect_doc_comment_above(child)
                fields = _extract_struct_fields(child)

                # Derive macros as decorators
                derives = _extract_derive_names(child)
                decorators = [f"derive({d})" for d in derives] if derives else []
                attr_macros = _get_attribute_macros(child)
                # Filter out derive attributes from attr_macros to avoid duplication
                for attr in attr_macros:
                    if "derive" not in attr:
                        decorators.append(attr)

                structs[name] = ClassDef(
                    name=name,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    bases=[],
                    docstring=docstring,
                    decorators=decorators,
                    methods=[],
                )

        # Associate methods from impl blocks
        for child in root.children:
            if child.type != "impl_item":
                continue
            # Skip trait impls (impl Trait for Type) — only direct impls
            type_name = _extract_impl_type_name(child)
            if type_name not in structs:
                continue

            decl_list = _child_of_type(child, "declaration_list")
            if decl_list is None:
                continue

            for item in decl_list.children:
                if item.type == "function_item":
                    func = self._extract_function_item(item, class_name=type_name)
                    if func is not None:
                        structs[type_name].methods.append(func)

        return list(structs.values())

    def extract_imports(self, file_path: str, source: bytes) -> list[Import]:
        """Extract use declarations."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        imports: list[Import] = []

        for child in root.children:
            if child.type != "use_declaration":
                continue
            text = _node_text(child)
            # Remove leading "use " and trailing ";"
            text = text.strip()
            if text.startswith("use "):
                text = text[4:]
            if text.endswith(";"):
                text = text[:-1]
            text = text.strip()

            is_relative = text.startswith("crate::") or text.startswith("self::") or text.startswith("super::")

            # Handle nested imports: use std::{io, fs};
            brace_match = re.match(r"(.+)::\{(.+)\}", text)
            if brace_match:
                base = brace_match.group(1)
                inner = brace_match.group(2)
                names: list[str] = []
                for part in inner.split(","):
                    stripped = part.strip()
                    if stripped:
                        names.append(stripped)
                imports.append(
                    Import(
                        module=base,
                        names=names,
                        is_relative=is_relative,
                    )
                )
                continue

            # Handle glob imports: use std::collections::*;
            if text.endswith("::*"):
                base = text[:-3]
                imports.append(
                    Import(
                        module=base,
                        names=["*"],
                        is_relative=is_relative,
                    )
                )
                continue

            # Handle aliased imports: use std::io::Result as IoResult;
            alias_match = re.match(r"(.+)\s+as\s+(\w+)", text)
            if alias_match:
                full_path = alias_match.group(1).strip()
                alias = alias_match.group(2).strip()
                # Split into module and imported name
                if "::" in full_path:
                    module, name = full_path.rsplit("::", 1)
                    imports.append(
                        Import(
                            module=module,
                            names=[name],
                            is_relative=is_relative,
                            alias=alias,
                        )
                    )
                else:
                    imports.append(
                        Import(
                            module=full_path,
                            names=[],
                            is_relative=is_relative,
                            alias=alias,
                        )
                    )
                continue

            # Simple import: use std::collections::HashMap;
            if "::" in text:
                module, name = text.rsplit("::", 1)
                imports.append(
                    Import(
                        module=module,
                        names=[name],
                        is_relative=is_relative,
                    )
                )
            else:
                imports.append(
                    Import(
                        module=text,
                        names=[],
                        is_relative=is_relative,
                    )
                )

        return imports

    def extract_type_defs(self, file_path: str, source: bytes) -> list[TypeDef]:
        """Extract struct, enum, trait, and type alias definitions."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        type_defs: list[TypeDef] = []

        for child in root.children:
            if child.type == "struct_item":
                name = _extract_struct_name(child)
                if name:
                    type_defs.append(
                        TypeDef(
                            name=name,
                            kind="struct",
                            line_start=child.start_point[0] + 1,
                            line_end=child.end_point[0] + 1,
                        )
                    )

            elif child.type == "enum_item":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    name_node = _child_of_type(child, "type_identifier")
                if name_node is not None:
                    type_defs.append(
                        TypeDef(
                            name=_node_text(name_node),
                            kind="enum",
                            line_start=child.start_point[0] + 1,
                            line_end=child.end_point[0] + 1,
                        )
                    )

            elif child.type == "trait_item":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    name_node = _child_of_type(child, "type_identifier")
                if name_node is not None:
                    type_defs.append(
                        TypeDef(
                            name=_node_text(name_node),
                            kind="interface",
                            line_start=child.start_point[0] + 1,
                            line_end=child.end_point[0] + 1,
                        )
                    )

            elif child.type == "type_item":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    name_node = _child_of_type(child, "type_identifier")
                if name_node is not None:
                    type_defs.append(
                        TypeDef(
                            name=_node_text(name_node),
                            kind="type_alias",
                            line_start=child.start_point[0] + 1,
                            line_end=child.end_point[0] + 1,
                        )
                    )

        return type_defs

    def extract_exports(self, file_path: str, source: bytes) -> list[Export]:
        """Extract pub items as exports."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        exports: list[Export] = []

        for child in root.children:
            if not _has_visibility_pub(child):
                continue

            if child.type == "function_item":
                name = _extract_fn_name(child)
                if name:
                    exports.append(Export(name=name, kind="function"))

            elif child.type == "struct_item":
                name = _extract_struct_name(child)
                if name:
                    exports.append(Export(name=name, kind="class"))

            elif child.type == "enum_item":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    name_node = _child_of_type(child, "type_identifier")
                if name_node is not None:
                    exports.append(Export(name=_node_text(name_node), kind="type"))

            elif child.type == "trait_item":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    name_node = _child_of_type(child, "type_identifier")
                if name_node is not None:
                    exports.append(Export(name=_node_text(name_node), kind="type"))

            elif child.type == "type_item":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    name_node = _child_of_type(child, "type_identifier")
                if name_node is not None:
                    exports.append(Export(name=_node_text(name_node), kind="type"))

            elif child.type == "const_item":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    name_node = _child_of_type(child, "identifier")
                if name_node is not None:
                    exports.append(Export(name=_node_text(name_node), kind="variable"))

            elif child.type == "static_item":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    name_node = _child_of_type(child, "identifier")
                if name_node is not None:
                    exports.append(Export(name=_node_text(name_node), kind="variable"))

        return exports

    def extract_constants(self, file_path: str, source: bytes) -> list[Constant]:
        """Extract const and static item definitions."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        constants: list[Constant] = []

        for child in root.children:
            if child.type == "const_item":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    name_node = _child_of_type(child, "identifier")
                if name_node is None:
                    continue
                name = _node_text(name_node)

                value_node = child.child_by_field_name("value")
                value: str | None = None
                if value_node is not None:
                    value = _node_text(value_node)

                constants.append(
                    Constant(
                        name=name,
                        value=value,
                        line=child.start_point[0] + 1,
                    )
                )

            elif child.type == "static_item":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    name_node = _child_of_type(child, "identifier")
                if name_node is None:
                    continue
                name = _node_text(name_node)

                value_node = child.child_by_field_name("value")
                value = None
                if value_node is not None:
                    value = _node_text(value_node)

                constants.append(
                    Constant(
                        name=name,
                        value=value,
                        line=child.start_point[0] + 1,
                    )
                )

        return constants

    def extract_routes(self, file_path: str, source: bytes) -> list[RouteHandler]:
        """Extract Actix/Rocket route handler annotations."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        routes: list[RouteHandler] = []

        for child in root.children:
            if child.type == "function_item":
                self._check_route_attributes(child, routes)
            elif child.type == "impl_item":
                decl_list = _child_of_type(child, "declaration_list")
                if decl_list is not None:
                    for item in decl_list.children:
                        if item.type == "function_item":
                            self._check_route_attributes(item, routes)

        # Also do a regex pass for attribute macros that tree-sitter might
        # associate differently
        text = source.decode("utf-8", errors="replace")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            match = _ROUTE_ATTR_RE.search(line)
            if match:
                method = match.group(1).upper()
                path = match.group(2)
                # Look for the next function name
                handler = ""
                for j in range(i + 1, min(i + 5, len(lines))):
                    fn_match = re.search(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", lines[j])
                    if fn_match:
                        handler = fn_match.group(1)
                        break
                # Avoid duplicates
                if not any(r.path == path and r.method == method for r in routes):
                    routes.append(
                        RouteHandler(
                            path=path,
                            method=method,
                            handler_function=handler,
                        )
                    )

        return routes

    def _check_route_attributes(self, node: Node, out: list[RouteHandler]) -> None:
        """Check attribute macros above a function_item for route patterns."""
        name = _extract_fn_name(node)
        prev = node.prev_named_sibling
        while prev is not None and prev.type == "attribute_item":
            text = _node_text(prev)
            match = _ROUTE_ATTR_RE.search(text)
            if match:
                method = match.group(1).upper()
                path = match.group(2)
                out.append(
                    RouteHandler(
                        path=path,
                        method=method,
                        handler_function=name,
                    )
                )
            prev = prev.prev_named_sibling

    def extract_db_models(self, file_path: str, source: bytes) -> list[DBModel]:
        """Extract structs with DB-related derive macros."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        models: list[DBModel] = []

        for child in root.children:
            if child.type != "struct_item":
                continue

            name = _extract_struct_name(child)
            if not name:
                continue

            derives = _extract_derive_names(child)
            is_db_model = bool(_DB_DERIVE_TRAITS.intersection(derives))

            if is_db_model:
                fields = _extract_struct_fields(child)
                models.append(DBModel(name=name, fields=fields))

        return models

    def extract_env_vars(self, file_path: str, source: bytes) -> list[str]:
        """Extract environment variable names from env::var calls."""
        try:
            text = source.decode("utf-8", errors="replace")
        except Exception:
            return []

        env_vars: list[str] = []
        seen: set[str] = set()

        for pattern in _ENV_PATTERNS:
            for match in pattern.finditer(text):
                var_name = match.group(1)
                if var_name not in seen:
                    seen.add(var_name)
                    env_vars.append(var_name)

        return env_vars
