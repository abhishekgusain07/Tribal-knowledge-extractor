"""Go-specific AST parser using tree-sitter."""

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


def _get_comment_above(node: Node) -> str | None:
    """Return the comment text immediately before *node*, if any."""
    prev = node.prev_named_sibling
    if prev is not None and prev.type == "comment":
        text = _node_text(prev)
        # Strip leading "//" and whitespace
        if text.startswith("//"):
            return text[2:].strip()
    return None


def _collect_comment_block_above(node: Node) -> str | None:
    """Collect consecutive // comment lines above a node into a docstring."""
    lines: list[str] = []
    prev = node.prev_named_sibling
    while prev is not None and prev.type == "comment":
        text = _node_text(prev)
        if text.startswith("//"):
            lines.insert(0, text[2:].strip())
        else:
            break
        prev = prev.prev_named_sibling
    if lines:
        return "\n".join(lines)
    return None


def _extract_parameters_from_param_list(param_list_node: Node) -> list[str]:
    """Extract parameter names from a Go parameter_list node."""
    params: list[str] = []
    for child in param_list_node.children:
        if child.type == "parameter_declaration":
            # Parameter names are identifiers before the type
            for sub in child.children:
                if sub.type == "identifier":
                    params.append(_node_text(sub))
        elif child.type == "variadic_parameter_declaration":
            for sub in child.children:
                if sub.type == "identifier":
                    params.append("..." + _node_text(sub))
    return params


def _extract_receiver_type(method_node: Node) -> str:
    """Extract the receiver type from a method_declaration node.

    Returns the type name (e.g. ``"MyStruct"`` or ``"*MyStruct"``).
    """
    param_list = _child_of_type(method_node, "parameter_list")
    if param_list is None:
        return ""
    for child in param_list.children:
        if child.type == "parameter_declaration":
            # The type is after the receiver name identifier
            for sub in child.children:
                if sub.type == "pointer_type":
                    inner = _child_of_type(sub, "type_identifier")
                    if inner is not None:
                        return "*" + _node_text(inner)
                elif sub.type == "type_identifier":
                    return _node_text(sub)
    return ""


def _build_function_signature(
    name: str,
    params: list[str],
    return_type: str | None,
    is_method: bool,
    receiver: str,
) -> str:
    """Build a concise Go function/method signature string."""
    parts: list[str] = []
    parts.append("func ")
    if is_method and receiver:
        parts.append(f"({receiver}) ")
    parts.append(f"{name}(")
    parts.append(", ".join(params))
    parts.append(")")
    if return_type:
        parts.append(f" {return_type}")
    return "".join(parts)


def _extract_return_type(func_node: Node) -> str | None:
    """Extract the return type from a function/method declaration.

    Looks for a result node (parameter_list for multiple returns, or a
    simple type_identifier / pointer_type / etc. for single return).
    """
    # In Go tree-sitter grammar, the result is in a "result" field
    # or appears after the parameters. We search for known return type nodes.
    children = list(func_node.children)
    # Find the index of the parameter_list(s) — the result is after the last one
    last_param_idx = -1
    for idx, child in enumerate(children):
        if child.type == "parameter_list":
            last_param_idx = idx

    if last_param_idx < 0:
        return None

    # Everything between the last param_list and the block is the return type
    for idx in range(last_param_idx + 1, len(children)):
        child = children[idx]
        if child.type == "block":
            break
        if child.type in (
            "type_identifier",
            "pointer_type",
            "qualified_type",
            "slice_type",
            "map_type",
            "interface_type",
            "struct_type",
            "array_type",
            "channel_type",
            "function_type",
        ):
            return _node_text(child)
        if child.type == "parameter_list":
            # Multiple return values: (int, error)
            return _node_text(child)
    return None


def _extract_struct_fields(body_node: Node) -> list[str]:
    """Extract field names from a struct's field_declaration_list."""
    fields: list[str] = []
    for child in body_node.children:
        if child.type == "field_declaration":
            for sub in child.children:
                if sub.type == "field_identifier":
                    fields.append(_node_text(sub))
                    break
    return fields


def _has_db_tag(field_node: Node) -> bool:
    """Check if a struct field has gorm:, db:, or bson: tags."""
    for child in field_node.children:
        if child.type == "raw_string_literal" or child.type == "interpreted_string_literal":
            tag_text = _node_text(child)
            if any(marker in tag_text for marker in ("gorm:", "db:", "bson:")):
                return True
    return False


# ── Route detection patterns ──────────────────────────────────────────────

_ROUTE_PATTERNS: list[re.Pattern[str]] = [
    # http.HandleFunc("/path", handler)
    re.compile(r'http\.HandleFunc\s*\(\s*"([^"]+)"'),
    # mux.Handle("/path", handler) or mux.HandleFunc("/path", handler)
    re.compile(r'\w+\.Handle(?:Func)?\s*\(\s*"([^"]+)"'),
    # r.GET("/path", handler), r.POST("/path", handler), etc.  (gin-style)
    re.compile(r'\w+\.(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s*\(\s*"([^"]+)"'),
    # router.Get("/path", handler), router.Post("/path", handler), etc. (chi-style)
    re.compile(r'\w+\.(Get|Post|Put|Delete|Patch|Head|Options)\s*\(\s*"([^"]+)"'),
]

# ── Env var patterns ──────────────────────────────────────────────────────

_ENV_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'os\.Getenv\s*\(\s*"([^"]+)"\s*\)'),
    re.compile(r'os\.LookupEnv\s*\(\s*"([^"]+)"\s*\)'),
]


# ── Main parser class ────────────────────────────────────────────────────


class GoParser(BaseParser):
    """Tree-sitter based parser for Go source files."""

    language: str = "go"

    def __init__(self) -> None:
        self._parser = get_parser("go")

    # ── helpers ────────────────────────────────────────────────────────

    def _parse_tree(self, source: bytes) -> Node:
        """Parse source bytes and return the root node."""
        tree = self._parser.parse(source)
        return tree.root_node

    def detect_module_docstring(self, source: bytes) -> bool:
        """Detect whether the Go file starts with a ``// Package ...`` comment."""
        text = source.decode("utf-8", errors="replace").lstrip()
        return text.startswith("// Package ")

    # ── extraction methods ─────────────────────────────────────────────

    def extract_functions(self, file_path: str, source: bytes) -> list[FunctionDef]:
        """Extract top-level function and method declarations."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        functions: list[FunctionDef] = []

        for child in root.children:
            if child.type == "function_declaration":
                func = self._extract_function_declaration(child)
                if func is not None:
                    functions.append(func)
            elif child.type == "method_declaration":
                func = self._extract_method_declaration(child)
                if func is not None:
                    functions.append(func)

        return functions

    def _extract_function_declaration(self, node: Node) -> FunctionDef | None:
        """Build a FunctionDef from a function_declaration node."""
        name_node = _child_of_type(node, "identifier")
        if name_node is None:
            return None
        name = _node_text(name_node)

        # Parameters
        param_list = _child_of_type(node, "parameter_list")
        params = _extract_parameters_from_param_list(param_list) if param_list is not None else []

        return_type = _extract_return_type(node)
        docstring = _collect_comment_block_above(node)
        signature = _build_function_signature(name, params, return_type, False, "")

        return FunctionDef(
            name=name,
            qualified_name=name,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            docstring=docstring,
            decorators=[],
            is_async=False,
            parameters=params,
            return_type=return_type,
        )

    def _extract_method_declaration(self, node: Node) -> FunctionDef | None:
        """Build a FunctionDef from a method_declaration node."""
        # Method name is a field_identifier in Go tree-sitter grammar
        name_node = _child_of_type(node, "field_identifier")
        if name_node is None:
            return None
        name = _node_text(name_node)

        receiver = _extract_receiver_type(node)
        receiver_base = receiver.lstrip("*")

        # Parameters — the second parameter_list is the method params
        param_lists = _children_of_type(node, "parameter_list")
        params: list[str] = []
        if len(param_lists) >= 2:
            params = _extract_parameters_from_param_list(param_lists[1])

        return_type = _extract_return_type(node)
        docstring = _collect_comment_block_above(node)
        qualified_name = f"{receiver_base}.{name}" if receiver_base else name
        signature = _build_function_signature(name, params, return_type, True, receiver)

        return FunctionDef(
            name=name,
            qualified_name=qualified_name,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            docstring=docstring,
            decorators=[],
            is_async=False,
            parameters=params,
            return_type=return_type,
        )

    def extract_classes(self, file_path: str, source: bytes) -> list[ClassDef]:
        """Extract struct type declarations as ClassDef (Go has no classes).

        Methods are associated with their receiver struct.
        """
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        # First pass: collect structs
        structs: dict[str, ClassDef] = {}

        for child in root.children:
            if child.type != "type_declaration":
                continue
            for spec in _children_of_type(child, "type_spec"):
                name_node = _child_of_type(spec, "type_identifier")
                if name_node is None:
                    continue
                struct_body = _child_of_type(spec, "struct_type")
                if struct_body is None:
                    continue

                name = _node_text(name_node)
                docstring = _collect_comment_block_above(child)

                # Extract field names as "bases" stand-in (embedded types)
                bases: list[str] = []
                field_decl_list = _child_of_type(struct_body, "field_declaration_list")
                if field_decl_list is not None:
                    for field in _children_of_type(field_decl_list, "field_declaration"):
                        # Embedded struct (no field name, just a type)
                        has_field_id = _child_of_type(field, "field_identifier")
                        if has_field_id is None:
                            type_id = _child_of_type(field, "type_identifier")
                            if type_id is not None:
                                bases.append(_node_text(type_id))
                            pointer = _child_of_type(field, "pointer_type")
                            if pointer is not None:
                                inner = _child_of_type(pointer, "type_identifier")
                                if inner is not None:
                                    bases.append("*" + _node_text(inner))

                structs[name] = ClassDef(
                    name=name,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    bases=bases,
                    docstring=docstring,
                    decorators=[],
                    methods=[],
                )

        # Second pass: associate methods with their receiver structs
        for child in root.children:
            if child.type != "method_declaration":
                continue
            receiver = _extract_receiver_type(child)
            receiver_base = receiver.lstrip("*")
            if receiver_base in structs:
                func = self._extract_method_declaration(child)
                if func is not None:
                    structs[receiver_base].methods.append(func)

        return list(structs.values())

    def extract_imports(self, file_path: str, source: bytes) -> list[Import]:
        """Extract import declarations."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        imports: list[Import] = []

        for child in root.children:
            if child.type != "import_declaration":
                continue
            # Single import: import "fmt"
            import_spec = _child_of_type(child, "import_spec")
            if import_spec is not None:
                self._process_import_spec(import_spec, imports)
                continue

            # Grouped imports: import ( ... )
            spec_list = _child_of_type(child, "import_spec_list")
            if spec_list is not None:
                for spec in _children_of_type(spec_list, "import_spec"):
                    self._process_import_spec(spec, imports)

        return imports

    def _process_import_spec(self, spec: Node, out: list[Import]) -> None:
        """Process a single import_spec and append to *out*."""
        path_node = _child_of_type(spec, "interpreted_string_literal")
        if path_node is None:
            return
        path = _node_text(path_node).strip('"')

        # Check for alias (named import)
        alias: str | None = None
        name_node = _child_of_type(spec, "package_identifier")
        if name_node is None:
            name_node = _child_of_type(spec, "dot")
            if name_node is not None:
                alias = "."
            else:
                name_node = _child_of_type(spec, "blank_identifier")
                if name_node is not None:
                    alias = "_"
        if name_node is not None and alias is None:
            alias = _node_text(name_node)

        # The "module" is the full import path; the package name is the last segment
        package_name = path.rsplit("/", 1)[-1] if "/" in path else path
        names: list[str] = [package_name]

        out.append(
            Import(
                module=path,
                names=names,
                is_relative=False,
                alias=alias,
            )
        )

    def extract_type_defs(self, file_path: str, source: bytes) -> list[TypeDef]:
        """Extract type declarations: structs, interfaces, and type aliases."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        type_defs: list[TypeDef] = []

        for child in root.children:
            if child.type != "type_declaration":
                continue
            for spec in _children_of_type(child, "type_spec"):
                name_node = _child_of_type(spec, "type_identifier")
                if name_node is None:
                    continue
                name = _node_text(name_node)

                # Determine the kind
                kind = "type_alias"
                if _child_of_type(spec, "struct_type") is not None:
                    kind = "struct"
                elif _child_of_type(spec, "interface_type") is not None:
                    kind = "interface"

                type_defs.append(
                    TypeDef(
                        name=name,
                        kind=kind,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                    )
                )

        return type_defs

    def extract_exports(self, file_path: str, source: bytes) -> list[Export]:
        """Extract exported identifiers (capitalized names in Go)."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        exports: list[Export] = []

        for child in root.children:
            if child.type == "function_declaration":
                name_node = _child_of_type(child, "identifier")
                if name_node is not None:
                    name = _node_text(name_node)
                    if name and name[0].isupper():
                        exports.append(Export(name=name, kind="function"))

            elif child.type == "method_declaration":
                name_node = _child_of_type(child, "field_identifier")
                if name_node is not None:
                    name = _node_text(name_node)
                    if name and name[0].isupper():
                        exports.append(Export(name=name, kind="function"))

            elif child.type == "type_declaration":
                for spec in _children_of_type(child, "type_spec"):
                    name_node = _child_of_type(spec, "type_identifier")
                    if name_node is not None:
                        name = _node_text(name_node)
                        if name and name[0].isupper():
                            exports.append(Export(name=name, kind="type"))

            elif child.type == "var_declaration":
                for spec in _children_of_type(child, "var_spec"):
                    for sub in spec.children:
                        if sub.type == "identifier":
                            name = _node_text(sub)
                            if name and name[0].isupper():
                                exports.append(Export(name=name, kind="variable"))

            elif child.type == "const_declaration":
                for spec in _children_of_type(child, "const_spec"):
                    for sub in spec.children:
                        if sub.type == "identifier":
                            name = _node_text(sub)
                            if name and name[0].isupper():
                                exports.append(Export(name=name, kind="variable"))

        return exports

    def extract_constants(self, file_path: str, source: bytes) -> list[Constant]:
        """Extract const declarations."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        constants: list[Constant] = []

        for child in root.children:
            if child.type != "const_declaration":
                continue
            for spec in _children_of_type(child, "const_spec"):
                name_node = _child_of_type(spec, "identifier")
                if name_node is None:
                    continue
                name = _node_text(name_node)

                # Extract value (last expression child)
                value: str | None = None
                expr_list = _child_of_type(spec, "expression_list")
                if expr_list is not None:
                    # Take the first expression
                    for sub in expr_list.children:
                        if sub.is_named:
                            value = _node_text(sub)
                            break
                else:
                    # Value might be a direct child
                    for sub in spec.children:
                        if sub.type in (
                            "interpreted_string_literal",
                            "raw_string_literal",
                            "int_literal",
                            "float_literal",
                            "rune_literal",
                            "true",
                            "false",
                            "iota",
                        ):
                            value = _node_text(sub)
                            break

                constants.append(
                    Constant(
                        name=name,
                        value=value,
                        line=spec.start_point[0] + 1,
                    )
                )

        return constants

    def extract_routes(self, file_path: str, source: bytes) -> list[RouteHandler]:
        """Extract HTTP route registrations from Go source."""
        try:
            text = source.decode("utf-8", errors="replace")
        except Exception:
            return []

        routes: list[RouteHandler] = []
        seen: set[tuple[str, str]] = set()

        for line in text.splitlines():
            stripped = line.strip()
            for pattern in _ROUTE_PATTERNS:
                match = pattern.search(stripped)
                if match is None:
                    continue
                groups = match.groups()
                if len(groups) == 1:
                    # http.HandleFunc or mux.Handle pattern
                    path = groups[0]
                    method = "*"
                elif len(groups) == 2:
                    # gin/chi style: method + path
                    method = groups[0].upper()
                    path = groups[1]
                else:
                    continue

                key = (path, method)
                if key not in seen:
                    seen.add(key)
                    # Try to extract handler function name
                    handler = ""
                    handler_match = re.search(
                        r',\s*(\w+)\s*[,)]', stripped[match.start():]
                    )
                    if handler_match:
                        handler = handler_match.group(1)
                    routes.append(
                        RouteHandler(
                            path=path,
                            method=method,
                            handler_function=handler,
                        )
                    )

        return routes

    def extract_db_models(self, file_path: str, source: bytes) -> list[DBModel]:
        """Extract structs with gorm:/db:/bson: tags as DB models."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        models: list[DBModel] = []

        for child in root.children:
            if child.type != "type_declaration":
                continue
            for spec in _children_of_type(child, "type_spec"):
                name_node = _child_of_type(spec, "type_identifier")
                if name_node is None:
                    continue
                struct_body = _child_of_type(spec, "struct_type")
                if struct_body is None:
                    continue

                field_decl_list = _child_of_type(struct_body, "field_declaration_list")
                if field_decl_list is None:
                    continue

                has_db_tags = False
                fields: list[str] = []

                for field in _children_of_type(field_decl_list, "field_declaration"):
                    field_name_node = _child_of_type(field, "field_identifier")
                    if field_name_node is not None:
                        fields.append(_node_text(field_name_node))
                    if _has_db_tag(field):
                        has_db_tags = True

                if has_db_tags:
                    name = _node_text(name_node)
                    models.append(DBModel(name=name, fields=fields))

        return models

    def extract_env_vars(self, file_path: str, source: bytes) -> list[str]:
        """Extract environment variable names from os.Getenv calls."""
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
