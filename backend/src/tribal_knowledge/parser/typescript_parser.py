"""TypeScript / TSX parser built on tree-sitter."""

from __future__ import annotations

import re

from tree_sitter import Node, Parser, Tree
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

# HTTP methods recognised when detecting Express / Hono / generic router patterns
_HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "options", "head", "all"})

# Router-like receiver names
_ROUTER_NAMES = frozenset({"app", "router", "server", "hono"})

# Regex for UPPER_CASE constant names (at least two chars to avoid false positives)
_UPPER_CASE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Next.js file-based route patterns
_NEXTJS_APP_ROUTE_RE = re.compile(r"app/api/.+/route\.(ts|tsx|js|jsx)$")
_NEXTJS_PAGES_API_RE = re.compile(r"pages/api/.+\.(ts|tsx|js|jsx)$")


# ── Helpers ────────────────────────────────────────────────────────────────


def _node_text(node: Node) -> str:
    """Decode node text to a UTF-8 string."""
    raw: bytes = node.text or b""
    return raw.decode("utf-8", errors="replace")


def _child_of_type(node: Node, type_name: str) -> Node | None:
    """Return the first named child whose ``type`` matches *type_name*."""
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _children_of_type(node: Node, type_name: str) -> list[Node]:
    """Return all named children whose ``type`` matches *type_name*."""
    return [c for c in node.children if c.type == type_name]


def _has_anonymous_child(node: Node, text: str) -> bool:
    """Return True if *node* has a non-named child whose text equals *text*."""
    for child in node.children:
        if not child.is_named and _node_text(child) == text:
            return True
    return False


def _is_async(node: Node) -> bool:
    """Check whether a function / method node has the ``async`` keyword."""
    return _has_anonymous_child(node, "async")


def _formal_params(node: Node) -> list[str]:
    """Extract parameter names from a ``formal_parameters`` node."""
    params_node = _child_of_type(node, "formal_parameters")
    if params_node is None:
        return []
    result: list[str] = []
    for child in params_node.named_children:
        if child.type in ("required_parameter", "optional_parameter"):
            pattern = child.child_by_field_name("pattern")
            if pattern is not None:
                result.append(_node_text(pattern))
            else:
                # Fall back: first identifier child
                ident = _child_of_type(child, "identifier")
                if ident is not None:
                    result.append(_node_text(ident))
        elif child.type == "identifier":
            result.append(_node_text(child))
        elif child.type == "rest_pattern":
            ident = _child_of_type(child, "identifier")
            if ident is not None:
                result.append(f"...{_node_text(ident)}")
    return result


def _return_type(node: Node) -> str | None:
    """Extract the return type annotation string (without leading colon)."""
    ta = _child_of_type(node, "type_annotation")
    if ta is None:
        return None
    # type_annotation text is ': ReturnType' — strip leading ': '
    raw = _node_text(ta)
    stripped = raw.lstrip(":").strip()
    return stripped if stripped else None


def _jsoc_above(node: Node) -> str | None:
    """Return the JSDoc comment text immediately before *node*, if any.

    Walks backwards past decorator nodes to find a ``/** ... */`` comment.
    """
    prev = node.prev_named_sibling
    # Skip over decorator siblings (decorators appear between comment and method)
    while prev is not None and prev.type == "decorator":
        prev = prev.prev_named_sibling
    if prev is not None and prev.type == "comment":
        text = _node_text(prev)
        if text.startswith("/**"):
            return text
    return None


def _decorator_texts(node: Node) -> list[str]:
    """Collect decorator strings attached to a class declaration node."""
    decorators: list[str] = []
    for child in node.children:
        if child.type == "decorator":
            decorators.append(_node_text(child))
    return decorators


def _method_decorators_from_body(body_children: list[Node], method_index: int) -> list[str]:
    """Collect decorator nodes that immediately precede a method_definition
    at *method_index* inside a class_body's children list."""
    decorators: list[str] = []
    idx = method_index - 1
    while idx >= 0 and body_children[idx].type == "decorator":
        decorators.insert(0, _node_text(body_children[idx]))
        idx -= 1
    return decorators


def _signature_text(node: Node) -> str:
    """Build a concise one-line signature from a function / method node."""
    parts: list[str] = []
    if _is_async(node):
        parts.append("async")
    name_node = node.child_by_field_name("name")
    name = _node_text(name_node) if name_node else ""
    if node.type == "function_declaration":
        parts.append(f"function {name}")
    elif node.type == "arrow_function":
        parts.append(name)
    elif node.type == "method_definition":
        parts.append(name)
    else:
        parts.append(name)
    params_node = _child_of_type(node, "formal_parameters")
    if params_node is not None:
        parts.append(_node_text(params_node))
    rt = _return_type(node)
    if rt:
        parts[-1] = parts[-1] + f": {rt}" if parts else f": {rt}"
    return " ".join(parts)


def _get_ts_parser(file_path: str) -> Parser:
    """Return the correct tree-sitter parser based on file extension."""
    if file_path.endswith(".tsx") or file_path.endswith(".jsx"):
        return get_parser("tsx")
    return get_parser("typescript")


def _parse_tree(file_path: str, source: bytes, lang: str | None = None) -> Tree:
    """Parse *source* and return the tree-sitter Tree."""
    if lang is not None:
        parser = get_parser(lang)
    else:
        parser = _get_ts_parser(file_path)
    return parser.parse(source)


def _extract_fields_from_object(obj_node: Node) -> list[str]:
    """Extract field names from an object literal node."""
    fields: list[str] = []
    for child in obj_node.named_children:
        if child.type == "pair":
            key = child.child_by_field_name("key")
            if key is None:
                key = _child_of_type(child, "property_identifier")
            if key is not None:
                fields.append(_node_text(key))
        elif child.type == "property_identifier":
            fields.append(_node_text(child))
        elif child.type == "shorthand_property_identifier":
            fields.append(_node_text(child))
    return fields


# ── Main parser class ─────────────────────────────────────────────────────


class TypeScriptParser(BaseParser):
    """Extract structural information from TypeScript / TSX source files."""

    language: str = "typescript"

    def __init__(self, lang_override: str | None = None) -> None:
        self._lang_override: str | None = lang_override

    def _parse(self, file_path: str, source: bytes) -> Tree:
        return _parse_tree(file_path, source, lang=self._lang_override)

    # ── Functions ─────────────────────────────────────────────────────

    def extract_functions(self, file_path: str, source: bytes) -> list[FunctionDef]:
        tree = self._parse(file_path, source)
        root = tree.root_node
        functions: list[FunctionDef] = []

        for node in root.named_children:
            # Top-level function declarations
            if node.type == "function_declaration":
                functions.append(self._function_from_declaration(node))

            # export function ...
            elif node.type == "export_statement":
                for child in node.named_children:
                    if child.type == "function_declaration":
                        functions.append(self._function_from_declaration(child))
                    elif child.type == "lexical_declaration":
                        functions.extend(self._arrow_functions_from_lexical(child))

            # const fn = (...) => { ... }
            elif node.type == "lexical_declaration" or node.type == "variable_declaration":
                functions.extend(self._arrow_functions_from_lexical(node))

        return functions

    def _function_from_declaration(self, node: Node) -> FunctionDef:
        name_node = node.child_by_field_name("name")
        name = _node_text(name_node) if name_node else "<anonymous>"
        return FunctionDef(
            name=name,
            qualified_name=name,
            signature=_signature_text(node),
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            docstring=_jsoc_above(node),
            decorators=[],
            is_async=_is_async(node),
            parameters=_formal_params(node),
            return_type=_return_type(node),
        )

    def _arrow_functions_from_lexical(self, node: Node) -> list[FunctionDef]:
        results: list[FunctionDef] = []
        for declarator in _children_of_type(node, "variable_declarator"):
            name_node = declarator.child_by_field_name("name")
            if name_node is None:
                name_node = _child_of_type(declarator, "identifier")
            if name_node is None:
                continue
            value_node = declarator.child_by_field_name("value")
            if value_node is None:
                # Fallback: look for arrow_function among children
                value_node = _child_of_type(declarator, "arrow_function")
            if value_node is None:
                continue
            if value_node.type != "arrow_function":
                continue
            name = _node_text(name_node)
            results.append(
                FunctionDef(
                    name=name,
                    qualified_name=name,
                    signature=f"const {name} = {_signature_text(value_node)}",
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    docstring=_jsoc_above(node),
                    decorators=[],
                    is_async=_is_async(value_node),
                    parameters=_formal_params(value_node),
                    return_type=_return_type(value_node),
                )
            )
        return results

    # ── Classes ───────────────────────────────────────────────────────

    def extract_classes(self, file_path: str, source: bytes) -> list[ClassDef]:
        tree = self._parse(file_path, source)
        root = tree.root_node
        classes: list[ClassDef] = []

        for node in root.named_children:
            if node.type == "class_declaration":
                classes.append(self._class_from_node(node))
            elif node.type == "export_statement":
                for child in node.named_children:
                    if child.type == "class_declaration":
                        classes.append(self._class_from_node(child))

        return classes

    def _class_from_node(self, node: Node) -> ClassDef:
        name_node = node.child_by_field_name("name")
        name = _node_text(name_node) if name_node else "<anonymous>"

        # Heritage: extends / implements
        bases: list[str] = []
        heritage = _child_of_type(node, "class_heritage")
        if heritage is not None:
            extends = _child_of_type(heritage, "extends_clause")
            if extends is not None:
                for c in extends.named_children:
                    bases.append(_node_text(c))
            implements = _child_of_type(heritage, "implements_clause")
            if implements is not None:
                for c in implements.named_children:
                    bases.append(_node_text(c))

        # Decorators
        decorators = _decorator_texts(node)

        # Methods
        methods: list[FunctionDef] = []
        body = node.child_by_field_name("body")
        if body is not None:
            body_children = list(body.named_children)
            for idx, child in enumerate(body_children):
                if child.type == "method_definition":
                    method_name_node = child.child_by_field_name("name")
                    method_name = _node_text(method_name_node) if method_name_node else "<anonymous>"
                    method_decorators = _method_decorators_from_body(body_children, idx)
                    methods.append(
                        FunctionDef(
                            name=method_name,
                            qualified_name=f"{name}.{method_name}",
                            signature=_signature_text(child),
                            line_start=child.start_point[0] + 1,
                            line_end=child.end_point[0] + 1,
                            docstring=_jsoc_above(child),
                            decorators=method_decorators,
                            is_async=_is_async(child),
                            parameters=_formal_params(child),
                            return_type=_return_type(child),
                        )
                    )

        return ClassDef(
            name=name,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            bases=bases,
            docstring=_jsoc_above(node),
            decorators=decorators,
            methods=methods,
        )

    # ── Imports ───────────────────────────────────────────────────────

    def extract_imports(self, file_path: str, source: bytes) -> list[Import]:
        tree = self._parse(file_path, source)
        root = tree.root_node
        imports: list[Import] = []

        for node in root.named_children:
            if node.type == "import_statement":
                self._parse_import_statement(node, imports)
            elif node.type in ("lexical_declaration", "variable_declaration"):
                self._parse_require_imports(node, imports)

        return imports

    def _parse_import_statement(self, node: Node, out: list[Import]) -> None:
        """Parse an ES ``import`` statement into one or more :class:`Import` objects."""
        source_node = _child_of_type(node, "string")
        if source_node is None:
            return
        frag = _child_of_type(source_node, "string_fragment")
        module = _node_text(frag) if frag is not None else _node_text(source_node).strip("\"'")

        is_relative = module.startswith("./") or module.startswith("../")

        clause = _child_of_type(node, "import_clause")
        if clause is None:
            # Side-effect import: import 'module'
            out.append(Import(module=module, names=[], is_relative=is_relative))
            return

        names: list[str] = []
        alias: str | None = None

        for child in clause.named_children:
            if child.type == "identifier":
                # Default import: import Foo from 'module'
                names.append(_node_text(child))
            elif child.type == "named_imports":
                for spec in _children_of_type(child, "import_specifier"):
                    spec_name_node = spec.child_by_field_name("name")
                    if spec_name_node is None:
                        spec_name_node = _child_of_type(spec, "identifier")
                    if spec_name_node is not None:
                        names.append(_node_text(spec_name_node))
            elif child.type == "namespace_import":
                # import * as X from 'module'
                ident = _child_of_type(child, "identifier")
                if ident is not None:
                    alias = _node_text(ident)
                    names.append("*")

        out.append(
            Import(
                module=module,
                names=names,
                is_relative=is_relative,
                alias=alias,
            )
        )

    def _parse_require_imports(self, node: Node, out: list[Import]) -> None:
        """Extract ``require("module")`` calls from variable declarations."""
        for declarator in _children_of_type(node, "variable_declarator"):
            value_node = declarator.child_by_field_name("value")
            if value_node is None:
                value_node = _child_of_type(declarator, "call_expression")
            if value_node is None or value_node.type != "call_expression":
                continue
            fn_node = value_node.child_by_field_name("function")
            if fn_node is None:
                fn_node = _child_of_type(value_node, "identifier")
            if fn_node is None or _node_text(fn_node) != "require":
                continue
            args = value_node.child_by_field_name("arguments")
            if args is None:
                args = _child_of_type(value_node, "arguments")
            if args is None:
                continue
            str_node = _child_of_type(args, "string")
            if str_node is None:
                continue
            frag = _child_of_type(str_node, "string_fragment")
            module = _node_text(frag) if frag else _node_text(str_node).strip("\"'")
            is_relative = module.startswith("./") or module.startswith("../")

            name_node = declarator.child_by_field_name("name")
            if name_node is None:
                name_node = _child_of_type(declarator, "identifier")
                if name_node is None:
                    name_node = _child_of_type(declarator, "object_pattern")

            names: list[str] = []
            alias: str | None = None
            if name_node is not None:
                if name_node.type == "identifier":
                    names = [_node_text(name_node)]
                    alias = _node_text(name_node)
                elif name_node.type == "object_pattern":
                    for prop in name_node.named_children:
                        if prop.type == "shorthand_property_identifier_pattern":
                            names.append(_node_text(prop))
                        elif prop.type == "pair_pattern":
                            key = prop.child_by_field_name("key")
                            if key is not None:
                                names.append(_node_text(key))
            out.append(Import(
                module=module,
                names=names,
                is_relative=is_relative,
                alias=alias,
            ))

    # ── Type Definitions ──────────────────────────────────────────────

    def extract_type_defs(self, file_path: str, source: bytes) -> list[TypeDef]:
        tree = self._parse(file_path, source)
        root = tree.root_node
        type_defs: list[TypeDef] = []

        for node in root.named_children:
            td = self._type_def_from_node(node)
            if td is not None:
                type_defs.append(td)
            # Also check inside export_statement
            if node.type == "export_statement":
                for child in node.named_children:
                    td = self._type_def_from_node(child)
                    if td is not None:
                        type_defs.append(td)

        return type_defs

    def _type_def_from_node(self, node: Node) -> TypeDef | None:
        if node.type == "interface_declaration":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                name_node = _child_of_type(node, "type_identifier")
            name = _node_text(name_node) if name_node else "<anonymous>"
            return TypeDef(
                name=name,
                kind="interface",
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
            )
        if node.type == "type_alias_declaration":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                name_node = _child_of_type(node, "type_identifier")
            name = _node_text(name_node) if name_node else "<anonymous>"
            return TypeDef(
                name=name,
                kind="type_alias",
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
            )
        if node.type == "enum_declaration":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                name_node = _child_of_type(node, "identifier")
            name = _node_text(name_node) if name_node else "<anonymous>"
            return TypeDef(
                name=name,
                kind="enum",
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
            )
        return None

    # ── Exports ───────────────────────────────────────────────────────

    def extract_exports(self, file_path: str, source: bytes) -> list[Export]:
        tree = self._parse(file_path, source)
        root = tree.root_node
        exports: list[Export] = []

        for node in root.named_children:
            if node.type != "export_statement":
                continue

            is_default = _has_anonymous_child(node, "default")

            # export { foo, bar } from './utils' or export { foo, bar }
            export_clause = _child_of_type(node, "export_clause")
            if export_clause is not None:
                for spec in _children_of_type(export_clause, "export_specifier"):
                    ident = _child_of_type(spec, "identifier")
                    if ident is not None:
                        exports.append(Export(name=_node_text(ident), kind="variable"))
                continue

            # Iterate named children for the exported declaration
            for child in node.named_children:
                if child.type == "function_declaration":
                    fname_node = child.child_by_field_name("name")
                    fname = _node_text(fname_node) if fname_node else "<anonymous>"
                    kind = "default" if is_default else "function"
                    exports.append(Export(name=fname, kind=kind))

                elif child.type == "class_declaration":
                    cname_node = child.child_by_field_name("name")
                    cname = _node_text(cname_node) if cname_node else "<anonymous>"
                    kind = "default" if is_default else "class"
                    exports.append(Export(name=cname, kind=kind))

                elif child.type in ("interface_declaration", "type_alias_declaration", "enum_declaration"):
                    td = self._type_def_from_node(child)
                    if td is not None:
                        exports.append(Export(name=td.name, kind="type"))

                elif child.type == "lexical_declaration" or child.type == "variable_declaration":
                    for decl in _children_of_type(child, "variable_declarator"):
                        vname_node = decl.child_by_field_name("name")
                        if vname_node is None:
                            vname_node = _child_of_type(decl, "identifier")
                        if vname_node is not None:
                            exports.append(Export(name=_node_text(vname_node), kind="variable"))

                elif child.type == "identifier":
                    # export default SomeIdentifier;
                    exports.append(Export(name=_node_text(child), kind="default"))

            # If we only saw 'export default' with no handled children,
            # try to see if the export statement has a bare expression
            if is_default and not exports:
                # Default export of an expression
                exports.append(Export(name="default", kind="default"))

        return exports

    # ── Constants ─────────────────────────────────────────────────────

    def extract_constants(self, file_path: str, source: bytes) -> list[Constant]:
        tree = self._parse(file_path, source)
        root = tree.root_node
        constants: list[Constant] = []

        for node in root.named_children:
            self._constants_from_node(node, constants)
            if node.type == "export_statement":
                for child in node.named_children:
                    self._constants_from_node(child, constants)

        return constants

    def _constants_from_node(self, node: Node, out: list[Constant]) -> None:
        if node.type not in ("lexical_declaration", "variable_declaration"):
            return
        # Must be 'const'
        if not _has_anonymous_child(node, "const"):
            return
        for decl in _children_of_type(node, "variable_declarator"):
            name_node = decl.child_by_field_name("name")
            if name_node is None:
                name_node = _child_of_type(decl, "identifier")
            if name_node is None:
                continue
            name = _node_text(name_node)
            if not _UPPER_CASE_RE.match(name):
                continue
            # Value
            value_node = decl.child_by_field_name("value")
            if value_node is None:
                # Try to get last named child that is not identifier
                for c in reversed(decl.named_children):
                    if c != name_node:
                        value_node = c
                        break
            value: str | None = None
            if value_node is not None and value_node.type in (
                "number",
                "string",
                "true",
                "false",
                "null",
                "template_string",
            ):
                value = _node_text(value_node)
            out.append(Constant(name=name, value=value, line=node.start_point[0] + 1))

    # ── Routes ────────────────────────────────────────────────────────

    def extract_routes(self, file_path: str, source: bytes) -> list[RouteHandler]:
        tree = self._parse(file_path, source)
        root = tree.root_node
        routes: list[RouteHandler] = []

        # Walk AST looking for call expressions that match route patterns
        self._walk_for_routes(root, routes)

        # Next.js file-based routing detection
        routes.extend(self._detect_nextjs_routes(file_path, source))

        # tRPC procedure detection
        self._walk_for_trpc(root, routes)

        return routes

    def _walk_for_routes(self, node: Node, out: list[RouteHandler]) -> None:
        """Recursively walk the tree looking for Express / Hono route patterns."""
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node is not None and func_node.type == "member_expression":
                obj = func_node.child_by_field_name("object")
                prop = func_node.child_by_field_name("property")
                if obj is None:
                    # Fallback
                    obj = _child_of_type(func_node, "identifier")
                if prop is None:
                    prop = _child_of_type(func_node, "property_identifier")
                if obj is not None and prop is not None:
                    obj_name = _node_text(obj)
                    method_name = _node_text(prop)
                    if (obj_name in _ROUTER_NAMES or obj_name.endswith("Router")) and method_name in _HTTP_METHODS:
                        args_node = node.child_by_field_name("arguments")
                        if args_node is None:
                            args_node = _child_of_type(node, "arguments")
                        path = ""
                        handler = ""
                        if args_node is not None and len(args_node.named_children) > 0:
                            first_arg = args_node.named_children[0]
                            if first_arg.type == "string":
                                frag = _child_of_type(first_arg, "string_fragment")
                                path = _node_text(frag) if frag else _node_text(first_arg).strip("\"'")
                            elif first_arg.type == "template_string":
                                path = _node_text(first_arg)
                            if len(args_node.named_children) > 1:
                                second_arg = args_node.named_children[-1]
                                if second_arg.type == "identifier":
                                    handler = _node_text(second_arg)
                                elif second_arg.type == "arrow_function":
                                    handler = "<arrow>"
                                elif second_arg.type == "function_expression":
                                    handler = "<anonymous>"
                        out.append(
                            RouteHandler(
                                path=path,
                                method=method_name.upper(),
                                handler_function=handler,
                            )
                        )

        for child in node.named_children:
            self._walk_for_routes(child, out)

    def _detect_nextjs_routes(self, file_path: str, source: bytes) -> list[RouteHandler]:
        """Detect Next.js file-based routes from the file path."""
        routes: list[RouteHandler] = []
        normalized = file_path.replace("\\", "/")

        if _NEXTJS_APP_ROUTE_RE.search(normalized):
            # app/api/*/route.ts — extract route path from directory structure
            match = re.search(r"app(/api/.+)/route\.[jt]sx?$", normalized)
            if match:
                route_path = match.group(1)
                # Check which HTTP methods are exported
                tree = self._parse(file_path, source)
                root = tree.root_node
                exported_methods: list[str] = []
                for node in root.named_children:
                    if node.type == "export_statement":
                        for child in node.named_children:
                            if child.type == "function_declaration":
                                fname_node = child.child_by_field_name("name")
                                if fname_node is not None:
                                    fname = _node_text(fname_node).upper()
                                    if fname in _HTTP_METHODS or fname in {m.upper() for m in _HTTP_METHODS}:
                                        exported_methods.append(fname)
                if exported_methods:
                    for method in exported_methods:
                        routes.append(RouteHandler(path=route_path, method=method, handler_function=method))
                else:
                    routes.append(RouteHandler(path=route_path, method="*"))

        elif _NEXTJS_PAGES_API_RE.search(normalized):
            match = re.search(r"pages(/api/.+)\.[jt]sx?$", normalized)
            if match:
                route_path = match.group(1)
                routes.append(RouteHandler(path=route_path, method="*"))

        return routes

    def _walk_for_trpc(self, node: Node, out: list[RouteHandler]) -> None:
        """Detect tRPC procedure definitions like ``router.query(...)`` or
        procedure-builder chains (``publicProcedure.input(...).query(...)``)."""
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node is not None and func_node.type == "member_expression":
                prop = func_node.child_by_field_name("property")
                if prop is None:
                    prop = _child_of_type(func_node, "property_identifier")
                if prop is not None:
                    prop_name = _node_text(prop)
                    if prop_name in ("query", "mutation", "subscription"):
                        # Try to find the procedure name from the parent context
                        # Common pattern: someRouter = router({ procedureName: ... })
                        # Or: export const appRouter = t.router({ hello: publicProcedure.query(...) })
                        out.append(
                            RouteHandler(
                                path=f"trpc.{prop_name}",
                                method=prop_name.upper(),
                                handler_function="",
                            )
                        )

        for child in node.named_children:
            self._walk_for_trpc(child, out)

    # ── DB Models ─────────────────────────────────────────────────────

    def extract_db_models(self, file_path: str, source: bytes) -> list[DBModel]:
        tree = self._parse(file_path, source)
        root = tree.root_node
        models: list[DBModel] = []

        self._walk_for_db_models(root, models)

        # TypeORM @Entity() decorated classes
        for node in root.named_children:
            cls_node: Node | None = None
            if node.type == "class_declaration":
                cls_node = node
            elif node.type == "export_statement":
                for child in node.named_children:
                    if child.type == "class_declaration":
                        cls_node = child
                        break
            if cls_node is not None:
                decos = _decorator_texts(cls_node)
                for d in decos:
                    if "@Entity" in d:
                        name_node = cls_node.child_by_field_name("name")
                        name = _node_text(name_node) if name_node else "<anonymous>"
                        # Collect field names from class body
                        body = cls_node.child_by_field_name("body")
                        fields: list[str] = []
                        if body is not None:
                            for member in body.named_children:
                                if member.type == "public_field_definition":
                                    prop = _child_of_type(member, "property_identifier")
                                    if prop is not None:
                                        fields.append(_node_text(prop))
                        models.append(DBModel(name=name, fields=fields))
                        break

        return models

    def _walk_for_db_models(self, node: Node, out: list[DBModel]) -> None:
        """Detect Mongoose, Prisma, and Drizzle DB model patterns."""
        # Mongoose: new Schema({...}) or new mongoose.Schema({...})
        if node.type == "new_expression":
            constructor = _child_of_type(node, "identifier")
            is_schema = constructor is not None and _node_text(constructor) == "Schema"
            # Also handle member_expression: new mongoose.Schema(...)
            if not is_schema:
                member = _child_of_type(node, "member_expression")
                if member is not None:
                    prop = member.child_by_field_name("property")
                    if prop is None:
                        prop = _child_of_type(member, "property_identifier")
                    if prop is not None and _node_text(prop) == "Schema":
                        is_schema = True
            if is_schema:
                args = _child_of_type(node, "arguments")
                fields: list[str] = []
                if args is not None and len(args.named_children) > 0:
                    obj = args.named_children[0]
                    if obj.type == "object":
                        fields = _extract_fields_from_object(obj)
                # Try to find the variable name this is assigned to
                name = "Schema"
                parent = node.parent
                if parent is not None and parent.type == "variable_declarator":
                    vname = parent.child_by_field_name("name")
                    if vname is None:
                        vname = _child_of_type(parent, "identifier")
                    if vname is not None:
                        name = _node_text(vname)
                out.append(DBModel(name=name, fields=fields))

        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node is None:
                func_node = _child_of_type(node, "member_expression")

            if func_node is not None and func_node.type == "member_expression":
                obj = func_node.child_by_field_name("object")
                prop = func_node.child_by_field_name("property")
                if obj is None:
                    obj = _child_of_type(func_node, "identifier")
                if prop is None:
                    prop = _child_of_type(func_node, "property_identifier")
                if obj is not None and prop is not None:
                    obj_text = _node_text(obj)
                    prop_text = _node_text(prop)

                    # mongoose.model('Name', schema)
                    if obj_text == "mongoose" and prop_text == "model":
                        args = node.child_by_field_name("arguments")
                        if args is None:
                            args = _child_of_type(node, "arguments")
                        name = "Model"
                        if args is not None and len(args.named_children) > 0:
                            first = args.named_children[0]
                            if first.type == "string":
                                frag = _child_of_type(first, "string_fragment")
                                name = _node_text(frag) if frag else _node_text(first).strip("\"'")
                        out.append(DBModel(name=name, fields=[]))

            # Drizzle: pgTable('name', {...}), mysqlTable, sqliteTable
            if func_node is not None and func_node.type == "identifier":
                func_name = _node_text(func_node)
                if func_name in ("pgTable", "mysqlTable", "sqliteTable"):
                    args = node.child_by_field_name("arguments")
                    if args is None:
                        args = _child_of_type(node, "arguments")
                    name = func_name
                    fields = []
                    if args is not None:
                        named = args.named_children
                        if len(named) > 0 and named[0].type == "string":
                            frag = _child_of_type(named[0], "string_fragment")
                            name = _node_text(frag) if frag else _node_text(named[0]).strip("\"'")
                        if len(named) > 1 and named[1].type == "object":
                            fields = _extract_fields_from_object(named[1])
                    # Try to get name from variable assignment
                    parent = node.parent
                    if parent is not None and parent.type == "variable_declarator":
                        vname = parent.child_by_field_name("name")
                        if vname is None:
                            vname = _child_of_type(parent, "identifier")
                        # Keep the table string name, not variable
                    out.append(DBModel(name=name, fields=fields))

        for child in node.named_children:
            self._walk_for_db_models(child, out)

    # ── Env Vars ──────────────────────────────────────────────────────

    def extract_env_vars(self, file_path: str, source: bytes) -> list[str]:
        tree = self._parse(file_path, source)
        root = tree.root_node
        env_vars: list[str] = []
        self._walk_for_env_vars(root, env_vars)
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for var in env_vars:
            if var not in seen:
                seen.add(var)
                result.append(var)
        return result

    def _walk_for_env_vars(self, node: Node, out: list[str]) -> None:
        """Find ``process.env.X`` and ``process.env["X"]`` patterns."""
        # process.env.VAR_NAME (member_expression)
        if node.type == "member_expression":
            obj = node.child_by_field_name("object")
            prop = node.child_by_field_name("property")
            if obj is None:
                # Fallback: first child that could be object
                for c in node.named_children:
                    if c.type == "member_expression":
                        obj = c
                        break
            if prop is None:
                prop = _child_of_type(node, "property_identifier")
            if obj is not None and prop is not None:
                if _node_text(obj) == "process.env" and prop.type == "property_identifier":
                    out.append(_node_text(prop))
                    return  # Don't recurse into children

        # process.env["VAR_NAME"] (subscript_expression)
        if node.type == "subscript_expression":
            obj = node.child_by_field_name("object")
            index = node.child_by_field_name("index")
            if obj is None:
                for c in node.named_children:
                    if c.type == "member_expression":
                        obj = c
                        break
            if index is None:
                for c in node.named_children:
                    if c.type == "string":
                        index = c
                        break
            if obj is not None and index is not None:
                if _node_text(obj) == "process.env" and index.type == "string":
                    frag = _child_of_type(index, "string_fragment")
                    var_name = _node_text(frag) if frag else _node_text(index).strip("\"'")
                    out.append(var_name)
                    return

        for child in node.named_children:
            self._walk_for_env_vars(child, out)
