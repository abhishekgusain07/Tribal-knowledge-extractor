"""Java-specific AST parser using tree-sitter."""

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


def _get_modifiers(node: Node) -> list[str]:
    """Extract modifier keywords (public, private, static, etc.) from a node."""
    modifiers_node = _child_of_type(node, "modifiers")
    if modifiers_node is None:
        return []
    mods: list[str] = []
    for child in modifiers_node.children:
        if child.type in (
            "public",
            "private",
            "protected",
            "static",
            "final",
            "abstract",
            "synchronized",
            "native",
            "transient",
            "volatile",
            "default",
        ):
            mods.append(_node_text(child))
        elif not child.is_named:
            text = _node_text(child)
            if text in (
                "public",
                "private",
                "protected",
                "static",
                "final",
                "abstract",
                "synchronized",
                "native",
                "transient",
                "volatile",
                "default",
            ):
                mods.append(text)
    return mods


def _get_annotations(node: Node) -> list[str]:
    """Extract annotation strings from a node's modifiers."""
    modifiers_node = _child_of_type(node, "modifiers")
    if modifiers_node is None:
        return []
    annotations: list[str] = []
    for child in modifiers_node.children:
        if child.type == "annotation" or child.type == "marker_annotation":
            annotations.append(_node_text(child))
    return annotations


def _get_javadoc_above(node: Node) -> str | None:
    """Extract Javadoc comment immediately before a node."""
    prev = node.prev_named_sibling
    # Skip annotations between comment and node
    while prev is not None and prev.type in ("annotation", "marker_annotation"):
        prev = prev.prev_named_sibling
    if prev is not None and prev.type == "block_comment":
        text = _node_text(prev)
        if text.startswith("/**"):
            return text
    # Also check directly for line_comment blocks
    if prev is not None and prev.type == "line_comment":
        lines: list[str] = []
        while prev is not None and prev.type == "line_comment":
            text = _node_text(prev)
            if text.startswith("//"):
                lines.insert(0, text[2:].strip())
            prev = prev.prev_named_sibling
        if lines:
            return "\n".join(lines)
    return None


def _extract_method_name(node: Node) -> str:
    """Extract the name from a method_declaration or constructor_declaration."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(name_node)
    ident = _child_of_type(node, "identifier")
    if ident is not None:
        return _node_text(ident)
    return ""


def _extract_parameters(node: Node) -> list[str]:
    """Extract parameter names from a method's formal_parameters node."""
    params: list[str] = []
    params_node = _child_of_type(node, "formal_parameters")
    if params_node is None:
        return params
    for child in params_node.children:
        if child.type == "formal_parameter":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                name_node = _child_of_type(child, "identifier")
            if name_node is not None:
                params.append(_node_text(name_node))
        elif child.type == "spread_parameter":
            name_node = _child_of_type(child, "variable_declarator")
            if name_node is None:
                name_node = _child_of_type(child, "identifier")
            if name_node is not None:
                params.append("..." + _node_text(name_node))
    return params


def _extract_return_type(node: Node) -> str | None:
    """Extract the return type from a method_declaration."""
    type_node = node.child_by_field_name("type")
    if type_node is not None:
        return _node_text(type_node)
    # Search among children for type nodes before the method name
    for child in node.children:
        if child.type in (
            "void_type",
            "integral_type",
            "floating_point_type",
            "boolean_type",
            "type_identifier",
            "generic_type",
            "array_type",
            "scoped_type_identifier",
        ):
            return _node_text(child)
        if child.type == "identifier":
            break
    return None


def _extract_class_name(node: Node) -> str:
    """Extract the name from a class/interface/enum/record declaration."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(name_node)
    ident = _child_of_type(node, "identifier")
    if ident is not None:
        return _node_text(ident)
    return ""


def _extract_superclass(node: Node) -> list[str]:
    """Extract superclass from a class declaration."""
    bases: list[str] = []
    superclass = _child_of_type(node, "superclass")
    if superclass is not None:
        for child in superclass.children:
            if child.type == "type_identifier" or child.type == "scoped_type_identifier" or child.type == "generic_type":
                bases.append(_node_text(child))
    return bases


def _extract_interfaces(node: Node) -> list[str]:
    """Extract implemented interfaces from a class declaration."""
    interfaces: list[str] = []
    iface_node = _child_of_type(node, "super_interfaces")
    if iface_node is None:
        # Try alternate name used in some tree-sitter versions
        iface_node = _child_of_type(node, "interfaces")
    if iface_node is not None:
        type_list = _child_of_type(iface_node, "type_list")
        if type_list is not None:
            for child in type_list.children:
                if child.type in ("type_identifier", "scoped_type_identifier", "generic_type"):
                    interfaces.append(_node_text(child))
        else:
            # Direct children
            for child in iface_node.children:
                if child.type in ("type_identifier", "scoped_type_identifier", "generic_type"):
                    interfaces.append(_node_text(child))
    return interfaces


def _build_method_signature(
    name: str,
    modifiers: list[str],
    params: list[str],
    return_type: str | None,
) -> str:
    """Build a concise Java method signature string."""
    parts: list[str] = []
    if modifiers:
        parts.append(" ".join(modifiers))
        parts.append(" ")
    if return_type:
        parts.append(f"{return_type} ")
    parts.append(f"{name}(")
    parts.append(", ".join(params))
    parts.append(")")
    return "".join(parts)


def _extract_field_name(field_node: Node) -> str | None:
    """Extract the field name from a field_declaration's variable_declarator."""
    for child in field_node.children:
        if child.type == "variable_declarator":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                name_node = _child_of_type(child, "identifier")
            if name_node is not None:
                return _node_text(name_node)
    return None


# ── Route annotation patterns ────────────────────────────────────────────

# Spring MVC mapping annotations
_SPRING_ROUTE_RE = re.compile(
    r'@(GetMapping|PostMapping|PutMapping|DeleteMapping|PatchMapping|RequestMapping)\s*'
    r'(?:\(\s*(?:value\s*=\s*)?(?:"([^"]*)"|\{[^}]*\})?'
    r'(?:.*?method\s*=\s*RequestMethod\.(\w+))?.*?\))?',
)

# JAX-RS annotations
_JAXRS_PATH_RE = re.compile(r'@Path\s*\(\s*"([^"]*)"\s*\)')
_JAXRS_METHOD_RE = re.compile(r"@(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)")

# ── Env var patterns ──────────────────────────────────────────────────────

_ENV_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'System\.getenv\s*\(\s*"([^"]+)"\s*\)'),
    re.compile(r'System\.getProperty\s*\(\s*"([^"]+)"\s*\)'),
]


# ── Main parser class ────────────────────────────────────────────────────


class JavaParser(BaseParser):
    """Tree-sitter based parser for Java source files."""

    language: str = "java"

    def __init__(self) -> None:
        self._parser = get_parser("java")

    # ── helpers ────────────────────────────────────────────────────────

    def _parse_tree(self, source: bytes) -> Node:
        """Parse source bytes and return the root node."""
        tree = self._parser.parse(source)
        return tree.root_node

    def detect_module_docstring(self, source: bytes) -> bool:
        """Detect whether the Java file starts with a Javadoc comment."""
        text = source.decode("utf-8", errors="replace").lstrip()
        return text.startswith("/**")

    # ── extraction methods ─────────────────────────────────────────────

    def extract_functions(self, file_path: str, source: bytes) -> list[FunctionDef]:
        """Extract method and constructor declarations from top-level classes."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        functions: list[FunctionDef] = []

        for child in root.children:
            if child.type in ("class_declaration", "interface_declaration", "enum_declaration", "record_declaration"):
                class_name = _extract_class_name(child)
                body = _child_of_type(child, "class_body")
                if body is None:
                    body = _child_of_type(child, "interface_body")
                if body is None:
                    body = _child_of_type(child, "enum_body")
                if body is not None:
                    self._extract_methods_from_body(body, class_name, functions)

        return functions

    def _extract_methods_from_body(
        self, body: Node, class_name: str, out: list[FunctionDef]
    ) -> None:
        """Extract method and constructor declarations from a class body."""
        for child in body.children:
            if child.type == "method_declaration":
                func = self._build_method_def(child, class_name)
                if func is not None:
                    out.append(func)
            elif child.type == "constructor_declaration":
                func = self._build_constructor_def(child, class_name)
                if func is not None:
                    out.append(func)

    def _build_method_def(self, node: Node, class_name: str) -> FunctionDef | None:
        """Build a FunctionDef from a method_declaration node."""
        name = _extract_method_name(node)
        if not name:
            return None

        modifiers = _get_modifiers(node)
        annotations = _get_annotations(node)
        params = _extract_parameters(node)
        return_type = _extract_return_type(node)
        docstring = _get_javadoc_above(node)
        is_async = False  # Java doesn't have async keyword; CompletableFuture is a type

        qualified_name = f"{class_name}.{name}" if class_name else name
        signature = _build_method_signature(name, modifiers, params, return_type)

        return FunctionDef(
            name=name,
            qualified_name=qualified_name,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            docstring=docstring,
            decorators=annotations,
            is_async=is_async,
            parameters=params,
            return_type=return_type,
        )

    def _build_constructor_def(self, node: Node, class_name: str) -> FunctionDef | None:
        """Build a FunctionDef from a constructor_declaration node."""
        name = _extract_method_name(node)
        if not name:
            name = class_name  # constructors have the class name

        modifiers = _get_modifiers(node)
        annotations = _get_annotations(node)
        params = _extract_parameters(node)
        docstring = _get_javadoc_above(node)

        qualified_name = f"{class_name}.{name}" if class_name else name
        signature = _build_method_signature(name, modifiers, params, None)

        return FunctionDef(
            name=name,
            qualified_name=qualified_name,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            docstring=docstring,
            decorators=annotations,
            is_async=False,
            parameters=params,
            return_type=None,
        )

    def extract_classes(self, file_path: str, source: bytes) -> list[ClassDef]:
        """Extract class, interface, enum, and record declarations."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        classes: list[ClassDef] = []

        for child in root.children:
            cls = self._extract_class_like(child)
            if cls is not None:
                classes.append(cls)

        return classes

    def _extract_class_like(self, node: Node) -> ClassDef | None:
        """Extract a ClassDef from a class/interface/enum/record declaration."""
        if node.type not in (
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        ):
            return None

        name = _extract_class_name(node)
        if not name:
            return None

        annotations = _get_annotations(node)
        docstring = _get_javadoc_above(node)

        # Bases: superclass + interfaces
        bases: list[str] = _extract_superclass(node) + _extract_interfaces(node)

        # For interfaces, check extends_interfaces
        extends_node = _child_of_type(node, "extends_interfaces")
        if extends_node is not None:
            type_list = _child_of_type(extends_node, "type_list")
            if type_list is not None:
                for child in type_list.children:
                    if child.type in ("type_identifier", "scoped_type_identifier", "generic_type"):
                        bases.append(_node_text(child))

        # Methods
        methods: list[FunctionDef] = []
        body = _child_of_type(node, "class_body")
        if body is None:
            body = _child_of_type(node, "interface_body")
        if body is None:
            body = _child_of_type(node, "enum_body")
        if body is not None:
            self._extract_methods_from_body(body, name, methods)

        return ClassDef(
            name=name,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            bases=bases,
            docstring=docstring,
            decorators=annotations,
            methods=methods,
        )

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

            text = _node_text(child).strip()
            # Remove trailing semicolons and "import" keyword
            if text.endswith(";"):
                text = text[:-1].strip()

            is_static = "static" in text
            if text.startswith("import static "):
                path = text[len("import static "):].strip()
            elif text.startswith("import "):
                path = text[len("import "):].strip()
            else:
                path = text

            # Handle wildcard imports: import java.util.*;
            is_wildcard = path.endswith(".*")
            if is_wildcard:
                module = path[:-2]
                names = ["*"]
            elif "." in path:
                module, name = path.rsplit(".", 1)
                names = [name]
            else:
                module = path
                names = []

            # Static imports get the method/field name as the imported name
            alias: str | None = None
            if is_static and names and names[0] != "*":
                alias = names[0]

            imports.append(
                Import(
                    module=module,
                    names=names,
                    is_relative=False,
                    alias=alias,
                )
            )

        return imports

    def extract_type_defs(self, file_path: str, source: bytes) -> list[TypeDef]:
        """Extract interface and annotation type definitions."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        type_defs: list[TypeDef] = []

        for child in root.children:
            if child.type == "interface_declaration":
                name = _extract_class_name(child)
                if name:
                    type_defs.append(
                        TypeDef(
                            name=name,
                            kind="interface",
                            line_start=child.start_point[0] + 1,
                            line_end=child.end_point[0] + 1,
                        )
                    )

            elif child.type == "annotation_type_declaration":
                name_node = _child_of_type(child, "identifier")
                if name_node is not None:
                    type_defs.append(
                        TypeDef(
                            name=_node_text(name_node),
                            kind="annotation",
                            line_start=child.start_point[0] + 1,
                            line_end=child.end_point[0] + 1,
                        )
                    )

            elif child.type == "enum_declaration":
                name = _extract_class_name(child)
                if name:
                    type_defs.append(
                        TypeDef(
                            name=name,
                            kind="enum",
                            line_start=child.start_point[0] + 1,
                            line_end=child.end_point[0] + 1,
                        )
                    )

        return type_defs

    def extract_exports(self, file_path: str, source: bytes) -> list[Export]:
        """Extract public top-level types as exports (Java convention)."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        exports: list[Export] = []

        for child in root.children:
            modifiers = _get_modifiers(child)
            if "public" not in modifiers:
                continue

            if child.type == "class_declaration":
                name = _extract_class_name(child)
                if name:
                    exports.append(Export(name=name, kind="class"))

            elif child.type == "interface_declaration":
                name = _extract_class_name(child)
                if name:
                    exports.append(Export(name=name, kind="type"))

            elif child.type == "enum_declaration":
                name = _extract_class_name(child)
                if name:
                    exports.append(Export(name=name, kind="type"))

            elif child.type == "record_declaration":
                name = _extract_class_name(child)
                if name:
                    exports.append(Export(name=name, kind="class"))

            elif child.type == "annotation_type_declaration":
                name_node = _child_of_type(child, "identifier")
                if name_node is not None:
                    exports.append(Export(name=_node_text(name_node), kind="type"))

        return exports

    def extract_constants(self, file_path: str, source: bytes) -> list[Constant]:
        """Extract static final field declarations from top-level classes."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        constants: list[Constant] = []

        for child in root.children:
            if child.type not in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
            ):
                continue
            body = _child_of_type(child, "class_body")
            if body is None:
                body = _child_of_type(child, "interface_body")
            if body is None:
                body = _child_of_type(child, "enum_body")
            if body is None:
                continue

            for member in body.children:
                if member.type != "field_declaration":
                    continue
                modifiers = _get_modifiers(member)
                if "static" not in modifiers or "final" not in modifiers:
                    continue

                field_name = _extract_field_name(member)
                if field_name is None:
                    continue

                # Extract value from variable_declarator
                value: str | None = None
                for sub in member.children:
                    if sub.type == "variable_declarator":
                        val_node = sub.child_by_field_name("value")
                        if val_node is not None:
                            value = _node_text(val_node)
                        break

                constants.append(
                    Constant(
                        name=field_name,
                        value=value,
                        line=member.start_point[0] + 1,
                    )
                )

        return constants

    def extract_routes(self, file_path: str, source: bytes) -> list[RouteHandler]:
        """Extract Spring MVC and JAX-RS route annotations."""
        try:
            text = source.decode("utf-8", errors="replace")
        except Exception:
            return []

        routes: list[RouteHandler] = []
        lines = text.splitlines()

        # Track class-level @RequestMapping or @Path
        class_path = ""
        for line in lines:
            stripped = line.strip()
            # Class-level @RequestMapping or @Path — typically before class declaration
            if re.search(r"class\s+\w+", stripped):
                break
            rm_match = re.search(r'@RequestMapping\s*\(\s*(?:value\s*=\s*)?["\']?([^"\')\s]+)', stripped)
            if rm_match:
                class_path = rm_match.group(1)
            path_match = _JAXRS_PATH_RE.search(stripped)
            if path_match:
                class_path = path_match.group(1)

        # Find method-level route annotations
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            # Spring MVC annotations
            spring_match = _SPRING_ROUTE_RE.search(stripped)
            if spring_match:
                annotation = spring_match.group(1)
                path = spring_match.group(2) or ""
                explicit_method = spring_match.group(3)

                # Determine HTTP method
                method_map: dict[str, str] = {
                    "GetMapping": "GET",
                    "PostMapping": "POST",
                    "PutMapping": "PUT",
                    "DeleteMapping": "DELETE",
                    "PatchMapping": "PATCH",
                }
                if annotation in method_map:
                    method = method_map[annotation]
                elif annotation == "RequestMapping" and explicit_method:
                    method = explicit_method.upper()
                elif annotation == "RequestMapping":
                    method = "*"
                else:
                    method = "*"

                full_path = class_path + path if class_path else path

                # Find the handler method name
                handler = ""
                for j in range(i + 1, min(i + 10, len(lines))):
                    fn_match = re.search(
                        r'(?:public|private|protected)?\s*(?:static\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\(',
                        lines[j],
                    )
                    if fn_match:
                        handler = fn_match.group(1)
                        break

                routes.append(
                    RouteHandler(
                        path=full_path,
                        method=method,
                        handler_function=handler,
                    )
                )
                i += 1
                continue

            # JAX-RS annotations
            jaxrs_method_match = _JAXRS_METHOD_RE.search(stripped)
            if jaxrs_method_match:
                method = jaxrs_method_match.group(1).upper()
                # Look for @Path on the same method or on a nearby line
                method_path = ""
                for j in range(max(0, i - 3), min(i + 3, len(lines))):
                    path_match = _JAXRS_PATH_RE.search(lines[j])
                    if path_match:
                        method_path = path_match.group(1)
                        break

                full_path = class_path + method_path if class_path else method_path

                handler = ""
                for j in range(i + 1, min(i + 10, len(lines))):
                    fn_match = re.search(
                        r'(?:public|private|protected)?\s*(?:static\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\(',
                        lines[j],
                    )
                    if fn_match:
                        handler = fn_match.group(1)
                        break

                routes.append(
                    RouteHandler(
                        path=full_path,
                        method=method,
                        handler_function=handler,
                    )
                )

            i += 1

        return routes

    def extract_db_models(self, file_path: str, source: bytes) -> list[DBModel]:
        """Extract JPA @Entity annotated classes."""
        try:
            root = self._parse_tree(source)
        except Exception:
            return []

        models: list[DBModel] = []

        for child in root.children:
            if child.type not in ("class_declaration", "record_declaration"):
                continue

            annotations = _get_annotations(child)
            is_entity = any("@Entity" in a for a in annotations)
            is_table = any("@Table" in a for a in annotations)

            if not (is_entity or is_table):
                continue

            name = _extract_class_name(child)
            if not name:
                continue

            # Extract field names from class body
            fields: list[str] = []
            body = _child_of_type(child, "class_body")
            if body is not None:
                for member in body.children:
                    if member.type == "field_declaration":
                        field_name = _extract_field_name(member)
                        if field_name is not None:
                            fields.append(field_name)

            models.append(DBModel(name=name, fields=fields))

        return models

    def extract_env_vars(self, file_path: str, source: bytes) -> list[str]:
        """Extract environment variable names from System.getenv calls."""
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
