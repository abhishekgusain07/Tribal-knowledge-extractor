"""Python-specific AST parser using tree-sitter."""

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


def _get_docstring(body_node: Node) -> str | None:
    """Extract docstring from the first child of a block if it is a string literal."""
    for child in body_node.children:
        if child.type == "string":
            raw = _node_text(child)
            # Strip triple-quote delimiters
            for delim in ('"""', "'''", 'r"""', "r'''"):
                if raw.startswith(delim):
                    inner = raw[len(delim) :]
                    end_delim = delim[-3:]  # always """ or '''
                    if inner.endswith(end_delim):
                        inner = inner[: -len(end_delim)]
                    return inner.strip()
            # Single-quote string used as docstring (unusual but valid)
            if (raw.startswith('"') and raw.endswith('"')) or (
                raw.startswith("'") and raw.endswith("'")
            ):
                return raw[1:-1].strip()
            return raw.strip()
        if child.type == "expression_statement":
            # Some tree-sitter versions wrap docstrings in expression_statement
            for sub in child.children:
                if sub.type == "string":
                    return _get_docstring_from_string_node(sub)
        # If the first non-comment/non-newline child is not a string, no docstring
        if child.type not in ("comment", "newline"):
            return None
    return None


def _get_docstring_from_string_node(node: Node) -> str | None:
    """Extract text from a string node, stripping delimiters."""
    raw = _node_text(node)
    for delim in ('"""', "'''", 'r"""', "r'''"):
        if raw.startswith(delim):
            end_delim = delim[-3:]
            inner = raw[len(delim) :]
            if inner.endswith(end_delim):
                inner = inner[: -len(end_delim)]
            return inner.strip()
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1].strip()
    return raw.strip()


def _get_decorators(node: Node) -> list[str]:
    """Extract decorator strings from a decorated_definition's parent or siblings."""
    decorators: list[str] = []
    parent = node.parent
    if parent is not None and parent.type == "decorated_definition":
        for child in parent.children:
            if child.type == "decorator":
                # The decorator text includes '@'; strip it
                dec_text = _node_text(child).lstrip("@").strip()
                decorators.append(dec_text)
    return decorators


def _get_parameters(params_node: Node) -> list[str]:
    """Extract parameter names from a parameters node."""
    params: list[str] = []
    for child in params_node.children:
        if child.type == "identifier":
            params.append(_node_text(child))
        elif child.type in ("typed_parameter", "default_parameter", "typed_default_parameter"):
            # First identifier child is the parameter name
            for sub in child.children:
                if sub.type == "identifier":
                    params.append(_node_text(sub))
                    break
        elif child.type == "list_splat_pattern":
            for sub in child.children:
                if sub.type == "identifier":
                    params.append("*" + _node_text(sub))
                    break
        elif child.type == "dictionary_splat_pattern":
            for sub in child.children:
                if sub.type == "identifier":
                    params.append("**" + _node_text(sub))
                    break
    return params


def _get_return_type(func_node: Node) -> str | None:
    """Extract the return type annotation from a function_definition node."""
    # The return type annotation appears as a `type` node child of the function_definition,
    # preceded by a `->` token.
    found_arrow = False
    for child in func_node.children:
        if child.type == "->" or _node_text(child) == "->":
            found_arrow = True
            continue
        if found_arrow and child.type == "type":
            return _node_text(child)
        if found_arrow and child.type != "type":
            # Could be a direct identifier in some grammars
            if child.type in ("identifier", "none", "attribute", "subscript", "generic_type"):
                return _node_text(child)
    return None


def _is_async(func_node: Node) -> bool:
    """Check if a function_definition node is async."""
    for child in func_node.children:
        if child.type == "async":
            return True
        # Stop once we hit 'def'
        if child.type == "def":
            break
    return False


def _get_function_body(func_node: Node) -> Node | None:
    """Get the block node (body) of a function_definition."""
    for child in func_node.children:
        if child.type == "block":
            return child
    return None


def _get_class_body(class_node: Node) -> Node | None:
    """Get the block node (body) of a class_definition."""
    for child in class_node.children:
        if child.type == "block":
            return child
    return None


def _get_bases(class_node: Node) -> list[str]:
    """Extract base class names from a class_definition node."""
    bases: list[str] = []
    for child in class_node.children:
        if child.type == "argument_list":
            for arg in child.children:
                if arg.type in ("identifier", "attribute", "subscript", "keyword_argument"):
                    # Skip metaclass=... keyword arguments for bases
                    if arg.type == "keyword_argument":
                        continue
                    bases.append(_node_text(arg))
    return bases


def _get_class_name(node: Node) -> str:
    """Get the name of a class_definition node."""
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child)
    return ""


def _get_func_name(node: Node) -> str:
    """Get the name of a function_definition node."""
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child)
    return ""


def _extract_string_content(node: Node) -> str:
    """Extract the raw string content from a string node (without quotes)."""
    for child in node.children:
        if child.type == "string_content":
            return _node_text(child)
    # Fallback: strip quotes manually
    raw = _node_text(node)
    if raw.startswith(('"""', "'''")):
        return raw[3:-3]
    if raw.startswith(('"', "'")):
        return raw[1:-1]
    return raw


# ── Route detection helpers ────────────────────────────────────────────────

_ROUTE_METHODS = {"get", "post", "put", "delete", "patch", "head", "options", "route"}


def _parse_route_decorator(decorator_text: str) -> tuple[str, str] | None:
    """Parse a route decorator and return (path, method) or None.

    Handles patterns like:
        app.route("/path", methods=["GET"])
        router.get("/path")
        app.post("/path")
    """
    # Pattern: something.method("/path")
    match = re.match(
        r'(\w+)\.(route|get|post|put|delete|patch|head|options)\s*\(\s*["\']([^"\']+)["\']',
        decorator_text,
    )
    if match:
        method_name = match.group(2)
        path = match.group(3)
        if method_name == "route":
            # Look for methods= keyword
            methods_match = re.search(
                r'methods\s*=\s*\[([^\]]+)\]', decorator_text
            )
            if methods_match:
                method = methods_match.group(1).replace('"', "").replace("'", "").strip()
                return (path, method.upper())
            return (path, "*")
        return (path, method_name.upper())
    return None


# ── DB Model detection helpers ─────────────────────────────────────────────

_DB_MODEL_BASES = {
    "Base",
    "db.Model",
    "models.Model",
    "Model",
}


def _is_db_model(bases: list[str]) -> bool:
    """Check if any base class indicates a DB model."""
    for base in bases:
        if base in _DB_MODEL_BASES:
            return True
    return False


_PYDANTIC_BASES = {"BaseModel"}


def _is_pydantic_model(bases: list[str]) -> bool:
    """Check if any base class indicates a Pydantic BaseModel."""
    for base in bases:
        if base in _PYDANTIC_BASES:
            return True
    return False


# ── Env var patterns ───────────────────────────────────────────────────────

_ENV_PATTERNS: list[re.Pattern[str]] = [
    # os.environ["KEY"] or os.environ['KEY']
    re.compile(r'os\.environ\s*\[\s*["\'](\w+)["\']\s*\]'),
    # os.environ.get("KEY") or os.environ.get('KEY')
    re.compile(r'os\.environ\.get\s*\(\s*["\'](\w+)["\']'),
    # os.getenv("KEY") or os.getenv('KEY')
    re.compile(r'os\.getenv\s*\(\s*["\'](\w+)["\']'),
]


# ── Main parser class ─────────────────────────────────────────────────────


class PythonParser(BaseParser):
    """Tree-sitter based parser for Python source files."""

    language: str = "python"

    def __init__(self) -> None:
        self._parser = get_parser("python")

    # ── helpers ────────────────────────────────────────────────────────

    def _parse_tree(self, source: bytes) -> Node:
        """Parse source bytes and return the root node."""
        tree = self._parser.parse(source)
        return tree.root_node

    def _extract_function(
        self, node: Node, class_name: str | None = None
    ) -> FunctionDef:
        """Build a FunctionDef from a function_definition node."""
        name = _get_func_name(node)
        decorators = _get_decorators(node)

        params_node: Node | None = None
        for child in node.children:
            if child.type == "parameters":
                params_node = child
                break

        parameters = _get_parameters(params_node) if params_node is not None else []
        return_type = _get_return_type(node)
        is_async = _is_async(node)

        body = _get_function_body(node)
        docstring = _get_docstring(body) if body is not None else None

        qualified_name = f"{class_name}.{name}" if class_name else name

        # Build signature
        sig_parts: list[str] = []
        if is_async:
            sig_parts.append("async ")
        sig_parts.append(f"def {name}(")
        sig_parts.append(", ".join(parameters))
        sig_parts.append(")")
        if return_type:
            sig_parts.append(f" -> {return_type}")
        signature = "".join(sig_parts)

        # Determine the actual definition node (might be wrapped in decorated_definition)
        definition_node = node
        if node.parent is not None and node.parent.type == "decorated_definition":
            definition_node = node.parent

        return FunctionDef(
            name=name,
            qualified_name=qualified_name,
            signature=signature,
            line_start=definition_node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            docstring=docstring,
            decorators=decorators,
            is_async=is_async,
            parameters=parameters,
            return_type=return_type,
        )

    # ── extraction methods ─────────────────────────────────────────────

    def extract_functions(self, file_path: str, source: bytes) -> list[FunctionDef]:
        """Extract top-level function definitions."""
        root = self._parse_tree(source)
        functions: list[FunctionDef] = []

        for child in root.children:
            if child.type == "function_definition":
                functions.append(self._extract_function(child))
            elif child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type == "function_definition":
                        functions.append(self._extract_function(sub))

        return functions

    def extract_classes(self, file_path: str, source: bytes) -> list[ClassDef]:
        """Extract class definitions with their methods."""
        root = self._parse_tree(source)
        classes: list[ClassDef] = []

        for child in root.children:
            class_node: Node | None = None
            decorators: list[str] = []

            if child.type == "class_definition":
                class_node = child
            elif child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type == "class_definition":
                        class_node = sub
                        decorators = _get_decorators(sub)

            if class_node is None:
                continue

            name = _get_class_name(class_node)
            bases = _get_bases(class_node)

            body = _get_class_body(class_node)
            docstring: str | None = None
            methods: list[FunctionDef] = []

            if body is not None:
                docstring = _get_docstring(body)

                for body_child in body.children:
                    if body_child.type == "function_definition":
                        methods.append(self._extract_function(body_child, class_name=name))
                    elif body_child.type == "decorated_definition":
                        for sub in body_child.children:
                            if sub.type == "function_definition":
                                methods.append(
                                    self._extract_function(sub, class_name=name)
                                )

            # Use the decorated_definition node for line_start if decorated
            definition_node = child if child.type == "decorated_definition" else class_node

            classes.append(
                ClassDef(
                    name=name,
                    line_start=definition_node.start_point[0] + 1,
                    line_end=class_node.end_point[0] + 1,
                    bases=bases,
                    docstring=docstring,
                    decorators=decorators,
                    methods=methods,
                )
            )

        return classes

    def extract_imports(self, file_path: str, source: bytes) -> list[Import]:
        """Extract import and from-import statements."""
        root = self._parse_tree(source)
        imports: list[Import] = []

        for child in root.children:
            if child.type == "import_statement":
                # import X, import X as Y, import X.Y
                for sub in child.children:
                    if sub.type == "dotted_name":
                        module_name = _node_text(sub)
                        imports.append(
                            Import(module=module_name, names=[], is_relative=False)
                        )
                    elif sub.type == "aliased_import":
                        dotted = None
                        alias = None
                        for part in sub.children:
                            if part.type == "dotted_name":
                                dotted = _node_text(part)
                            elif part.type == "identifier" and dotted is not None:
                                alias = _node_text(part)
                        if dotted:
                            imports.append(
                                Import(
                                    module=dotted,
                                    names=[],
                                    is_relative=False,
                                    alias=alias,
                                )
                            )

            elif child.type == "import_from_statement":
                module_name = ""
                is_relative = False
                names: list[str] = []
                alias: str | None = None

                for sub in child.children:
                    if sub.type == "relative_import":
                        is_relative = True
                        # Build module path from prefix dots + optional dotted_name
                        parts: list[str] = []
                        for ri_child in sub.children:
                            if ri_child.type == "import_prefix":
                                parts.append(_node_text(ri_child))
                            elif ri_child.type == "dotted_name":
                                parts.append(_node_text(ri_child))
                        module_name = "".join(parts)
                    elif sub.type == "dotted_name":
                        # Could be the module or an imported name
                        if module_name == "" and not is_relative:
                            # Haven't seen a module yet and it's not relative —
                            # this is the module in `from X import Y`
                            module_name = _node_text(sub)
                        else:
                            names.append(_node_text(sub))
                    elif sub.type == "aliased_import":
                        for part in sub.children:
                            if part.type == "dotted_name":
                                names.append(_node_text(part))
                            elif part.type == "identifier":
                                # The alias identifier (field name "alias")
                                field_name = sub.field_name_for_child(
                                    list(sub.children).index(part)
                                )
                                if field_name == "alias":
                                    alias = _node_text(part)
                    elif sub.type == "wildcard_import":
                        names.append("*")

                # For `from X import Y, Z` the imported names are dotted_name children
                # that come AFTER the `import` keyword.  We need a second pass to capture
                # them correctly, since the first dotted_name is the module itself.
                # The above logic already handles this — the first dotted_name becomes
                # module_name, subsequent ones become names.

                imports.append(
                    Import(
                        module=module_name,
                        names=names,
                        is_relative=is_relative,
                        alias=alias,
                    )
                )

        return imports

    def extract_type_defs(self, file_path: str, source: bytes) -> list[TypeDef]:
        """Extract type definitions: Protocol, TypedDict, Enum classes, TypeAlias assignments."""
        root = self._parse_tree(source)
        type_defs: list[TypeDef] = []

        for child in root.children:
            # Check for class-based type defs
            class_node: Node | None = None
            if child.type == "class_definition":
                class_node = child
            elif child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type == "class_definition":
                        class_node = sub

            if class_node is not None:
                name = _get_class_name(class_node)
                bases = _get_bases(class_node)

                kind = ""
                for base in bases:
                    base_simple = base.rsplit(".", 1)[-1]
                    if base_simple == "Protocol":
                        kind = "protocol"
                        break
                    elif base_simple == "TypedDict":
                        kind = "type_alias"
                        break
                    elif base_simple == "Enum":
                        kind = "enum"
                        break
                    elif base_simple == "IntEnum":
                        kind = "enum"
                        break
                    elif base_simple == "StrEnum":
                        kind = "enum"
                        break

                if kind:
                    type_defs.append(
                        TypeDef(
                            name=name,
                            kind=kind,
                            line_start=child.start_point[0] + 1,
                            line_end=class_node.end_point[0] + 1,
                        )
                    )
                continue

            # Check for TypeAlias assignments: X: TypeAlias = ...
            if child.type == "assignment":
                # Look for typed assignment with TypeAlias annotation
                has_type_alias = False
                var_name = ""
                for sub in child.children:
                    if sub.type == "identifier" and var_name == "":
                        var_name = _node_text(sub)
                    elif sub.type == "type":
                        type_text = _node_text(sub)
                        if "TypeAlias" in type_text:
                            has_type_alias = True

                if has_type_alias and var_name:
                    type_defs.append(
                        TypeDef(
                            name=var_name,
                            kind="type_alias",
                            line_start=child.start_point[0] + 1,
                            line_end=child.end_point[0] + 1,
                        )
                    )

        return type_defs

    def extract_exports(self, file_path: str, source: bytes) -> list[Export]:
        """Extract exports: __all__ list and public module-level names."""
        root = self._parse_tree(source)
        exports: list[Export] = []
        all_names: list[str] | None = None

        # First pass: look for __all__
        for child in root.children:
            if child.type == "assignment":
                for sub in child.children:
                    if sub.type == "identifier" and _node_text(sub) == "__all__":
                        # Find the list node
                        for val in child.children:
                            if val.type == "list":
                                all_names = []
                                for item in val.children:
                                    if item.type == "string":
                                        all_names.append(_extract_string_content(item))
                        break

        if all_names is not None:
            # Use __all__ as the authoritative export list
            for name in all_names:
                exports.append(Export(name=name, kind="variable"))
            return exports

        # Fallback: everything at module level without leading underscore
        for child in root.children:
            if child.type == "function_definition":
                name = _get_func_name(child)
                if not name.startswith("_"):
                    exports.append(Export(name=name, kind="function"))
            elif child.type == "class_definition":
                name = _get_class_name(child)
                if not name.startswith("_"):
                    exports.append(Export(name=name, kind="class"))
            elif child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type == "function_definition":
                        name = _get_func_name(sub)
                        if not name.startswith("_"):
                            exports.append(Export(name=name, kind="function"))
                    elif sub.type == "class_definition":
                        name = _get_class_name(sub)
                        if not name.startswith("_"):
                            exports.append(Export(name=name, kind="class"))
            elif child.type == "assignment":
                for sub in child.children:
                    if sub.type == "identifier":
                        name = _node_text(sub)
                        if not name.startswith("_"):
                            exports.append(Export(name=name, kind="variable"))
                        break  # only first identifier (the target)

        return exports

    def extract_constants(self, file_path: str, source: bytes) -> list[Constant]:
        """Extract module-level UPPER_CASE constant assignments."""
        root = self._parse_tree(source)
        constants: list[Constant] = []

        for child in root.children:
            if child.type != "assignment":
                continue

            # Get the target name
            target_name = ""
            value_text: str | None = None

            for sub in child.children:
                if sub.type == "identifier" and target_name == "":
                    target_name = _node_text(sub)
                elif sub.type in (
                    "integer",
                    "float",
                    "string",
                    "true",
                    "false",
                    "none",
                    "unary_operator",
                    "concatenated_string",
                ):
                    value_text = _node_text(sub)

            # Check if name is UPPER_CASE (with underscores)
            if target_name and re.match(r"^[A-Z][A-Z0-9_]*$", target_name):
                constants.append(
                    Constant(
                        name=target_name,
                        value=value_text,
                        line=child.start_point[0] + 1,
                    )
                )

        return constants

    def extract_routes(self, file_path: str, source: bytes) -> list[RouteHandler]:
        """Extract Flask/FastAPI route handlers from decorated functions."""
        root = self._parse_tree(source)
        routes: list[RouteHandler] = []

        def _scan_for_routes(node: Node) -> None:
            """Recursively scan for decorated functions with route decorators."""
            for child in node.children:
                if child.type == "decorated_definition":
                    func_node: Node | None = None
                    decorator_texts: list[str] = []

                    for sub in child.children:
                        if sub.type == "decorator":
                            decorator_texts.append(_node_text(sub).lstrip("@").strip())
                        elif sub.type == "function_definition":
                            func_node = sub

                    if func_node is not None:
                        func_name = _get_func_name(func_node)
                        for dec_text in decorator_texts:
                            route_info = _parse_route_decorator(dec_text)
                            if route_info is not None:
                                path, method = route_info
                                routes.append(
                                    RouteHandler(
                                        path=path,
                                        method=method,
                                        handler_function=func_name,
                                    )
                                )

                # Also recurse into class bodies for class-based views
                if child.type == "class_definition":
                    body = _get_class_body(child)
                    if body is not None:
                        _scan_for_routes(body)

        _scan_for_routes(root)
        return routes

    def extract_db_models(self, file_path: str, source: bytes) -> list[DBModel]:
        """Extract SQLAlchemy, Django ORM, and Pydantic DB models."""
        root = self._parse_tree(source)
        db_models: list[DBModel] = []

        for child in root.children:
            class_node: Node | None = None
            if child.type == "class_definition":
                class_node = child
            elif child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type == "class_definition":
                        class_node = sub

            if class_node is None:
                continue

            name = _get_class_name(class_node)
            bases = _get_bases(class_node)

            if not _is_db_model(bases):
                continue

            # Extract field names from class body assignments
            fields: list[str] = []
            body = _get_class_body(class_node)
            if body is not None:
                for body_child in body.children:
                    if body_child.type == "assignment":
                        for sub in body_child.children:
                            if sub.type == "identifier":
                                field_name = _node_text(sub)
                                # Skip dunder attributes like __tablename__
                                if not field_name.startswith("__"):
                                    fields.append(field_name)
                                break
                    elif body_child.type == "expression_statement":
                        # Handle type-annotated fields: field_name: Type
                        for sub in body_child.children:
                            if sub.type == "assignment":
                                for part in sub.children:
                                    if part.type == "identifier":
                                        field_name = _node_text(part)
                                        if not field_name.startswith("__"):
                                            fields.append(field_name)
                                        break

            db_models.append(DBModel(name=name, fields=fields))

        return db_models

    def extract_env_vars(self, file_path: str, source: bytes) -> list[str]:
        """Extract environment variable names from os.environ and os.getenv calls."""
        text = source.decode("utf-8", errors="replace")
        env_vars: list[str] = []
        seen: set[str] = set()

        for pattern in _ENV_PATTERNS:
            for match in pattern.finditer(text):
                var_name = match.group(1)
                if var_name not in seen:
                    seen.add(var_name)
                    env_vars.append(var_name)

        return env_vars
