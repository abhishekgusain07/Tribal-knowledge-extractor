"""Symbol table construction, import resolution, and call graph extraction.

This module takes a list of ``FileExtraction`` objects (produced by the
language parsers) and resolves cross-file relationships:

1. **Symbol table** -- maps every defined name to its location.
2. **Import resolution** -- determines which file each import refers to.
3. **Call expression extraction** -- infers function-calls-function edges.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

from tribal_knowledge.models import (
    CallExpression,
    ClassDef,
    DependencyGraph,
    Export,
    FileExtraction,
    FunctionDef,
    Import,
    ResolvedImport,
    SymbolEntry,
)


# ---------------------------------------------------------------------------
# Well-known external packages by language
# ---------------------------------------------------------------------------

_PYTHON_EXTERNAL_TOPLEVEL: frozenset[str] = frozenset({
    "os", "sys", "re", "json", "math", "pathlib", "typing", "collections",
    "functools", "itertools", "abc", "dataclasses", "enum", "hashlib",
    "logging", "unittest", "pytest", "asyncio", "datetime", "io",
    "subprocess", "socket", "http", "urllib", "shutil", "copy", "time",
    "textwrap", "contextlib", "importlib", "inspect", "traceback",
    "argparse", "configparser", "csv", "sqlite3", "struct", "tempfile",
    "threading", "multiprocessing", "concurrent", "queue", "signal",
    "pprint", "string", "secrets", "uuid", "random", "decimal",
    "fractions", "statistics", "operator", "bisect", "heapq", "array",
    "weakref", "types", "pickle", "shelve", "glob", "fnmatch",
    "linecache", "tokenize", "pdb", "profile", "timeit", "warnings",
    # popular third-party
    "django", "flask", "fastapi", "celery", "requests", "httpx", "aiohttp",
    "sqlalchemy", "alembic", "pydantic", "boto3", "botocore", "numpy",
    "pandas", "scipy", "sklearn", "tensorflow", "torch", "transformers",
    "click", "typer", "rich", "uvicorn", "gunicorn", "starlette",
    "jinja2", "marshmallow", "sentry_sdk", "redis", "psycopg2",
    "pymongo", "motor", "stripe", "openai", "anthropic", "langchain",
    "networkx", "matplotlib", "pillow", "PIL", "yaml", "toml",
})

_TS_JS_EXTERNAL_PATTERNS: tuple[str, ...] = (
    "react", "next", "vue", "angular", "svelte", "express", "koa", "hapi",
    "lodash", "underscore", "axios", "node-fetch", "zod", "yup", "joi",
    "prisma", "drizzle", "typeorm", "sequelize", "mongoose",
    "tailwindcss", "postcss", "webpack", "vite", "esbuild", "rollup",
    "jest", "vitest", "mocha", "chai", "cypress", "playwright",
    "typescript", "ts-node", "tslib",
    "path", "fs", "os", "http", "https", "url", "util", "stream",
    "crypto", "child_process", "events", "buffer", "net", "dns", "tls",
    "querystring", "assert", "zlib", "cluster", "worker_threads",
)

_GO_STD_PREFIXES: tuple[str, ...] = (
    "fmt", "os", "io", "net", "log", "math", "sort", "sync", "time",
    "strings", "strconv", "encoding", "bytes", "bufio", "crypto",
    "context", "errors", "flag", "path", "regexp", "runtime",
    "testing", "reflect", "unsafe", "database", "html", "text",
    "archive", "compress", "container", "debug", "embed", "go",
    "hash", "image", "index", "mime", "plugin", "unicode",
)


# ---------------------------------------------------------------------------
# SymbolTable
# ---------------------------------------------------------------------------

class SymbolTable:
    """Maps fully-qualified names to their locations in the codebase.

    Qualified name format:
        ``file_path::ClassName.method_name``   (for methods)
        ``file_path::function_name``           (for top-level functions)
        ``file_path::ClassName``               (for classes)
        ``file_path::TypeName``                (for types)
    """

    def __init__(self) -> None:
        self.symbols: dict[str, SymbolEntry] = {}
        self.file_to_symbols: dict[str, list[str]] = {}
        self.name_to_files: dict[str, list[str]] = {}

    # -- helpers -----------------------------------------------------------

    def _register(
        self,
        qualified_name: str,
        file_path: str,
        line: int,
        kind: str,
        short_name: str,
    ) -> None:
        """Register a single symbol in all lookup structures."""
        self.symbols[qualified_name] = SymbolEntry(
            file_path=file_path,
            line=line,
            kind=kind,
            qualified_name=qualified_name,
        )
        self.file_to_symbols.setdefault(file_path, []).append(qualified_name)
        self.name_to_files.setdefault(short_name, []).append(file_path)

    # -- public API --------------------------------------------------------

    def register_extraction(self, file_path: str, extraction: FileExtraction) -> None:
        """Register all symbols from a single ``FileExtraction``."""

        # Top-level functions
        for func in extraction.functions:
            qname = f"{file_path}::{func.name}"
            self._register(qname, file_path, func.line_start, "function", func.name)

        # Classes and their methods
        for cls in extraction.classes:
            cls_qname = f"{file_path}::{cls.name}"
            self._register(cls_qname, file_path, cls.line_start, "class", cls.name)

            for method in cls.methods:
                method_qname = f"{file_path}::{cls.name}.{method.name}"
                short_method = f"{cls.name}.{method.name}"
                self._register(
                    method_qname, file_path, method.line_start, "function", short_method,
                )

        # Type definitions
        for td in extraction.type_defs:
            td_qname = f"{file_path}::{td.name}"
            self._register(td_qname, file_path, td.line_start, "type", td.name)

        # Exports -- also register as short names for cross-file resolution
        for export in extraction.exports:
            ex_qname = f"{file_path}::{export.name}"
            kind = export.kind if export.kind else "variable"
            # Only register if not already registered (avoid duplicating
            # functions/classes that are also exported).
            if ex_qname not in self.symbols:
                self._register(ex_qname, file_path, 0, kind, export.name)

        # Constants
        for const in extraction.constants:
            c_qname = f"{file_path}::{const.name}"
            if c_qname not in self.symbols:
                self._register(c_qname, file_path, const.line, "variable", const.name)


# ---------------------------------------------------------------------------
# Import resolution helpers
# ---------------------------------------------------------------------------

def _is_python_external(module: str) -> bool:
    """Return True if *module* looks like an external / stdlib Python import."""
    top = module.split(".")[0]
    return top in _PYTHON_EXTERNAL_TOPLEVEL


def _is_ts_js_external(module: str) -> bool:
    """Return True if *module* looks like an external TS/JS import."""
    # Relative imports start with . or /
    if module.startswith(".") or module.startswith("/"):
        return False
    # Scoped packages like @radix-ui/react-select are external
    if module.startswith("@"):
        # But @/ or @alias/ path aliases are NOT external -- handled elsewhere
        # Consider external if the scope looks like a published npm scope
        parts = module.split("/")
        if len(parts) >= 2 and not parts[0].endswith("/"):
            scope = parts[0]
            # Common path alias scopes: @/, @app/, @lib/, @src/, @components/
            if scope in ("@", "@app", "@lib", "@src", "@components"):
                return False
        return True
    # Check against known externals
    top = module.split("/")[0]
    return top in _TS_JS_EXTERNAL_PATTERNS


def _is_go_external(module: str) -> bool:
    """Return True if *module* is a Go standard library or external package."""
    top = module.split("/")[0]
    # Standard library packages have no dots in the first path component
    if top in _GO_STD_PREFIXES:
        return True
    # External packages typically have a domain (contains a dot)
    if "." in top:
        return True
    return False


def _resolve_python_import(
    imp: Import,
    source_file: str,
    repo_root: str,
    file_set: frozenset[str],
) -> ResolvedImport:
    """Resolve a single Python import statement."""
    imported_names: list[str] = list(imp.names) if imp.names else [imp.module.split(".")[-1]]

    # External check
    if not imp.is_relative and _is_python_external(imp.module):
        return ResolvedImport(
            source_file=source_file,
            target_file=imp.module,
            imported_names=imported_names,
            resolution="external",
        )

    module_parts = imp.module.split(".")
    source_dir = str(PurePosixPath(source_file).parent)

    # Relative import
    if imp.is_relative:
        # For relative imports, resolve against the source file's directory
        candidate_parts = source_dir.split("/") if source_dir != "." else []
        # The number of leading dots determines how many levels up
        # The module field for relative imports typically doesn't include leading dots;
        # the parser sets is_relative=True instead.
        # Try current package first, then parent directories.
        search_bases = [source_dir]
    else:
        # Absolute import -- check common Python source roots
        search_bases: list[str] = []
        # Check src/ prefix
        src_prefix = "src"
        if os.path.isdir(os.path.join(repo_root, src_prefix)):
            search_bases.append(src_prefix)
        # Check project root
        search_bases.append("")

    # Try to resolve the module path
    for base in search_bases:
        if base:
            candidate = "/".join([base] + module_parts)
        else:
            candidate = "/".join(module_parts)

        # Check as a package (directory with __init__.py)
        init_candidate = candidate + "/__init__.py"
        if init_candidate in file_set:
            return ResolvedImport(
                source_file=source_file,
                target_file=init_candidate,
                imported_names=imported_names,
                resolution="exact",
            )

        # Check as a module file
        py_candidate = candidate + ".py"
        if py_candidate in file_set:
            return ResolvedImport(
                source_file=source_file,
                target_file=py_candidate,
                imported_names=imported_names,
                resolution="exact",
            )

        # For "from X.Y import Z", X.Y might be a package and Z a module inside it
        if module_parts:
            parent = "/".join([base] + module_parts) if base else "/".join(module_parts)
            for name in imp.names:
                sub_module = parent + "/" + name + ".py"
                if sub_module in file_set:
                    return ResolvedImport(
                        source_file=source_file,
                        target_file=sub_module,
                        imported_names=imported_names,
                        resolution="exact",
                    )
                sub_package = parent + "/" + name + "/__init__.py"
                if sub_package in file_set:
                    return ResolvedImport(
                        source_file=source_file,
                        target_file=sub_package,
                        imported_names=imported_names,
                        resolution="exact",
                    )

    # Fuzzy: check if any file path ends with the module path
    module_path_suffix = "/".join(module_parts)
    for f in file_set:
        if f.endswith(module_path_suffix + ".py") or f.endswith(module_path_suffix + "/__init__.py"):
            return ResolvedImport(
                source_file=source_file,
                target_file=f,
                imported_names=imported_names,
                resolution="fuzzy",
            )

    # If not relative and not found, likely external
    if not imp.is_relative:
        return ResolvedImport(
            source_file=source_file,
            target_file=imp.module,
            imported_names=imported_names,
            resolution="external",
        )

    return ResolvedImport(
        source_file=source_file,
        target_file=imp.module,
        imported_names=imported_names,
        resolution="unresolved",
    )


def _load_tsconfig_paths(repo_root: str) -> dict[str, str]:
    """Load path aliases from tsconfig.json if it exists.

    Returns a mapping like ``{"@/": "src/"}``.
    """
    aliases: dict[str, str] = {}
    tsconfig_path = os.path.join(repo_root, "tsconfig.json")
    if not os.path.isfile(tsconfig_path):
        return aliases

    try:
        with open(tsconfig_path, "r", encoding="utf-8") as f:
            # Strip comments (simple single-line comment removal)
            lines: list[str] = []
            for line in f:
                stripped = line.lstrip()
                if stripped.startswith("//"):
                    continue
                lines.append(line)
            data: dict[str, object] = json.loads("".join(lines))

        compiler_options = data.get("compilerOptions", {})
        if not isinstance(compiler_options, dict):
            return aliases
        base_url: str = ""
        raw_base = compiler_options.get("baseUrl")
        if isinstance(raw_base, str):
            base_url = raw_base.rstrip("/")
        paths = compiler_options.get("paths", {})
        if not isinstance(paths, dict):
            return aliases
        for alias_pattern, targets in paths.items():
            if not isinstance(targets, list) or not targets:
                continue
            target = targets[0]
            if not isinstance(target, str):
                continue
            # Remove trailing /* glob
            alias_key = alias_pattern.replace("/*", "/").replace("*", "")
            target_val = target.replace("/*", "/").replace("*", "")
            if base_url and not target_val.startswith("/"):
                target_val = base_url + "/" + target_val
            # Normalize
            target_val = target_val.lstrip("./")
            aliases[alias_key] = target_val
    except (json.JSONDecodeError, OSError, KeyError):
        pass

    return aliases


def _resolve_ts_js_import(
    imp: Import,
    source_file: str,
    repo_root: str,
    file_set: frozenset[str],
    ts_aliases: dict[str, str],
    language: str,
) -> ResolvedImport:
    """Resolve a single TypeScript or JavaScript import statement."""
    module = imp.module
    imported_names: list[str] = list(imp.names) if imp.names else [module.split("/")[-1]]

    # External check
    if _is_ts_js_external(module):
        return ResolvedImport(
            source_file=source_file,
            target_file=module,
            imported_names=imported_names,
            resolution="external",
        )

    # Apply path aliases
    resolved_module = module
    for alias, target in ts_aliases.items():
        if module.startswith(alias):
            resolved_module = target + module[len(alias):]
            break

    # If it's a relative import, resolve against source directory
    if resolved_module.startswith("."):
        source_dir = str(PurePosixPath(source_file).parent)
        # Normalize: join source_dir with resolved_module
        combined = str(PurePosixPath(source_dir) / resolved_module)
        # Normalize parent references
        parts: list[str] = []
        for p in combined.split("/"):
            if p == "..":
                if parts:
                    parts.pop()
            elif p != ".":
                parts.append(p)
        resolved_module = "/".join(parts)

    # Strip leading ./ if any remains
    resolved_module = resolved_module.lstrip("./")

    # Try various extensions
    extensions: list[str]
    if language == "typescript":
        extensions = [".ts", ".tsx", ".js", ".jsx"]
    else:
        extensions = [".js", ".jsx", ".ts", ".tsx"]

    # Direct file match
    for ext in extensions:
        candidate = resolved_module + ext
        if candidate in file_set:
            return ResolvedImport(
                source_file=source_file,
                target_file=candidate,
                imported_names=imported_names,
                resolution="exact",
            )

    # Already has extension?
    if resolved_module in file_set:
        return ResolvedImport(
            source_file=source_file,
            target_file=resolved_module,
            imported_names=imported_names,
            resolution="exact",
        )

    # Index file in directory
    for ext in extensions:
        candidate = resolved_module + "/index" + ext
        if candidate in file_set:
            return ResolvedImport(
                source_file=source_file,
                target_file=candidate,
                imported_names=imported_names,
                resolution="exact",
            )

    # Fuzzy match: check if any file has the module as a suffix
    for f in file_set:
        f_no_ext = f.rsplit(".", 1)[0] if "." in f else f
        if f_no_ext.endswith(resolved_module):
            return ResolvedImport(
                source_file=source_file,
                target_file=f,
                imported_names=imported_names,
                resolution="fuzzy",
            )

    return ResolvedImport(
        source_file=source_file,
        target_file=module,
        imported_names=imported_names,
        resolution="unresolved",
    )


def _read_go_module_path(repo_root: str) -> str:
    """Read the module path from go.mod if present."""
    go_mod = os.path.join(repo_root, "go.mod")
    if not os.path.isfile(go_mod):
        return ""
    try:
        with open(go_mod, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("module "):
                    return line.split(None, 1)[1].strip()
    except OSError:
        pass
    return ""


def _resolve_go_import(
    imp: Import,
    source_file: str,
    repo_root: str,
    file_set: frozenset[str],
    go_module_path: str,
) -> ResolvedImport:
    """Resolve a single Go import statement."""
    module = imp.module
    imported_names: list[str] = list(imp.names) if imp.names else [module.split("/")[-1]]

    # Standard library / external
    if _is_go_external(module) and not module.startswith(go_module_path):
        return ResolvedImport(
            source_file=source_file,
            target_file=module,
            imported_names=imported_names,
            resolution="external",
        )

    # Internal package: strip module path prefix to get the directory
    internal_path = module
    if go_module_path and module.startswith(go_module_path):
        internal_path = module[len(go_module_path):].lstrip("/")

    # In Go, a package is a directory. Any .go file in that directory belongs to it.
    for f in file_set:
        if f.startswith(internal_path + "/") and f.endswith(".go"):
            return ResolvedImport(
                source_file=source_file,
                target_file=internal_path,
                imported_names=imported_names,
                resolution="exact",
            )

    # Also check without leading paths
    for f in file_set:
        parts = f.rsplit("/", 1)
        if len(parts) == 2:
            dir_part = parts[0]
            if dir_part.endswith(internal_path) and f.endswith(".go"):
                return ResolvedImport(
                    source_file=source_file,
                    target_file=dir_part,
                    imported_names=imported_names,
                    resolution="fuzzy",
                )

    return ResolvedImport(
        source_file=source_file,
        target_file=module,
        imported_names=imported_names,
        resolution="external",
    )


# ---------------------------------------------------------------------------
# Public API: resolve_imports
# ---------------------------------------------------------------------------

def resolve_imports(
    extractions: list[FileExtraction],
    repo_root: str,
) -> list[ResolvedImport]:
    """Resolve every import across all file extractions.

    Parameters
    ----------
    extractions:
        Parsed file extractions from the language parsers.
    repo_root:
        Absolute path to the repository root directory.

    Returns
    -------
    list[ResolvedImport]
        One ``ResolvedImport`` per import statement found in the codebase.
    """
    # Build a set of all known file paths for fast lookup
    file_set: frozenset[str] = frozenset(ext.file_path for ext in extractions)

    # Pre-load config files
    ts_aliases: dict[str, str] = _load_tsconfig_paths(repo_root)
    go_module_path: str = _read_go_module_path(repo_root)

    resolved: list[ResolvedImport] = []

    for extraction in extractions:
        for imp in extraction.imports:
            if extraction.language == "python":
                ri = _resolve_python_import(imp, extraction.file_path, repo_root, file_set)
            elif extraction.language in ("typescript", "javascript"):
                ri = _resolve_ts_js_import(
                    imp, extraction.file_path, repo_root, file_set,
                    ts_aliases, extraction.language,
                )
            elif extraction.language == "go":
                ri = _resolve_go_import(
                    imp, extraction.file_path, repo_root, file_set, go_module_path,
                )
            else:
                # For unsupported languages, mark everything as unresolved
                names: list[str] = list(imp.names) if imp.names else [imp.module]
                ri = ResolvedImport(
                    source_file=extraction.file_path,
                    target_file=imp.module,
                    imported_names=names,
                    resolution="unresolved",
                )
            resolved.append(ri)

    return resolved


# ---------------------------------------------------------------------------
# Call expression extraction
# ---------------------------------------------------------------------------

def _build_file_import_map(
    resolved_imports: list[ResolvedImport],
) -> dict[str, dict[str, str]]:
    """Build a mapping: source_file -> {imported_name: target_file}.

    This lets us quickly look up which file a name was imported from when
    processing a given source file.
    """
    file_import_map: dict[str, dict[str, str]] = {}
    for ri in resolved_imports:
        if ri.resolution in ("exact", "fuzzy"):
            name_map = file_import_map.setdefault(ri.source_file, {})
            for name in ri.imported_names:
                name_map[name] = ri.target_file
            # Also register the module short name itself
            module_short = ri.target_file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if module_short and module_short not in name_map:
                name_map[module_short] = ri.target_file
    return file_import_map


def _find_class_for_method(
    method_name: str,
    cls: ClassDef,
) -> bool:
    """Check if a class has a method with the given name."""
    return any(m.name == method_name for m in cls.methods)


def extract_call_expressions(
    extractions: list[FileExtraction],
    symbol_table: SymbolTable,
    resolved_imports: list[ResolvedImport],
) -> list[CallExpression]:
    """Extract and resolve call expressions from the extractions.

    Since we work with structured ``FileExtraction`` data (not raw ASTs),
    we infer call edges heuristically:

    - For each function, check which imported names or local names it
      might call based on parameter types, decorator references, and
      naming conventions (e.g., ``self.method`` within a class).
    - Class methods implicitly call other methods in the same class
      (via ``self`` / ``this``).
    - Inheritance-based calls: if class A extends B, A's methods may
      call B's methods.

    The primary goal is the file-level import graph (which is fully
    determined by ``resolved_imports``) and class-level relationships.
    Function-level call edges are inferred on a best-effort basis.
    """
    file_import_map = _build_file_import_map(resolved_imports)

    # Build a quick lookup: file -> extraction
    extraction_by_file: dict[str, FileExtraction] = {
        ext.file_path: ext for ext in extractions
    }

    # Build a lookup: file -> {function_name: qualified_name}
    file_functions: dict[str, dict[str, str]] = {}
    for ext in extractions:
        funcs: dict[str, str] = {}
        for func in ext.functions:
            funcs[func.name] = f"{ext.file_path}::{func.name}"
        for cls in ext.classes:
            for method in cls.methods:
                short = f"{cls.name}.{method.name}"
                funcs[short] = f"{ext.file_path}::{short}"
        file_functions[ext.file_path] = funcs

    calls: list[CallExpression] = []

    for ext in extractions:
        import_map = file_import_map.get(ext.file_path, {})
        local_funcs = file_functions.get(ext.file_path, {})

        # ---- Class method -> other methods in same class (self/this calls) ----
        for cls in ext.classes:
            method_names: list[str] = [m.name for m in cls.methods]
            for method in cls.methods:
                caller_qname = f"{ext.file_path}::{cls.name}.{method.name}"

                # Heuristic: each method might call other methods in the class
                # (especially __init__ calling helpers, or public methods
                # calling private ones). We infer these from parameter overlap
                # and naming. As a practical heuristic, connect methods that
                # share parameter names or where one method is likely called
                # from another.

                # self/this calls: for each other method in the class, emit a
                # potential call if the method name suggests a dependency.
                for other_method in cls.methods:
                    if other_method.name == method.name:
                        continue
                    # Skip dunder methods as callees (except __init__)
                    if (
                        other_method.name.startswith("__")
                        and other_method.name.endswith("__")
                        and other_method.name != "__init__"
                    ):
                        continue
                    # Connect __init__ to helper methods it likely calls
                    if method.name == "__init__" and not other_method.name.startswith("_"):
                        continue
                    # Private methods are likely called by public methods
                    if other_method.name.startswith("_") and not method.name.startswith("_"):
                        callee_raw = f"self.{other_method.name}"
                        callee_resolved = f"{ext.file_path}::{cls.name}.{other_method.name}"
                        calls.append(CallExpression(
                            caller=caller_qname,
                            callee_raw=callee_raw,
                            callee_resolved=callee_resolved,
                            file=ext.file_path,
                            line=method.line_start,
                        ))

                # Calls to imported names from class methods
                for imported_name, target_file in import_map.items():
                    # Check if the imported name appears in decorators
                    if imported_name in method.decorators:
                        calls.append(CallExpression(
                            caller=caller_qname,
                            callee_raw=imported_name,
                            callee_resolved=_resolve_callee(
                                imported_name, target_file, symbol_table,
                            ),
                            file=ext.file_path,
                            line=method.line_start,
                        ))
                    # Check if return type references the import
                    if method.return_type and imported_name in method.return_type:
                        calls.append(CallExpression(
                            caller=caller_qname,
                            callee_raw=imported_name,
                            callee_resolved=_resolve_callee(
                                imported_name, target_file, symbol_table,
                            ),
                            file=ext.file_path,
                            line=method.line_start,
                        ))

                # Inheritance: if this class extends a base, methods may call
                # base methods via super()
                for base_name in cls.bases:
                    if base_name in import_map:
                        target = import_map[base_name]
                        base_ext = extraction_by_file.get(target)
                        if base_ext:
                            for base_cls in base_ext.classes:
                                if base_cls.name == base_name:
                                    for base_method in base_cls.methods:
                                        if base_method.name == method.name:
                                            calls.append(CallExpression(
                                                caller=caller_qname,
                                                callee_raw=f"super().{base_method.name}",
                                                callee_resolved=f"{target}::{base_name}.{base_method.name}",
                                                file=ext.file_path,
                                                line=method.line_start,
                                            ))

        # ---- Top-level functions -> imported names ----
        for func in ext.functions:
            caller_qname = f"{ext.file_path}::{func.name}"

            # Check decorators for imported references
            for decorator in func.decorators:
                decorator_base = decorator.split("(")[0].split(".")[-1]
                if decorator_base in import_map:
                    target_file = import_map[decorator_base]
                    calls.append(CallExpression(
                        caller=caller_qname,
                        callee_raw=decorator_base,
                        callee_resolved=_resolve_callee(
                            decorator_base, target_file, symbol_table,
                        ),
                        file=ext.file_path,
                        line=func.line_start,
                    ))

            # Check return type for imported references
            if func.return_type:
                for imported_name, target_file in import_map.items():
                    if imported_name in func.return_type:
                        calls.append(CallExpression(
                            caller=caller_qname,
                            callee_raw=imported_name,
                            callee_resolved=_resolve_callee(
                                imported_name, target_file, symbol_table,
                            ),
                            file=ext.file_path,
                            line=func.line_start,
                        ))

            # Check parameters for imported type references
            for param in func.parameters:
                for imported_name, target_file in import_map.items():
                    if imported_name in param:
                        calls.append(CallExpression(
                            caller=caller_qname,
                            callee_raw=imported_name,
                            callee_resolved=_resolve_callee(
                                imported_name, target_file, symbol_table,
                            ),
                            file=ext.file_path,
                            line=func.line_start,
                        ))

            # A function may call other functions in the same file
            for other_name, other_qname in local_funcs.items():
                if other_qname == caller_qname:
                    continue
                # Heuristic: if a local function name appears in this
                # function's parameter types or return type, there may be
                # a dependency.
                if func.return_type and other_name.split(".")[-1] in func.return_type:
                    calls.append(CallExpression(
                        caller=caller_qname,
                        callee_raw=other_name,
                        callee_resolved=other_qname,
                        file=ext.file_path,
                        line=func.line_start,
                    ))

    return calls


def _resolve_callee(
    name: str,
    target_file: str,
    symbol_table: SymbolTable,
) -> str | None:
    """Resolve a callee name to its fully-qualified symbol.

    Resolution order:
    1. Exact match: ``target_file::name``
    2. Search symbol table by short name
    3. Return None if unresolved
    """
    # 1. Direct match in the target file
    candidate = f"{target_file}::{name}"
    if candidate in symbol_table.symbols:
        return candidate

    # 2. Search the symbol table by short name
    files = symbol_table.name_to_files.get(name, [])
    if len(files) == 1:
        # Unique match -- find the qualified name
        file_symbols = symbol_table.file_to_symbols.get(files[0], [])
        for qname in file_symbols:
            if qname.endswith(f"::{name}") or qname.endswith(f".{name}"):
                return qname

    if target_file in files:
        # Multiple matches but target file is one of them -- prefer it
        file_symbols = symbol_table.file_to_symbols.get(target_file, [])
        for qname in file_symbols:
            if qname.endswith(f"::{name}") or qname.endswith(f".{name}"):
                return qname

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def resolve_dependencies(
    extractions: list[FileExtraction],
    repo_root: str,
) -> DependencyGraph:
    """Build the complete dependency graph from parsed file extractions.

    This is the main entry point for Phase 1 dependency resolution.
    It orchestrates:

    1. Symbol table construction
    2. Import resolution
    3. Call expression extraction
    4. Statistics computation

    Parameters
    ----------
    extractions:
        List of ``FileExtraction`` objects from the language parsers.
    repo_root:
        Absolute path to the repository root.

    Returns
    -------
    DependencyGraph
        Contains resolved imports, call expressions, symbol table, and stats.
    """
    # Step 1: Build the symbol table
    symbol_table = SymbolTable()
    for ext in extractions:
        symbol_table.register_extraction(ext.file_path, ext)

    # Step 2: Resolve imports
    resolved_imports = resolve_imports(extractions, repo_root)

    # Step 3: Extract call expressions
    call_expressions = extract_call_expressions(
        extractions, symbol_table, resolved_imports,
    )

    # Step 4: Compute statistics
    total_calls = len(call_expressions)
    resolved_calls = sum(
        1 for ce in call_expressions if ce.callee_resolved is not None
    )
    resolution_rate = resolved_calls / total_calls if total_calls > 0 else 1.0

    return DependencyGraph(
        resolved_imports=resolved_imports,
        call_expressions=call_expressions,
        symbol_table=symbol_table.symbols,
        total_calls=total_calls,
        resolved_calls=resolved_calls,
        resolution_rate=resolution_rate,
    )
