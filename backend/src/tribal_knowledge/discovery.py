"""File discovery: walk a repo tree, classify files, and compute metadata."""

from __future__ import annotations

import re
from pathlib import Path

import pathspec

from tribal_knowledge.models import (
    EXTENSION_MAP,
    DiscoveredFile,
    FileType,
    compute_file_hash,
)

# ── Skip rules ─────────────────────────────────────────────────────────────

SKIP_DIRS: frozenset[str] = frozenset({
    "node_modules",
    ".git",
    "__pycache__",
    ".next",
    "dist",
    "build",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "vendor",
    ".cargo",
    "target",
    ".eggs",
    "__pypackages__",
    ".ruff_cache",
})

SKIP_SUFFIXES: frozenset[str] = frozenset({
    ".min.js",
    ".min.css",
    ".map",
    ".lock",
    ".sum",
})

BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".svg",
    ".png",
    ".jpg",
    ".gif",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
})

MAX_FILE_SIZE: int = 100 * 1024  # 100 KB

# ── Test detection ─────────────────────────────────────────────────────────

_TEST_DIR_PATTERN: re.Pattern[str] = re.compile(
    r"(?:^|/)(?:tests?|__tests__|spec)/"
)

_TEST_FILE_PATTERN: re.Pattern[str] = re.compile(
    r"(?:"
    r"^test_[^/]*\.[^/]+$"       # test_foo.py
    r"|^[^/]*_test\.[^/]+$"      # foo_test.go
    r"|^[^/]*\.test\.[^/]+$"     # foo.test.ts
    r"|^[^/]*\.spec\.[^/]+$"     # foo.spec.ts
    r"|^[^/]*_spec\.[^/]+$"      # foo_spec.rb
    r")"
)

# ── Config detection ───────────────────────────────────────────────────────

CONFIG_EXACT_NAMES: frozenset[str] = frozenset({
    "package.json",
    "tsconfig.json",
    "pyproject.toml",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "makefile",
    "cargo.toml",
    "go.mod",
    "go.sum",
    "setup.py",
    "setup.cfg",
    ".gitignore",
})

CONFIG_PREFIX_PATTERNS: tuple[str, ...] = (
    ".env",
    ".eslintrc",
    ".prettierrc",
)

CONFIG_GLOB_PATTERNS: tuple[str, ...] = (
    "*.config.*",
    "jest.config.*",
    "webpack.config.*",
    "vite.config.*",
    "next.config.*",
)

# ── Doc detection ──────────────────────────────────────────────────────────

DOC_EXTENSIONS: frozenset[str] = frozenset({".md", ".mdx", ".rst"})

DOC_PREFIXES: tuple[str, ...] = ("license", "changelog", "contributing")


def _should_skip_dir(dir_name: str) -> bool:
    """Return True if a directory should be entirely skipped."""
    return dir_name in SKIP_DIRS


def _should_skip_file(file_path: Path) -> bool:
    """Return True if a file should be excluded from discovery."""
    name: str = file_path.name

    # Binary extensions
    suffix: str = file_path.suffix.lower()
    if suffix in BINARY_EXTENSIONS:
        return True

    # Compound-suffix checks (e.g. .min.js)
    name_lower: str = name.lower()
    for skip_suffix in SKIP_SUFFIXES:
        if name_lower.endswith(skip_suffix):
            return True

    # File size
    try:
        if file_path.stat().st_size > MAX_FILE_SIZE:
            return True
    except OSError:
        return True

    return False


def _is_under_skipped_dir(rel_path: Path) -> bool:
    """Return True if any component of the relative path is a skipped dir."""
    for part in rel_path.parts[:-1]:  # exclude the filename itself
        if part in SKIP_DIRS:
            return True
    return False


def _classify_file(rel_path_str: str, file_name: str) -> FileType:
    """Classify a file into one of the FileType categories."""
    name_lower: str = file_name.lower()
    suffix: str = Path(file_name).suffix.lower()

    # --- Test ---
    # Check directory-based test detection
    if _TEST_DIR_PATTERN.search(rel_path_str + "/"):
        # Only classify as test if the file's parent path contains test dirs
        pass_test_dir = True
    else:
        pass_test_dir = False

    if pass_test_dir or _TEST_FILE_PATTERN.match(name_lower):
        return "test"

    # --- Config ---
    if name_lower in CONFIG_EXACT_NAMES:
        return "config"

    for prefix in CONFIG_PREFIX_PATTERNS:
        if name_lower.startswith(prefix):
            return "config"

    for pattern in CONFIG_GLOB_PATTERNS:
        # Convert simple glob to check: *.config.* means name contains ".config."
        # jest.config.* means name starts with "jest.config."
        if pattern.startswith("*"):
            # e.g. *.config.* → contains ".config."
            core = pattern.lstrip("*").rstrip("*").rstrip(".")
            # core = ".config"
            if core in name_lower and name_lower.index(core) > 0:
                return "config"
        else:
            # e.g. jest.config.* → starts with "jest.config."
            prefix_part = pattern.rstrip("*").rstrip(".")
            if name_lower.startswith(prefix_part.lower()):
                return "config"

    # --- Doc ---
    if suffix in DOC_EXTENSIONS:
        return "doc"

    for doc_prefix in DOC_PREFIXES:
        if name_lower.startswith(doc_prefix):
            return "doc"

    # --- Code (default) ---
    return "code"


def _load_gitignore(repo_path: Path) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from the repo root, if the file exists."""
    gitignore_path: Path = repo_path / ".gitignore"
    if not gitignore_path.is_file():
        return None
    try:
        text: str = gitignore_path.read_text(encoding="utf-8", errors="replace")
        return pathspec.PathSpec.from_lines("gitwildmatch", text.splitlines())
    except OSError:
        return None


def _count_lines(file_path: Path) -> int:
    """Count lines in a text file, returning 0 on read errors."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def discover_files(
    repo_path: Path,
    language_filter: set[str] | None = None,
) -> list[DiscoveredFile]:
    """Walk *repo_path* and return classified ``DiscoveredFile`` entries.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to scan.
    language_filter:
        If provided, only include *code* and *test* files whose detected
        language is in this set. Config and doc files are always included
        (they are language-agnostic).

    Returns
    -------
    list[DiscoveredFile]
        Sorted list of discovered files.
    """
    repo_path = repo_path.resolve()
    results: list[DiscoveredFile] = []
    gitignore_spec: pathspec.PathSpec | None = _load_gitignore(repo_path)

    for item in repo_path.rglob("*"):
        if not item.is_file():
            continue

        # Compute the path relative to repo root
        rel_path: Path = item.relative_to(repo_path)

        # Skip if any parent dir is in the skip list
        if _is_under_skipped_dir(rel_path):
            continue

        # Respect .gitignore patterns
        if gitignore_spec is not None and gitignore_spec.match_file(str(rel_path)):
            continue

        # Skip unwanted files
        if _should_skip_file(item):
            continue

        rel_path_str: str = str(rel_path)
        file_name: str = item.name
        suffix: str = item.suffix.lower()

        # Classify the file
        file_type: FileType = _classify_file(rel_path_str, file_name)

        # Detect language from extension
        language: str | None = EXTENSION_MAP.get(suffix)

        # Apply language filter — config and doc files are always included
        if language_filter is not None:
            if file_type in ("code", "test"):
                if language is None or language not in language_filter:
                    continue

        # Skip code files whose extension isn't in EXTENSION_MAP
        # (but keep config and doc files regardless)
        if file_type == "code" and language is None:
            # Only include code files that have a recognized language
            # unless the extension is unknown — still skip
            continue

        # Compute metadata
        size_bytes: int = item.stat().st_size
        line_count: int = _count_lines(item)
        content_hash: str = compute_file_hash(item)

        results.append(
            DiscoveredFile(
                path=rel_path_str,
                absolute_path=str(item),
                language=language,
                file_type=file_type,
                size_bytes=size_bytes,
                line_count=line_count,
                content_hash=content_hash,
            )
        )

    return results
