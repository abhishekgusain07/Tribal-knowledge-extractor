"""Fix broken file-path references in generated Markdown docs."""

from __future__ import annotations

from pathlib import Path

# Same skip set as discovery.py / detector.py.
SKIP_DIRS: frozenset[str] = frozenset(
    {
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
    }
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_under_skipped_dir(rel_path: Path) -> bool:
    """Return True if any component of *rel_path* (excluding filename) is skipped."""
    for part in rel_path.parts[:-1]:
        if part in SKIP_DIRS:
            return True
    return False


def _collect_current_files(repo_path: Path) -> set[str]:
    """Walk *repo_path* and return all relative file paths (respecting SKIP_DIRS)."""
    repo_path = repo_path.resolve()
    paths: set[str] = set()

    for item in repo_path.rglob("*"):
        if not item.is_file():
            continue

        rel_path = item.relative_to(repo_path)

        if _is_under_skipped_dir(rel_path):
            continue

        paths.add(str(rel_path))

    return paths


def _find_replacement(
    broken_ref: str,
    current_file_paths: set[str],
) -> str:
    """Find the best replacement for *broken_ref* among *current_file_paths*.

    Strategy:
    1. Exact basename match -- if a file with the same name exists, use it.
    2. Among multiple matches pick the one with the shortest path.
    3. If nothing matches return ``""`` (the reference should be removed).
    """
    target_basename = Path(broken_ref).name

    candidates: list[str] = [
        p for p in current_file_paths if Path(p).name == target_basename
    ]

    if not candidates:
        return ""

    # Pick shortest path (most specific / least nested).
    candidates.sort(key=len)
    return candidates[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fix_broken_references(
    context_dir: Path,
    broken_refs: list[str],
    current_file_paths: set[str],
) -> dict[str, list[tuple[str, str]]]:
    """Replace broken backtick file-path references in all ``.md`` files under *context_dir*.

    Parameters
    ----------
    context_dir:
        Directory containing generated Markdown docs.
    broken_refs:
        List of file-path strings that no longer resolve in the repo.
    current_file_paths:
        Set of relative paths that *do* currently exist.

    Returns
    -------
    dict[str, list[tuple[str, str]]]
        Mapping of ``{doc_filename: [(old_ref, new_ref_or_"REMOVED"), ...]}``.
    """
    if not broken_refs:
        return {}

    # Pre-compute replacements for every broken reference.
    replacement_map: dict[str, str] = {}
    for ref in broken_refs:
        replacement_map[ref] = _find_replacement(ref, current_file_paths)

    changes: dict[str, list[tuple[str, str]]] = {}

    if not context_dir.is_dir():
        return changes

    for md_file in context_dir.rglob("*.md"):
        try:
            original_text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        updated_text = original_text
        file_changes: list[tuple[str, str]] = []

        for old_ref, new_ref in replacement_map.items():
            if old_ref not in updated_text:
                continue

            if new_ref:
                updated_text = updated_text.replace(old_ref, new_ref)
                file_changes.append((old_ref, new_ref))
            else:
                # Remove the reference (replace with empty string).
                updated_text = updated_text.replace(old_ref, "")
                file_changes.append((old_ref, "REMOVED"))

        if file_changes:
            try:
                md_file.write_text(updated_text, encoding="utf-8")
            except OSError:
                continue
            changes[md_file.name] = file_changes

    return changes
