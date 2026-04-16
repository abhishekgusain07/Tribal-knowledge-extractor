"""Staleness detection -- compare current repo state against saved manifest."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from tribal_knowledge.models import ParseManifest, compute_file_hash

# Same skip set used by discovery.py (minus the extras that discovery adds).
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

# Regex to extract backtick-quoted file paths from Markdown.
# Matches content inside single backticks that looks like a file path
# (contains at least one '/' or a dot-extension).
_BACKTICK_PATH_RE: re.Pattern[str] = re.compile(r"`([^`\n]+\.[a-zA-Z0-9]+)`")


# ---------------------------------------------------------------------------
# Staleness report
# ---------------------------------------------------------------------------


class StalenessReport(BaseModel):
    """Result of comparing current repo state against a saved manifest."""

    stale_files: list[str] = Field(default_factory=list)
    new_files: list[str] = Field(default_factory=list)
    deleted_files: list[str] = Field(default_factory=list)
    stale_modules: list[str] = Field(default_factory=list)
    broken_references: list[str] = Field(default_factory=list)
    commit_distance: int = 0
    needs_full_reparse: bool = False
    needs_partial_reparse: bool = False
    needs_doc_refresh_only: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_under_skipped_dir(rel_path: Path) -> bool:
    """Return True if any component of *rel_path* (excluding filename) is skipped."""
    for part in rel_path.parts[:-1]:
        if part in SKIP_DIRS:
            return True
    return False


def _walk_code_files(repo_path: Path) -> dict[str, str]:
    """Walk *repo_path* and return ``{relative_path: sha256_hash}`` for code files."""
    repo_path = repo_path.resolve()
    hashes: dict[str, str] = {}

    for item in repo_path.rglob("*"):
        if not item.is_file():
            continue

        rel_path = item.relative_to(repo_path)

        if _is_under_skipped_dir(rel_path):
            continue

        # Skip files without recognised code extensions
        # (We hash *all* non-skipped files so the manifest comparison is complete.)
        try:
            file_hash = compute_file_hash(item)
        except OSError:
            continue

        hashes[str(rel_path)] = file_hash

    return hashes


def _compute_commit_distance(repo_path: Path, base_commit: str | None) -> int:
    """Return the number of commits between *base_commit* and HEAD.

    Returns 0 on any error (missing git, not a repo, bad commit, etc.).
    """
    if base_commit is None:
        return 0

    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", f"{base_commit}..HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_path),
            timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        pass

    return 0


def _scan_doc_references(context_dir: Path) -> list[str]:
    """Extract all backtick-quoted file paths from every ``.md`` file in *context_dir*."""
    refs: list[str] = []

    if not context_dir.is_dir():
        return refs

    for md_file in context_dir.rglob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for match in _BACKTICK_PATH_RE.finditer(text):
            candidate = match.group(1)
            # Only keep entries that look like relative file paths
            # (contain a slash or start with a recognisable directory).
            if "/" in candidate or candidate.startswith("."):
                refs.append(candidate)

    return refs


def _check_references(refs: list[str], repo_path: Path) -> list[str]:
    """Return the subset of *refs* that do **not** exist under *repo_path*."""
    repo_path = repo_path.resolve()
    broken: list[str] = []

    for ref in refs:
        candidate = repo_path / ref
        if not candidate.exists():
            broken.append(ref)

    return broken


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_staleness(
    repo_path: Path,
    manifest: ParseManifest,
    context_dir: Path,
) -> StalenessReport:
    """Compare the current repo state against *manifest* and return a report.

    Steps
    -----
    1. Walk repo, compute SHA-256 hashes for all non-skipped files.
    2. Compare against ``manifest.file_hashes``:
       - hash differs -> stale
       - in current but not manifest -> new
       - in manifest but not current -> deleted
    3. Compute commit distance via ``git rev-list --count``.
    4. Identify stale modules (>30 % of files changed).
    5. Scan generated docs for backtick path refs; flag broken ones.
    6. Decision:
       - >50 % stale -> ``needs_full_reparse``
       - any stale -> ``needs_partial_reparse``
       - only broken refs -> ``needs_doc_refresh_only``
    """
    report = StalenessReport()

    # 1 & 2 -- hash comparison
    current_hashes = _walk_code_files(repo_path)
    manifest_files = set(manifest.file_hashes.keys())
    current_files = set(current_hashes.keys())

    for rel_path in current_files & manifest_files:
        if current_hashes[rel_path] != manifest.file_hashes[rel_path]:
            report.stale_files.append(rel_path)

    report.new_files = sorted(current_files - manifest_files)
    report.deleted_files = sorted(manifest_files - current_files)

    # 3 -- commit distance
    report.commit_distance = _compute_commit_distance(repo_path, manifest.head_commit)

    # 4 -- stale modules (>30 % of module's files are stale/new/deleted)
    stale_set: set[str] = set(report.stale_files) | set(report.new_files) | set(report.deleted_files)

    module_file_counts: dict[str, int] = {}
    module_stale_counts: dict[str, int] = {}

    for file_path, module_name in manifest.module_assignments.items():
        module_file_counts[module_name] = module_file_counts.get(module_name, 0) + 1
        if file_path in stale_set:
            module_stale_counts[module_name] = module_stale_counts.get(module_name, 0) + 1

    for module_name, total in module_file_counts.items():
        stale_count = module_stale_counts.get(module_name, 0)
        if total > 0 and stale_count / total > 0.30:
            report.stale_modules.append(module_name)

    # 5 -- broken references in generated docs
    doc_refs = _scan_doc_references(context_dir)
    report.broken_references = _check_references(doc_refs, repo_path)

    # 6 -- decision
    total_manifest_files = len(manifest.file_hashes) or 1  # avoid division by zero
    stale_ratio = len(report.stale_files) / total_manifest_files

    if stale_ratio > 0.50:
        report.needs_full_reparse = True
    elif report.stale_files or report.new_files or report.deleted_files:
        report.needs_partial_reparse = True
    elif report.broken_references:
        report.needs_doc_refresh_only = True

    return report
