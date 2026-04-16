"""Tests for tribal_knowledge.refresh.detector — staleness detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from tribal_knowledge.models import ParseManifest, compute_file_hash
from tribal_knowledge.refresh.detector import StalenessReport, detect_staleness


def _write_file(path: Path, content: str) -> str:
    """Write *content* to *path* and return its SHA-256 hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return compute_file_hash(path)


def _make_manifest(
    file_hashes: dict[str, str],
    module_assignments: dict[str, str] | None = None,
    head_commit: str | None = None,
) -> ParseManifest:
    """Build a ParseManifest with the given hashes and optional module map."""
    return ParseManifest(
        generated_at="2025-01-01T00:00:00",
        head_commit=head_commit,
        repo_name="test-repo",
        total_files_parsed=len(file_hashes),
        file_hashes=file_hashes,
        module_assignments=module_assignments or {},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectStaleness:
    """Tests for detect_staleness."""

    def test_no_changes(self, tmp_path: Path) -> None:
        """Matching hashes -> all fields empty/false."""
        repo = tmp_path / "repo"
        context = tmp_path / "context"
        context.mkdir()

        h1 = _write_file(repo / "src" / "main.py", "print('hello')")
        h2 = _write_file(repo / "src" / "util.py", "def util(): pass")

        manifest = _make_manifest({
            "src/main.py": h1,
            "src/util.py": h2,
        })

        report = detect_staleness(repo, manifest, context)

        assert report.stale_files == []
        assert report.new_files == []
        assert report.deleted_files == []
        assert report.stale_modules == []
        assert report.broken_references == []
        assert report.needs_full_reparse is False
        assert report.needs_partial_reparse is False
        assert report.needs_doc_refresh_only is False

    def test_one_file_changed(self, tmp_path: Path) -> None:
        """Changing content of one file -> stale_files contains it."""
        repo = tmp_path / "repo"
        context = tmp_path / "context"
        context.mkdir()

        h1 = _write_file(repo / "src" / "main.py", "print('hello')")
        h2 = _write_file(repo / "src" / "util.py", "def util(): pass")

        manifest = _make_manifest({
            "src/main.py": h1,
            "src/util.py": h2,
        })

        # Now change main.py
        (repo / "src" / "main.py").write_text("print('changed')", encoding="utf-8")

        report = detect_staleness(repo, manifest, context)

        assert "src/main.py" in report.stale_files
        assert "src/util.py" not in report.stale_files
        assert report.needs_partial_reparse is True

    def test_new_file_added(self, tmp_path: Path) -> None:
        """Adding a file not in the manifest -> new_files contains it."""
        repo = tmp_path / "repo"
        context = tmp_path / "context"
        context.mkdir()

        h1 = _write_file(repo / "src" / "main.py", "print('hello')")

        manifest = _make_manifest({"src/main.py": h1})

        # Add a new file
        _write_file(repo / "src" / "new_module.py", "# brand new")

        report = detect_staleness(repo, manifest, context)

        assert "src/new_module.py" in report.new_files
        assert report.needs_partial_reparse is True

    def test_file_deleted(self, tmp_path: Path) -> None:
        """Manifest references a file that does not exist -> deleted_files."""
        repo = tmp_path / "repo"
        context = tmp_path / "context"
        context.mkdir()

        h1 = _write_file(repo / "src" / "main.py", "print('hello')")

        manifest = _make_manifest({
            "src/main.py": h1,
            "src/removed.py": "fakehash123",
        })

        report = detect_staleness(repo, manifest, context)

        assert "src/removed.py" in report.deleted_files
        assert report.needs_partial_reparse is True

    def test_over_50_percent_stale_triggers_full_reparse(self, tmp_path: Path) -> None:
        """>50% of manifest files changed -> needs_full_reparse = True."""
        repo = tmp_path / "repo"
        context = tmp_path / "context"
        context.mkdir()

        # Create 4 files with known hashes
        hashes: dict[str, str] = {}
        for i in range(4):
            fname = f"src/file_{i}.py"
            hashes[fname] = _write_file(repo / fname, f"content_{i}")

        manifest = _make_manifest(hashes)

        # Now change 3 of 4 files (75% > 50%)
        for i in range(3):
            (repo / f"src/file_{i}.py").write_text(
                f"changed_content_{i}", encoding="utf-8"
            )

        report = detect_staleness(repo, manifest, context)

        assert report.needs_full_reparse is True
        assert len(report.stale_files) == 3

    def test_under_50_percent_triggers_partial_reparse(self, tmp_path: Path) -> None:
        """Some changed but <50% -> needs_partial_reparse = True."""
        repo = tmp_path / "repo"
        context = tmp_path / "context"
        context.mkdir()

        hashes: dict[str, str] = {}
        for i in range(4):
            fname = f"src/file_{i}.py"
            hashes[fname] = _write_file(repo / fname, f"content_{i}")

        manifest = _make_manifest(hashes)

        # Change 1 of 4 files (25% < 50%)
        (repo / "src/file_0.py").write_text("changed", encoding="utf-8")

        report = detect_staleness(repo, manifest, context)

        assert report.needs_partial_reparse is True
        assert report.needs_full_reparse is False

    def test_module_staleness_over_30_percent(self, tmp_path: Path) -> None:
        """>30% of a module's files stale -> stale_modules contains it."""
        repo = tmp_path / "repo"
        context = tmp_path / "context"
        context.mkdir()

        # auth module: 3 files, change 2 -> 66% > 30%
        h_auth1 = _write_file(repo / "src" / "auth" / "login.py", "login code")
        h_auth2 = _write_file(repo / "src" / "auth" / "token.py", "token code")
        h_auth3 = _write_file(repo / "src" / "auth" / "models.py", "models code")

        # db module: 2 files, change 0
        h_db1 = _write_file(repo / "src" / "db" / "pool.py", "pool code")
        h_db2 = _write_file(repo / "src" / "db" / "query.py", "query code")

        manifest = _make_manifest(
            file_hashes={
                "src/auth/login.py": h_auth1,
                "src/auth/token.py": h_auth2,
                "src/auth/models.py": h_auth3,
                "src/db/pool.py": h_db1,
                "src/db/query.py": h_db2,
            },
            module_assignments={
                "src/auth/login.py": "auth",
                "src/auth/token.py": "auth",
                "src/auth/models.py": "auth",
                "src/db/pool.py": "db",
                "src/db/query.py": "db",
            },
        )

        # Change 2 of 3 auth files
        (repo / "src" / "auth" / "login.py").write_text("changed login", encoding="utf-8")
        (repo / "src" / "auth" / "token.py").write_text("changed token", encoding="utf-8")

        report = detect_staleness(repo, manifest, context)

        assert "auth" in report.stale_modules
        assert "db" not in report.stale_modules

    def test_broken_references_detected(self, tmp_path: Path) -> None:
        """Markdown doc referencing nonexistent path -> broken_references."""
        repo = tmp_path / "repo"
        context = tmp_path / "context"
        context.mkdir()

        # Create a file in the repo
        h1 = _write_file(repo / "src" / "main.py", "print('hello')")

        manifest = _make_manifest({"src/main.py": h1})

        # Create a markdown doc that references a non-existent path
        doc_content = "Check `src/deleted/old_module.py` for the implementation."
        (context / "auth.md").write_text(doc_content, encoding="utf-8")

        report = detect_staleness(repo, manifest, context)

        assert "src/deleted/old_module.py" in report.broken_references

    def test_needs_doc_refresh_only(self, tmp_path: Path) -> None:
        """No code changes but broken refs -> needs_doc_refresh_only = True."""
        repo = tmp_path / "repo"
        context = tmp_path / "context"
        context.mkdir()

        h1 = _write_file(repo / "src" / "main.py", "print('hello')")

        manifest = _make_manifest({"src/main.py": h1})

        # Create a markdown doc that references a non-existent path
        doc_content = "See `src/gone/deleted.py` for deprecated code."
        (context / "module.md").write_text(doc_content, encoding="utf-8")

        report = detect_staleness(repo, manifest, context)

        assert report.needs_doc_refresh_only is True
        assert report.needs_full_reparse is False
        assert report.needs_partial_reparse is False
        assert len(report.broken_references) >= 1

    def test_empty_manifest_all_files_new(self, tmp_path: Path) -> None:
        """Empty manifest with existing files -> all files are new."""
        repo = tmp_path / "repo"
        context = tmp_path / "context"
        context.mkdir()

        _write_file(repo / "src" / "main.py", "print('hello')")
        _write_file(repo / "src" / "util.py", "def util(): pass")

        manifest = _make_manifest({})

        report = detect_staleness(repo, manifest, context)

        assert len(report.new_files) >= 2
        assert report.stale_files == []
        assert report.deleted_files == []

    def test_staleness_report_model_defaults(self) -> None:
        """StalenessReport has correct default values."""
        report = StalenessReport()

        assert report.stale_files == []
        assert report.new_files == []
        assert report.deleted_files == []
        assert report.stale_modules == []
        assert report.broken_references == []
        assert report.commit_distance == 0
        assert report.needs_full_reparse is False
        assert report.needs_partial_reparse is False
        assert report.needs_doc_refresh_only is False
