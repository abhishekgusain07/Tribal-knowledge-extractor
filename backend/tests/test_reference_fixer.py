"""Tests for tribal_knowledge.refresh.reference_fixer — broken reference repair."""

from __future__ import annotations

from pathlib import Path

import pytest

from tribal_knowledge.refresh.reference_fixer import (
    _collect_current_files,
    fix_broken_references,
)


def _write_file(path: Path, content: str) -> None:
    """Write *content* to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# fix_broken_references
# ---------------------------------------------------------------------------


class TestFixBrokenReferences:
    """Tests for fix_broken_references."""

    def test_broken_ref_exact_filename_match_fixed(self, tmp_path: Path) -> None:
        """Broken ref whose basename exists elsewhere -> replaced with new path."""
        context = tmp_path / "context"
        context.mkdir()

        # Doc references an old path
        doc = "Check `src/old/util.py` for details."
        (context / "module.md").write_text(doc, encoding="utf-8")

        broken_refs = ["src/old/util.py"]
        current_files = {"src/new/util.py", "src/main.py"}

        changes = fix_broken_references(context, broken_refs, current_files)

        assert "module.md" in changes
        old_ref, new_ref = changes["module.md"][0]
        assert old_ref == "src/old/util.py"
        assert new_ref == "src/new/util.py"

        # Verify the file was actually updated
        updated_text = (context / "module.md").read_text(encoding="utf-8")
        assert "src/new/util.py" in updated_text
        assert "src/old/util.py" not in updated_text

    def test_broken_ref_no_match_marked_removed(self, tmp_path: Path) -> None:
        """Broken ref with no matching basename -> marked as REMOVED."""
        context = tmp_path / "context"
        context.mkdir()

        doc = "See `src/deleted/completely_gone.py` for the old code."
        (context / "module.md").write_text(doc, encoding="utf-8")

        broken_refs = ["src/deleted/completely_gone.py"]
        current_files = {"src/main.py", "src/util.py"}

        changes = fix_broken_references(context, broken_refs, current_files)

        assert "module.md" in changes
        old_ref, new_ref = changes["module.md"][0]
        assert old_ref == "src/deleted/completely_gone.py"
        assert new_ref == "REMOVED"

    def test_no_broken_refs_no_changes(self, tmp_path: Path) -> None:
        """Empty broken_refs list -> no changes made."""
        context = tmp_path / "context"
        context.mkdir()

        doc = "All references are fine: `src/main.py`."
        (context / "module.md").write_text(doc, encoding="utf-8")

        changes = fix_broken_references(context, [], {"src/main.py"})

        assert changes == {}

    def test_multiple_broken_refs_in_one_doc(self, tmp_path: Path) -> None:
        """Multiple broken refs in one doc -> all get fixed."""
        context = tmp_path / "context"
        context.mkdir()

        doc = (
            "See `src/old/login.py` for auth.\n"
            "And `src/old/token.py` for tokens.\n"
            "Also `src/vanished/nope.py` which is gone.\n"
        )
        (context / "auth.md").write_text(doc, encoding="utf-8")

        broken_refs = [
            "src/old/login.py",
            "src/old/token.py",
            "src/vanished/nope.py",
        ]
        current_files = {
            "src/auth/login.py",
            "src/auth/token.py",
            "src/main.py",
        }

        changes = fix_broken_references(context, broken_refs, current_files)

        assert "auth.md" in changes
        change_map = dict(changes["auth.md"])
        assert change_map["src/old/login.py"] == "src/auth/login.py"
        assert change_map["src/old/token.py"] == "src/auth/token.py"
        assert change_map["src/vanished/nope.py"] == "REMOVED"

    def test_broken_ref_not_present_in_doc(self, tmp_path: Path) -> None:
        """Broken ref that doesn't appear in any doc -> no changes."""
        context = tmp_path / "context"
        context.mkdir()

        doc = "This doc only mentions `src/main.py`."
        (context / "module.md").write_text(doc, encoding="utf-8")

        broken_refs = ["src/other/file.py"]
        current_files = {"src/main.py"}

        changes = fix_broken_references(context, broken_refs, current_files)

        # The broken ref is not in the doc text, so no changes
        assert changes == {}

    def test_nonexistent_context_dir(self, tmp_path: Path) -> None:
        """Non-existent context_dir -> returns empty dict."""
        context = tmp_path / "nonexistent"

        changes = fix_broken_references(
            context,
            ["src/old.py"],
            {"src/new.py"},
        )

        assert changes == {}

    def test_multiple_docs_with_same_broken_ref(self, tmp_path: Path) -> None:
        """Same broken ref in multiple docs -> all get fixed."""
        context = tmp_path / "context"
        context.mkdir()

        doc1 = "See `src/old/util.py` here."
        doc2 = "Also `src/old/util.py` referenced here."
        (context / "mod1.md").write_text(doc1, encoding="utf-8")
        (context / "mod2.md").write_text(doc2, encoding="utf-8")

        broken_refs = ["src/old/util.py"]
        current_files = {"src/new/util.py"}

        changes = fix_broken_references(context, broken_refs, current_files)

        assert "mod1.md" in changes
        assert "mod2.md" in changes


# ---------------------------------------------------------------------------
# _collect_current_files
# ---------------------------------------------------------------------------


class TestCollectCurrentFiles:
    """Tests for _collect_current_files."""

    def test_returns_correct_relative_paths(self, tmp_path: Path) -> None:
        """Returns relative paths for all files under repo."""
        repo = tmp_path / "repo"
        _write_file(repo / "src" / "main.py", "print('hello')")
        _write_file(repo / "src" / "util.py", "def util(): pass")
        _write_file(repo / "README.md", "# readme")

        paths = _collect_current_files(repo)

        assert "src/main.py" in paths
        assert "src/util.py" in paths
        assert "README.md" in paths

    def test_skips_git_directory(self, tmp_path: Path) -> None:
        """Files under .git should be skipped."""
        repo = tmp_path / "repo"
        _write_file(repo / "src" / "main.py", "print('hello')")
        _write_file(repo / ".git" / "config", "git config content")

        paths = _collect_current_files(repo)

        assert "src/main.py" in paths
        git_paths = [p for p in paths if ".git" in p]
        assert git_paths == []

    def test_skips_pycache_directory(self, tmp_path: Path) -> None:
        """Files under __pycache__ should be skipped."""
        repo = tmp_path / "repo"
        _write_file(repo / "src" / "main.py", "print('hello')")
        _write_file(repo / "src" / "__pycache__" / "main.cpython-312.pyc", "bytecode")

        paths = _collect_current_files(repo)

        assert "src/main.py" in paths
        pycache_paths = [p for p in paths if "__pycache__" in p]
        assert pycache_paths == []

    def test_skips_node_modules(self, tmp_path: Path) -> None:
        """Files under node_modules should be skipped."""
        repo = tmp_path / "repo"
        _write_file(repo / "index.js", "module.exports = {}")
        _write_file(repo / "node_modules" / "pkg" / "index.js", "dep code")

        paths = _collect_current_files(repo)

        assert "index.js" in paths
        nm_paths = [p for p in paths if "node_modules" in p]
        assert nm_paths == []

    def test_empty_repo(self, tmp_path: Path) -> None:
        """Empty repo directory -> returns empty set."""
        repo = tmp_path / "repo"
        repo.mkdir()

        paths = _collect_current_files(repo)

        assert paths == set()

    def test_nested_directory_structure(self, tmp_path: Path) -> None:
        """Deeply nested files have correct relative paths."""
        repo = tmp_path / "repo"
        _write_file(repo / "a" / "b" / "c" / "deep.py", "deep code")

        paths = _collect_current_files(repo)

        assert "a/b/c/deep.py" in paths
