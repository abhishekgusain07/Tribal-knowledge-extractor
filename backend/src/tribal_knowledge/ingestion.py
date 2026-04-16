"""Repo ingestion: validate local repos or clone from GitHub, extract git metadata."""

from __future__ import annotations

import re
from pathlib import Path

from git import InvalidGitRepositoryError, Repo

from tribal_knowledge.models import EXTENSION_MAP, RepoInfo, RepoSource

CACHE_DIR: Path = Path.home() / ".tribal-knowledge" / "cache"

# Patterns for GitHub URL parsing
_HTTPS_PATTERN: re.Pattern[str] = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?/?$"
)
_SSH_PATTERN: re.Pattern[str] = re.compile(
    r"git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$"
)


def _parse_github_url(url: str) -> tuple[str, str]:
    """Extract (owner, repo) from a GitHub URL.

    Raises ValueError if the URL doesn't match any known GitHub format.
    """
    for pattern in (_HTTPS_PATTERN, _SSH_PATTERN):
        match = pattern.match(url)
        if match:
            return match.group("owner"), match.group("repo")
    raise ValueError(f"Could not parse GitHub owner/repo from URL: {url}")


def _is_git_repo(path: Path) -> bool:
    """Check whether *path* is the root of a valid git repository."""
    try:
        Repo(path)
        return True
    except InvalidGitRepositoryError:
        return False


def _clone_or_pull(url: str, dest: Path, branch: str | None) -> Repo:
    """Clone a repo to *dest*, or pull if it already exists and is a git repo.

    Uses a full clone (no --depth) so that git history is available for
    churn analysis.  If the repo is already cached, pulls latest instead.
    """
    if dest.exists() and _is_git_repo(dest):
        repo = Repo(dest)
        # Pull latest changes
        origin = repo.remotes.origin
        if branch:
            origin.pull(branch)
        else:
            origin.pull()
        return repo

    dest.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, str] = {}
    if branch:
        kwargs["branch"] = branch
    return Repo.clone_from(url, str(dest), **kwargs)


def _extract_git_metadata(repo: Repo) -> tuple[str | None, dict[str, int]]:
    """Return (head_commit_sha, file_change_frequency) from the last 200 commits."""
    head_commit: str | None = None
    file_change_frequency: dict[str, int] = {}

    try:
        head_commit = repo.head.commit.hexsha
    except ValueError:
        # Empty repo, no commits
        return None, {}

    for commit in repo.iter_commits(max_count=200):
        for file_path in commit.stats.files:
            file_change_frequency[file_path] = file_change_frequency.get(file_path, 0) + 1

    return head_commit, file_change_frequency


def _count_files_and_languages(local_path: Path) -> tuple[int, dict[str, int]]:
    """Walk a directory and return (total_file_count, {language: file_count})."""
    total_files: int = 0
    languages: dict[str, int] = {}

    for item in local_path.rglob("*"):
        if not item.is_file():
            continue
        total_files += 1
        ext = item.suffix.lower()
        lang = EXTENSION_MAP.get(ext)
        if lang is not None:
            languages[lang] = languages.get(lang, 0) + 1

    return total_files, languages


def ingest(source: RepoSource) -> RepoInfo:
    """Ingest a repository from a local path or GitHub URL.

    Parameters
    ----------
    source:
        A ``RepoSource`` containing either a local filesystem path or a
        GitHub URL (HTTPS or SSH).

    Returns
    -------
    RepoInfo
        Metadata about the ingested repository.
    """
    path_or_url: str = source.path_or_url.strip()
    is_remote: bool = False
    owner: str | None = None
    repo_name: str = ""
    local_path: Path
    default_branch: str | None = source.branch

    # Determine if this is a URL or a local path
    if path_or_url.startswith(("https://", "http://", "git@")):
        is_remote = True
        owner, repo_name = _parse_github_url(path_or_url)
        local_path = CACHE_DIR / f"{owner}_{repo_name}"
        repo = _clone_or_pull(path_or_url, local_path, source.branch)

        # Detect default branch from remote HEAD if not explicitly provided
        if default_branch is None:
            try:
                default_branch = repo.active_branch.name
            except TypeError:
                default_branch = None
    else:
        local_path = Path(path_or_url).resolve()
        if not local_path.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {local_path}")
        repo_name = local_path.name

    # Extract git metadata if applicable
    head_commit: str | None = None
    file_change_frequency: dict[str, int] = {}

    if _is_git_repo(local_path):
        repo_obj = Repo(local_path)
        head_commit, file_change_frequency = _extract_git_metadata(repo_obj)
        if default_branch is None:
            try:
                default_branch = repo_obj.active_branch.name
            except TypeError:
                default_branch = None

    total_files, languages_detected = _count_files_and_languages(local_path)

    return RepoInfo(
        local_path=local_path,
        name=repo_name,
        owner=owner,
        is_remote=is_remote,
        default_branch=default_branch,
        total_files=total_files,
        languages_detected=languages_detected,
        head_commit=head_commit,
        file_change_frequency=file_change_frequency,
    )
