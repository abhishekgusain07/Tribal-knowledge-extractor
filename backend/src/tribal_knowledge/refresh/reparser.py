"""Selective re-parse orchestration for the self-refresh system."""

from __future__ import annotations

import json
from pathlib import Path

from tribal_knowledge.discovery import discover_files
from tribal_knowledge.models import FileExtraction, ParseManifest
from tribal_knowledge.refresh.detector import StalenessReport
from tribal_knowledge.refresh.reference_fixer import (
    _collect_current_files,
    fix_broken_references,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_extractions(output_dir: Path) -> list[FileExtraction]:
    """Load existing file extractions from the ``.tribal-knowledge`` directory."""
    extractions_path = output_dir / ".tribal-knowledge" / "extractions.json"

    if not extractions_path.exists():
        return []

    raw = extractions_path.read_text(encoding="utf-8")
    data: list[dict[str, object]] = json.loads(raw)
    return [FileExtraction.model_validate(item) for item in data]


def _save_extractions(extractions: list[FileExtraction], output_dir: Path) -> None:
    """Save file extractions to the ``.tribal-knowledge`` directory."""
    tk_dir = output_dir / ".tribal-knowledge"
    tk_dir.mkdir(parents=True, exist_ok=True)

    data = [ext.model_dump() for ext in extractions]
    (tk_dir / "extractions.json").write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )


def _parse_all_files(files: list[object]) -> list[FileExtraction]:
    """Parse ASTs for discovered files, mirroring the CLI's ``_parse_all_files``.

    We import the parser dynamically so this module stays lightweight when
    the parser isn't needed.
    """
    from tribal_knowledge.models import DiscoveredFile

    extractions: list[FileExtraction] = []

    for discovered in files:
        if not isinstance(discovered, DiscoveredFile):
            continue

        if discovered.language is None:
            continue

        if discovered.file_type not in ("code", "test"):
            continue

        try:
            from tribal_knowledge.parser import get_parser_for_language

            parser = get_parser_for_language(discovered.language)
        except (ValueError, ImportError):
            continue

        try:
            source = Path(discovered.absolute_path).read_bytes()
            extraction = parser.parse_file(discovered.path, source)
            extraction.file_type = discovered.file_type
            extractions.append(extraction)
        except Exception:  # noqa: BLE001
            continue

    return extractions


# ---------------------------------------------------------------------------
# Dispatch strategies
# ---------------------------------------------------------------------------


def _full_reparse(repo_path: Path, output_dir: Path) -> None:
    """Perform a complete re-parse of the repository.

    This delegates to the main CLI ``parse`` workflow. For now we print a
    message indicating that a full reparse is needed and invoke the core
    parse helpers directly.
    """
    print(
        f"[refresh] Full reparse required for {repo_path}. "
        "Re-running full parse pipeline..."
    )

    files = discover_files(repo_path)
    extractions = _parse_all_files(files)
    _save_extractions(extractions, output_dir)

    print(
        f"[refresh] Full reparse complete: "
        f"{len(files)} files discovered, {len(extractions)} parsed."
    )


def _partial_reparse(
    repo_path: Path,
    output_dir: Path,
    staleness: StalenessReport,
) -> None:
    """Re-parse only the stale and new files, then merge with existing extractions."""
    print("[refresh] Partial reparse: re-parsing stale and new files...")

    # 1. Load existing extractions
    existing = _load_extractions(output_dir)

    # Build a lookup by file path for easy merging
    extraction_map: dict[str, FileExtraction] = {
        ext.file_path: ext for ext in existing
    }

    # 2. Determine which relative paths need re-parsing
    targets: set[str] = set(staleness.stale_files) | set(staleness.new_files)

    # 3. Re-discover only the target files
    all_discovered = discover_files(repo_path)
    filtered = [f for f in all_discovered if f.path in targets]

    # 4. Re-parse them
    new_extractions = _parse_all_files(filtered)

    # 5. Merge
    for ext in new_extractions:
        extraction_map[ext.file_path] = ext

    # Remove deleted files
    for deleted_path in staleness.deleted_files:
        extraction_map.pop(deleted_path, None)

    merged = list(extraction_map.values())

    # 6. Save
    _save_extractions(merged, output_dir)

    print(
        f"[refresh] Partial reparse complete: "
        f"{len(new_extractions)} files re-parsed, "
        f"{len(staleness.deleted_files)} removed, "
        f"{len(merged)} total extractions."
    )


def _doc_refresh(
    repo_path: Path,
    output_dir: Path,
    staleness: StalenessReport,
) -> None:
    """Fix broken file references in generated docs without re-parsing code."""
    print("[refresh] Doc refresh: fixing broken references...")

    context_dir = output_dir / "context"
    current_files = _collect_current_files(repo_path)

    changes = fix_broken_references(
        context_dir=context_dir,
        broken_refs=staleness.broken_references,
        current_file_paths=current_files,
    )

    total_fixes = sum(len(fixes) for fixes in changes.values())

    print(
        f"[refresh] Doc refresh complete: "
        f"{total_fixes} reference(s) fixed across {len(changes)} file(s)."
    )

    for doc_name, fixes in changes.items():
        for old_ref, new_ref in fixes:
            label = new_ref if new_ref != "REMOVED" else "(removed)"
            print(f"  {doc_name}: {old_ref} -> {label}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def selective_reparse(
    repo_path: Path,
    output_dir: Path,
    staleness: StalenessReport,
) -> None:
    """Orchestrate a refresh based on the *staleness* report.

    - Full reparse when >50 % of files are stale.
    - Partial reparse when some files changed.
    - Doc-only refresh when only Markdown references are broken.
    """
    if staleness.needs_full_reparse:
        _full_reparse(repo_path, output_dir)
    elif staleness.needs_partial_reparse:
        _partial_reparse(repo_path, output_dir, staleness)
    elif staleness.needs_doc_refresh_only:
        _doc_refresh(repo_path, output_dir, staleness)
    else:
        print("[refresh] Everything is up to date -- no action needed.")
