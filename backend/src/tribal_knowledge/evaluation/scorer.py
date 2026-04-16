"""Automated scoring functions for generated context documents.

All scorers are pure computation -- no LLM calls.  Each returns a numeric
score in [0.0, 1.0] and, where appropriate, a list of diagnostic strings.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# 1. Entity coverage
# ---------------------------------------------------------------------------


def entity_coverage_score(
    doc_text: str,
    top_entities: list[str],
) -> tuple[float, list[str]]:
    """Score how many of *top_entities* appear in *doc_text* (case-insensitive).

    Returns
    -------
    tuple[float, list[str]]
        (fraction_covered, missing_entity_names)
    """
    if not top_entities:
        return 1.0, []

    doc_lower = doc_text.lower()
    missing: list[str] = []

    for entity in top_entities:
        if entity.lower() not in doc_lower:
            missing.append(entity)

    covered = len(top_entities) - len(missing)
    score = covered / len(top_entities)
    return score, missing


# ---------------------------------------------------------------------------
# 2. Factual accuracy (reference validation)
# ---------------------------------------------------------------------------

_BACKTICK_RE = re.compile(r"`([^`]+)`")

_CODE_FILE_EXTENSIONS = frozenset({".py", ".ts", ".js", ".go", ".rs", ".java"})


def _looks_like_file_path(ref: str) -> bool:
    """Heuristic: does *ref* look like a file path rather than an entity name?"""
    if "/" in ref:
        return True
    for ext in _CODE_FILE_EXTENSIONS:
        if ref.endswith(ext):
            return True
    return False


def factual_accuracy_score(
    doc_text: str,
    entity_names: set[str],
    file_paths: set[str],
) -> tuple[float, list[str]]:
    """Validate backtick-quoted references in *doc_text*.

    Extracts every ``\\`reference\\``` from the document, classifies each as a
    file path or an entity name, then checks it against the known sets.

    Returns
    -------
    tuple[float, list[str]]
        (valid_fraction, list_of_invalid_references)
    """
    refs = _BACKTICK_RE.findall(doc_text)
    if not refs:
        return 1.0, []

    # Deduplicate while preserving order for deterministic output
    seen: set[str] = set()
    unique_refs: list[str] = []
    for ref in refs:
        ref_stripped = ref.strip()
        if ref_stripped and ref_stripped not in seen:
            seen.add(ref_stripped)
            unique_refs.append(ref_stripped)

    invalid: list[str] = []
    valid_count = 0

    # Build a case-insensitive lookup for entity names
    entity_names_lower: set[str] = {n.lower() for n in entity_names}

    # File-path lookup: normalise to forward-slash, strip leading ./
    normalised_paths: set[str] = set()
    for fp in file_paths:
        normalised = fp.replace("\\", "/").lstrip("./")
        normalised_paths.add(normalised)
        # Also keep the original for exact-match
        normalised_paths.add(fp)

    for ref in unique_refs:
        if _looks_like_file_path(ref):
            # Normalise the reference the same way
            normalised_ref = ref.replace("\\", "/").lstrip("./")
            if normalised_ref in normalised_paths or ref in normalised_paths:
                valid_count += 1
            else:
                invalid.append(ref)
        else:
            # Entity name check (case-insensitive)
            if ref.lower() in entity_names_lower:
                valid_count += 1
            else:
                invalid.append(ref)

    total = len(unique_refs)
    score = valid_count / total if total > 0 else 1.0
    return score, invalid


# ---------------------------------------------------------------------------
# 3. Conciseness
# ---------------------------------------------------------------------------

_ENCYCLOPEDIA_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("function_signature", re.compile(r"^\s*def\s+\w+\(", re.MULTILINE)),
    ("parameter_description", re.compile(r"^\s*[-*]\s*`?\w+`?\s*[-:].*(?:parameter|param|arg)", re.MULTILINE | re.IGNORECASE)),
    ("code_block", re.compile(r"^```", re.MULTILINE)),
    ("type_annotation_line", re.compile(r"^\s*\w+\s*:\s*\w+", re.MULTILINE)),
]


def conciseness_score(
    doc_text: str,
    max_lines: int = 35,
) -> tuple[float, list[str]]:
    """Score conciseness based on line count and encyclopedia anti-patterns.

    A perfect document is at or under *max_lines* with no encyclopedia
    padding.  The score degrades as lines exceed the cap or anti-patterns
    are found.

    Returns
    -------
    tuple[float, list[str]]
        (score, list_of_issues)
    """
    lines = doc_text.strip().splitlines()
    line_count = len(lines)
    issues: list[str] = []

    # --- Line-count penalty ---
    if line_count <= max_lines:
        length_score = 1.0
    elif line_count <= max_lines * 2:
        # Linear decay from 1.0 to 0.5 between max_lines and 2*max_lines
        overshoot = (line_count - max_lines) / max_lines
        length_score = 1.0 - 0.5 * overshoot
        issues.append(f"Document has {line_count} lines (target <= {max_lines})")
    else:
        length_score = max(0.2, 0.5 - 0.1 * ((line_count - 2 * max_lines) / max_lines))
        issues.append(f"Document is very long: {line_count} lines (target <= {max_lines})")

    # --- Encyclopedia pattern penalty ---
    pattern_penalty = 0.0
    for pattern_name, pattern_re in _ENCYCLOPEDIA_PATTERNS:
        matches = pattern_re.findall(doc_text)
        if len(matches) > 2:
            penalty = min(0.15, 0.05 * len(matches))
            pattern_penalty += penalty
            issues.append(f"Encyclopedia pattern '{pattern_name}' found {len(matches)} times")

    score = max(0.0, min(1.0, length_score - pattern_penalty))
    return score, issues


# ---------------------------------------------------------------------------
# 4. Structural completeness
# ---------------------------------------------------------------------------

_REQUIRED_SECTIONS = [
    "Quick Commands",
    "Key Files",
    "Non-Obvious Patterns",
    "See Also",
]


def structural_completeness_score(doc_text: str) -> float:
    """Check for the presence of required sections in the document.

    Returns the fraction of ``_REQUIRED_SECTIONS`` found (case-insensitive
    substring match).
    """
    doc_lower = doc_text.lower()
    found = sum(
        1
        for section in _REQUIRED_SECTIONS
        if section.lower() in doc_lower
    )
    return found / len(_REQUIRED_SECTIONS)
