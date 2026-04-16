"""Benchmark runner -- run parse + generate + evaluate across multiple repos.

Collects per-repo metrics (timing, cost, scores) and writes a consolidated
``benchmark-report.json``.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

console = Console()


# ── Models ───────────────────────────────────────────────────────────────────


class BenchmarkResult(BaseModel):
    """Metrics for a single benchmarked repository."""

    repo_name: str
    repo_url: str = ""
    lines_of_code: int = 0
    languages: list[str] = Field(default_factory=list)
    parse_time_seconds: float = 0.0
    generate_time_seconds: float = 0.0
    total_cost: float = 0.0
    module_docs_count: int = 0
    average_composite_score: float = 0.0
    error: str = ""


class BenchmarkReport(BaseModel):
    """Aggregate report across all benchmarked repositories."""

    results: list[BenchmarkResult] = Field(default_factory=list)
    run_date: str = ""
    total_cost: float = 0.0
    total_time_seconds: float = 0.0
    average_score: float = 0.0


# ── Internal helpers ─────────────────────────────────────────────────────────


def _count_lines_of_code(repo_path: Path) -> int:
    """Count total lines across all non-hidden, non-vendored files."""
    total: int = 0
    skip_dirs: frozenset[str] = frozenset(
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
            "vendor",
            ".cargo",
            "target",
        }
    )

    for item in repo_path.rglob("*"):
        if not item.is_file():
            continue

        # Skip items under excluded directories
        skip = False
        for part in item.relative_to(repo_path).parts[:-1]:
            if part in skip_dirs:
                skip = True
                break
        if skip:
            continue

        try:
            total += sum(1 for _ in item.open("r", encoding="utf-8", errors="replace"))
        except OSError:
            continue

    return total


def _detect_languages(repo_path: Path) -> list[str]:
    """Detect programming languages present in the repository."""
    from tribal_knowledge.models import EXTENSION_MAP

    found: set[str] = set()
    for item in repo_path.rglob("*"):
        if item.is_file():
            suffix = item.suffix.lower()
            lang = EXTENSION_MAP.get(suffix)
            if lang is not None:
                found.add(lang)
    return sorted(found)


def _run_single_repo(
    repo_info: dict[str, str],
    output_base: str,
) -> BenchmarkResult:
    """Run the full pipeline on a single repo and collect metrics."""
    from tribal_knowledge.cli import evaluate, generate, parse

    repo_name = repo_info.get("name", "unknown")
    repo_url = repo_info.get("url", "")
    branch = repo_info.get("branch") or None

    output_dir = str(Path(output_base) / repo_name)
    source = repo_url if repo_url else repo_name

    result = BenchmarkResult(repo_name=repo_name, repo_url=repo_url)

    # 1. Parse
    parse_start = time.time()
    parse(source=source, output=output_dir, branch=branch, languages=None)
    result.parse_time_seconds = time.time() - parse_start

    # 2. Generate
    gen_start = time.time()
    generate(source=source, output=output_dir)
    result.generate_time_seconds = time.time() - gen_start

    # 3. Evaluate
    context_dir = str(Path(output_dir) / "context")
    evaluate(context=context_dir, output_dir=output_dir, repo=source, use_llm=False)

    # 4. Collect metrics from outputs
    output_path = Path(output_dir)

    # Read evaluation report
    eval_report_path = Path(context_dir) / "evaluation-report.json"
    if eval_report_path.exists():
        eval_data: dict[str, object] = json.loads(
            eval_report_path.read_text(encoding="utf-8")
        )
        avg_composite = eval_data.get("average_composite", 0.0)
        if isinstance(avg_composite, (int, float)):
            result.average_composite_score = float(avg_composite)

    # Count module docs
    context_path = Path(context_dir)
    if context_path.is_dir():
        md_files = list(context_path.glob("*.md"))
        result.module_docs_count = len(md_files)

    # Detect languages and LOC from the source
    source_path = Path(source)
    if source_path.is_dir():
        result.lines_of_code = _count_lines_of_code(source_path)
        result.languages = _detect_languages(source_path)

    # Read cost from pipeline output if available
    tk_dir = output_path / ".tribal-knowledge"
    quality_report_path = output_path / "context" / "quality-report.json"
    if quality_report_path.exists():
        try:
            qr_data: dict[str, object] = json.loads(
                quality_report_path.read_text(encoding="utf-8")
            )
            cost_val = qr_data.get("total_cost", 0.0)
            if isinstance(cost_val, (int, float)):
                result.total_cost = float(cost_val)
        except (json.JSONDecodeError, OSError):
            pass

    return result


# ── Public API ───────────────────────────────────────────────────────────────


def run_benchmark(
    repos: list[dict[str, str]],
    output_base: str,
) -> BenchmarkReport:
    """Run the full tribal-knowledge pipeline on each repo and aggregate results.

    Parameters
    ----------
    repos:
        List of repo dicts with keys ``"url"``, ``"name"``, and optionally ``"branch"``.
    output_base:
        Base directory where per-repo outputs are written (``{output_base}/{name}/``).

    Returns
    -------
    BenchmarkReport
        Consolidated report with per-repo results and aggregate metrics.
    """
    overall_start = time.time()
    report = BenchmarkReport(
        run_date=datetime.now(timezone.utc).isoformat(),
    )

    for repo_info in repos:
        repo_name = repo_info.get("name", "unknown")
        console.print(f"\n[bold blue]{'=' * 60}[/bold blue]")
        console.print(f"[bold]Benchmarking:[/bold] {repo_name}")
        console.print(f"[bold blue]{'=' * 60}[/bold blue]")

        try:
            result = _run_single_repo(repo_info, output_base)
            report.results.append(result)
            console.print(
                f"[green]Completed {repo_name}:[/green] "
                f"composite={result.average_composite_score:.4f}, "
                f"parse={result.parse_time_seconds:.1f}s, "
                f"generate={result.generate_time_seconds:.1f}s"
            )
        except Exception as exc:
            tb = traceback.format_exc()
            console.print(f"[red]Failed {repo_name}:[/red] {exc}")
            console.print(f"[dim]{tb}[/dim]")
            error_result = BenchmarkResult(
                repo_name=repo_name,
                repo_url=repo_info.get("url", ""),
                error=str(exc),
            )
            report.results.append(error_result)

    # Aggregate metrics
    report.total_time_seconds = time.time() - overall_start
    successful_results = [r for r in report.results if not r.error]
    report.total_cost = sum(r.total_cost for r in successful_results)

    if successful_results:
        report.average_score = sum(
            r.average_composite_score for r in successful_results
        ) / len(successful_results)

    # Write benchmark report
    output_base_path = Path(output_base)
    output_base_path.mkdir(parents=True, exist_ok=True)
    report_path = output_base_path / "benchmark-report.json"
    report_path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Print summary table
    _print_benchmark_summary(report)
    console.print(f"\n  Report written to [bold]{report_path}[/bold]")

    return report


def _print_benchmark_summary(report: BenchmarkReport) -> None:
    """Print a Rich summary table for the benchmark run."""
    table = Table(title="tribal-knowledge benchmark", border_style="blue")
    table.add_column("Repository", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Parse (s)", justify="right")
    table.add_column("Generate (s)", justify="right")
    table.add_column("Cost ($)", justify="right")
    table.add_column("Docs", justify="right")
    table.add_column("Status")

    for result in report.results:
        if result.error:
            table.add_row(
                result.repo_name,
                "-",
                "-",
                "-",
                "-",
                "-",
                f"[red]FAILED: {result.error[:40]}[/red]",
            )
        else:
            table.add_row(
                result.repo_name,
                f"{result.average_composite_score:.4f}",
                f"{result.parse_time_seconds:.1f}",
                f"{result.generate_time_seconds:.1f}",
                f"{result.total_cost:.4f}",
                str(result.module_docs_count),
                "[green]OK[/green]",
            )

    # Footer row with totals
    table.add_section()
    table.add_row(
        "[bold]Totals / Averages[/bold]",
        f"{report.average_score:.4f}",
        "",
        "",
        f"{report.total_cost:.4f}",
        "",
        f"{report.total_time_seconds:.1f}s total",
    )

    console.print()
    console.print(table)
