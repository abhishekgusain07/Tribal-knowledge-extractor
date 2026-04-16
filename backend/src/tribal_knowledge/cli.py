"""Typer CLI for the Tribal Knowledge Extractor."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from tribal_knowledge.discovery import discover_files
from tribal_knowledge.embeddings.chunker import create_chunks
from tribal_knowledge.embeddings.store import embed_and_store
from tribal_knowledge.ingestion import ingest as ingest_repo
from tribal_knowledge.models import (
    CodeChunk,
    DiscoveredFile,
    FileExtraction,
    GraphAnalysis,
    ParseManifest,
    RepoInfo,
    RepoSource,
)
from tribal_knowledge.parser import get_parser_for_language
from tribal_knowledge.pipeline.pipeline_models import QualityReport
from tribal_knowledge.serialization import (
    load_analysis,
    load_graph,
    save_analysis,
    save_extractions,
    save_graph,
    save_manifest,
)

app = typer.Typer(name="tribal-knowledge", help="Extract tribal knowledge from any codebase")
console = Console()


# ── Helpers ────────────────────────────────────────────────────────────────


def _parse_language_filter(languages: str | None) -> set[str] | None:
    """Parse a comma-separated language filter string into a set."""
    if languages is None:
        return None
    lang_set = {lang.strip().lower() for lang in languages.split(",") if lang.strip()}
    return lang_set if lang_set else None


def _count_languages(files: list[DiscoveredFile]) -> int:
    """Count distinct languages among discovered files."""
    langs: set[str] = set()
    for f in files:
        if f.language is not None:
            langs.add(f.language)
    return len(langs)


def _count_by_type(files: list[DiscoveredFile]) -> dict[str, int]:
    """Count files by file_type."""
    counts: dict[str, int] = {}
    for f in files:
        counts[f.file_type] = counts.get(f.file_type, 0) + 1
    return counts


def _parse_all_files(files: list[DiscoveredFile]) -> list[FileExtraction]:
    """Parse ASTs for all discovered files using the appropriate language parser.

    Iterates over each file, gets the parser for its language, reads the file,
    and calls ``parse_file``. Files whose language has no parser are skipped.
    """
    extractions: list[FileExtraction] = []

    for discovered in files:
        if discovered.language is None:
            continue

        # Only attempt parsing for code and test files
        if discovered.file_type not in ("code", "test"):
            continue

        try:
            parser = get_parser_for_language(discovered.language)
        except ValueError:
            # No parser available for this language yet — skip
            continue

        try:
            source = Path(discovered.absolute_path).read_bytes()
            extraction = parser.parse_file(discovered.path, source)
            extraction.file_type = discovered.file_type
            extractions.append(extraction)
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Failed to parse {discovered.path}: {exc}")

    return extractions


def _try_resolve_dependencies(
    extractions: list[FileExtraction],
) -> tuple[object, float]:
    """Attempt to resolve dependencies. Returns (dep_graph_or_None, resolution_rate)."""
    try:
        from tribal_knowledge.graph.dependency_resolver import resolve_dependencies

        dep_graph = resolve_dependencies(extractions)
        rate: float = dep_graph.resolution_rate
        return dep_graph, rate
    except (ImportError, AttributeError):
        # Module not yet implemented — return None
        return None, 0.0


def _try_build_graph(
    extractions: list[FileExtraction],
    dep_graph: object,
) -> object:
    """Attempt to build the knowledge graph. Returns nx.DiGraph or a new empty one."""
    try:
        from tribal_knowledge.graph.builder import build_knowledge_graph

        import networkx as nx

        if dep_graph is not None:
            from tribal_knowledge.models import DependencyGraph

            if isinstance(dep_graph, DependencyGraph):
                graph: nx.DiGraph = build_knowledge_graph(extractions, dep_graph)
                return graph

        # If dep_graph is None or wrong type, build without it
        from tribal_knowledge.models import DependencyGraph as DG

        empty_deps = DG()
        graph_fallback: nx.DiGraph = build_knowledge_graph(extractions, empty_deps)
        return graph_fallback
    except (ImportError, AttributeError):
        import networkx as nx

        return nx.DiGraph()


def _try_analyze_graph(graph: object) -> GraphAnalysis:
    """Attempt to run PageRank + community detection. Returns GraphAnalysis."""
    try:
        from tribal_knowledge.graph.analysis import analyze_graph

        import networkx as nx

        if isinstance(graph, nx.DiGraph):
            result: GraphAnalysis = analyze_graph(graph)
            return result
    except (ImportError, AttributeError):
        pass

    return GraphAnalysis()


def _build_manifest(
    repo_info: RepoInfo,
    files: list[DiscoveredFile],
    analysis: GraphAnalysis,
    resolution_rate: float,
) -> ParseManifest:
    """Build a parse manifest for self-refresh comparison."""
    file_hashes: dict[str, str] = {}
    for f in files:
        if f.content_hash:
            file_hashes[f.path] = f.content_hash

    module_assignments: dict[str, str] = {}
    for mod in analysis.modules:
        for file_path in mod.files:
            module_assignments[file_path] = mod.name

    return ParseManifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        head_commit=repo_info.head_commit,
        repo_name=repo_info.name,
        total_files_parsed=len(files),
        file_hashes=file_hashes,
        module_assignments=module_assignments,
        resolution_rate=resolution_rate,
        embedding_model="voyage-code-3",
    )


def _print_summary(
    repo_info: RepoInfo,
    files: list[DiscoveredFile],
    analysis: GraphAnalysis,
    chunks: list[CodeChunk],
    resolution_rate: float,
    elapsed: float,
    output_dir: str,
) -> None:
    """Print a Rich summary table after parsing completes."""
    table = Table(title="tribal-knowledge parse", show_header=False, border_style="blue")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    # Repository
    owner_prefix = f"{repo_info.owner}/" if repo_info.owner else ""
    table.add_row("Repository", f"{owner_prefix}{repo_info.name}")

    # Files by type
    type_counts = _count_by_type(files)
    code_count = type_counts.get("code", 0)
    test_count = type_counts.get("test", 0)
    config_count = type_counts.get("config", 0)
    table.add_row("Files", f"{code_count} code / {test_count} test / {config_count} config")

    # Languages
    lang_parts: list[str] = []
    total_lang_files = sum(repo_info.languages_detected.values()) or 1
    for lang, count in sorted(
        repo_info.languages_detected.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        pct = count * 100 // total_lang_files
        lang_parts.append(f"{lang} ({pct}%)")
    table.add_row("Languages", ", ".join(lang_parts) if lang_parts else "none detected")

    # Graph
    table.add_row("Graph", f"{analysis.total_nodes} nodes / {analysis.total_edges} edges")

    # Resolution rate
    table.add_row("Resolution", f"{resolution_rate:.0%} of calls resolved")

    # Modules
    table.add_row("Modules", f"{len(analysis.modules)} detected")

    # Chunks
    table.add_row("Chunks", f"{len(chunks)} created")

    # Time
    table.add_row("Time", f"{elapsed:.1f}s")

    # Top modules
    if analysis.modules:
        table.add_section()
        table.add_row("[bold]Top modules by importance[/bold]", "")
        for idx, mod in enumerate(analysis.modules[:5], 1):
            entity_str = ", ".join(mod.top_entities[:3]) if mod.top_entities else "n/a"
            table.add_row(f"  {idx}. {mod.name}", entity_str)

    # Cross-module deps
    table.add_section()
    table.add_row("Cross-module dependencies", str(len(analysis.cross_module_deps)))

    # Tribal signals
    tribal_count = 0
    # Count from chunks isn't quite right; we'd need extractions — just show 0 or skip
    table.add_row("Tribal signals", f"detected (see extractions.json)")

    # Output location
    table.add_section()
    table.add_row("Output", output_dir)
    table.add_row("  knowledge_graph.json", "")
    table.add_row("  analysis.json", "")
    table.add_row("  manifest.json", "(for self-refresh)")
    table.add_row("  chromadb/", "(vector store)")

    console.print()
    console.print(table)


# ── Commands ───────────────────────────────────────────────────────────────


@app.command()
def parse(
    source: str = typer.Argument(help="GitHub URL or local path to repo"),
    output: str = typer.Option(
        "./tribal-knowledge-output",
        help="Output directory",
    ),
    branch: str | None = typer.Option(None, help="Branch to clone (for GitHub URLs)"),
    languages: str | None = typer.Option(
        None, help="Comma-separated language filter (e.g. 'python,typescript')"
    ),
) -> None:
    """Parse a repository and build the knowledge graph + vector store."""
    import networkx as nx

    start_time = time.time()
    output_dir = Path(output)
    language_filter = _parse_language_filter(languages)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Ingest repo
        task = progress.add_task("Cloning/reading repository...", total=None)
        repo_source = RepoSource(path_or_url=source, branch=branch)
        repo_info = ingest_repo(repo_source)
        progress.update(task, completed=1, total=1)

        # Step 2: Discover files
        task = progress.add_task("Discovering files...", total=None)
        files = discover_files(repo_info.local_path, language_filter)
        progress.update(task, completed=1, total=1)
        console.print(
            f"  Found [bold]{len(files)}[/bold] files across "
            f"[bold]{_count_languages(files)}[/bold] languages"
        )

        # Step 3: Parse ASTs
        task = progress.add_task("Parsing source code...", total=None)
        extractions = _parse_all_files(files)
        progress.update(task, completed=1, total=1)
        console.print(f"  Parsed [bold]{len(extractions)}[/bold] files successfully")

        # Step 4: Resolve dependencies
        task = progress.add_task("Resolving dependencies...", total=None)
        dep_graph, resolution_rate = _try_resolve_dependencies(extractions)
        progress.update(task, completed=1, total=1)
        console.print(f"  Call resolution rate: [bold]{resolution_rate:.0%}[/bold]")

        # Step 5: Build knowledge graph
        task = progress.add_task("Building knowledge graph...", total=None)
        graph = _try_build_graph(extractions, dep_graph)
        progress.update(task, completed=1, total=1)

        # Step 6: Analyze (PageRank + communities)
        task = progress.add_task("Analyzing importance & modules...", total=None)
        if isinstance(graph, nx.DiGraph):
            analysis = _try_analyze_graph(graph)
        else:
            analysis = GraphAnalysis()
        progress.update(task, completed=1, total=1)

        # Step 7: Create chunks
        task = progress.add_task("Creating chunks...", total=None)
        if isinstance(graph, nx.DiGraph):
            chunks = create_chunks(extractions, graph, analysis)
        else:
            chunks = []
        progress.update(task, completed=1, total=1)
        console.print(f"  Created [bold]{len(chunks)}[/bold] chunks")

        # Step 8: Embed and store
        task = progress.add_task("Embedding and storing...", total=None)
        try:
            embed_and_store(chunks, output_dir)
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Embedding failed: {exc}")
        progress.update(task, completed=1, total=1)

        # Step 9: Serialize everything
        task = progress.add_task("Saving outputs...", total=None)
        if isinstance(graph, nx.DiGraph):
            save_graph(graph, output_dir)
        save_analysis(analysis, output_dir)
        save_extractions(extractions, output_dir)

        manifest = _build_manifest(repo_info, files, analysis, resolution_rate)
        save_manifest(manifest, output_dir)
        progress.update(task, completed=1, total=1)

    elapsed = time.time() - start_time

    _print_summary(
        repo_info=repo_info,
        files=files,
        analysis=analysis,
        chunks=chunks,
        resolution_rate=resolution_rate,
        elapsed=elapsed,
        output_dir=output,
    )


# ── Generate command ──────────────────────────────────────────────────────


@app.command()
def generate(
    source: str = typer.Argument(help="Path to repo (must have been parsed first)"),
    output: str = typer.Option(
        "./tribal-knowledge-output",
        help="Output directory (from parse step)",
    ),
) -> None:
    """Generate compass context files from a parsed repository."""
    import json

    from networkx.readwrite import json_graph

    from tribal_knowledge.pipeline.graph import build_pipeline
    from tribal_knowledge.pipeline.helpers import build_file_tree_summary
    from tribal_knowledge.pipeline.output import write_outputs

    start_time = time.time()
    output_dir = Path(output)
    tk_dir = output_dir / ".tribal-knowledge"

    # 1. Check that Phase 1 output files exist
    required_files: list[tuple[str, Path]] = [
        ("knowledge_graph.json", tk_dir / "knowledge_graph.json"),
        ("analysis.json", tk_dir / "analysis.json"),
        ("extractions.json", tk_dir / "extractions.json"),
    ]
    chromadb_dir = tk_dir / "chromadb"

    missing: list[str] = []
    for name, path in required_files:
        if not path.exists():
            missing.append(name)
    if not chromadb_dir.exists():
        missing.append("chromadb/")

    if missing:
        console.print(
            f"[red]Error:[/red] Missing Phase 1 outputs: {', '.join(missing)}\n"
            f"Run [bold]tribal-knowledge parse[/bold] first."
        )
        raise typer.Exit(code=1)

    source_path = Path(source)
    if not source_path.is_dir():
        console.print(f"[red]Error:[/red] Source path does not exist: {source}")
        raise typer.Exit(code=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # 2. Load Phase 1 data
        task = progress.add_task("Loading Phase 1 data...", total=None)

        graph = load_graph(output_dir)
        knowledge_graph_data: dict[str, object] = json_graph.node_link_data(graph)

        analysis = load_analysis(output_dir)
        analysis_data: dict[str, object] = json.loads(analysis.model_dump_json())

        extractions_path = tk_dir / "extractions.json"
        extractions_raw = extractions_path.read_text(encoding="utf-8")
        extractions_data: list[dict[str, object]] = json.loads(extractions_raw)

        progress.update(task, completed=1, total=1)
        console.print(
            f"  Loaded graph ({len(graph.nodes)} nodes), "
            f"analysis ({len(analysis.modules)} modules), "
            f"extractions ({len(extractions_data)} files)"
        )

        # 3. Build file tree summary
        task = progress.add_task("Building file tree summary...", total=None)
        file_tree_summary = build_file_tree_summary(str(source_path))
        progress.update(task, completed=1, total=1)

        # 4. Build initial PipelineState
        initial_state: dict[str, object] = {
            "repo_path": str(source_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "knowledge_graph_data": knowledge_graph_data,
            "analysis_data": analysis_data,
            "extractions_data": extractions_data,
            "chromadb_path": str(chromadb_dir),
            "file_tree_summary": file_tree_summary,
            # Per-module iteration state (initialised by Explorer)
            "module_map": [],
            "module_queue": [],
            "current_module": "",
            "analyst_output": {},
            "current_draft": "",
            "critique": {},
            "score": 0.0,
            "iteration": 0,
            "previous_score": 0.0,
            # Accumulated outputs
            "module_docs": {},
            "all_analyst_outputs": [],
            "overview_doc": "",
            "conventions_doc": "",
            "agents_doc": "",
            "dependency_index_doc": "",
            # Tracking
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "generation_log": [],
            "errors": [],
        }

        # 5. Compile and run the pipeline
        task = progress.add_task("Running generation pipeline...", total=None)
        pipeline = build_pipeline()
        result: dict[str, object] = pipeline.invoke(initial_state)
        progress.update(task, completed=1, total=1)

        # 6. Write outputs
        task = progress.add_task("Writing output files...", total=None)
        report = write_outputs(result, str(output_dir))  # type: ignore[arg-type]
        progress.update(task, completed=1, total=1)

    elapsed = time.time() - start_time

    # 7. Print Rich summary table
    _print_generate_summary(result, report, elapsed, str(output_dir))


def _print_generate_summary(
    result: dict[str, object],
    report: QualityReport,
    elapsed: float,
    output_dir: str,
) -> None:
    """Print a Rich summary table after generation completes."""
    table = Table(title="tribal-knowledge generate", show_header=False, border_style="green")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    # Module docs generated
    module_docs = result.get("module_docs", {})
    if isinstance(module_docs, dict):
        table.add_row("Module docs", str(len(module_docs)))
    else:
        table.add_row("Module docs", "0")

    # Per-document scores
    if report.documents:
        table.add_section()
        table.add_row("[bold]Documents[/bold]", "[bold]Score / Tokens[/bold]")
        for doc in report.documents:
            score_str = f"{doc.score:.2f}" if doc.score > 0 else "n/a"
            table.add_row(
                f"  {doc.name}",
                f"score={score_str}, tokens={doc.tokens_used}",
            )

    # Aggregates
    table.add_section()
    table.add_row("Average score", f"{report.average_score:.2f}")
    table.add_row("Total tokens", str(report.total_tokens))
    table.add_row("Total cost", f"${report.total_cost:.4f}")
    table.add_row("Time", f"{elapsed:.1f}s")

    # Output location
    table.add_section()
    table.add_row("Output", f"{output_dir}/context/")

    # Synthesis docs
    synthesis_names = ["OVERVIEW.md", "CONVENTIONS.md", "AGENTS.md", "dependency-index.md"]
    for name in synthesis_names:
        table.add_row(f"  {name}", "")

    console.print()
    console.print(table)


# ── Evaluate command ─────────────────────────────────────────────────────


def _print_evaluate_summary(
    report: "EvaluationReport",
    gaps: list["CoverageGap"],
    elapsed: float,
) -> None:
    """Print a Rich summary table after evaluation completes."""
    table = Table(title="tribal-knowledge evaluate", show_header=False, border_style="magenta")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    # Per-document scores
    if report.documents:
        table.add_section()
        table.add_row(
            "[bold]Document[/bold]",
            "[bold]Composite / Coverage / Accuracy / Conciseness[/bold]",
        )
        for doc in report.documents:
            composite = doc.composite_score
            issues_str = f"  ({len(doc.issues)} issue(s))" if doc.issues else ""
            table.add_row(
                f"  {doc.name}",
                (
                    f"composite={composite:.2f}, "
                    f"coverage={doc.entity_coverage:.2f}, "
                    f"accuracy={doc.factual_accuracy:.2f}, "
                    f"conciseness={doc.conciseness:.2f}"
                    f"{issues_str}"
                ),
            )

    # Aggregates
    table.add_section()
    table.add_row("Average composite", f"{report.average_composite:.4f}")
    for dim_name, dim_avg in report.average_by_dimension.items():
        table.add_row(f"  {dim_name}", f"{dim_avg:.4f}")
    table.add_row("Invalid references", str(report.total_invalid_references))
    table.add_row("Missing entities", str(report.total_missing_entities))

    # Coverage gaps
    table.add_section()
    table.add_row("Coverage gaps", str(len(gaps)))
    if gaps:
        for gap in gaps[:5]:
            severity_pct = f"{gap.severity:.0%}"
            table.add_row(
                f"  [{gap.gap_type}]",
                f"severity={severity_pct}: {gap.description[:80]}",
            )
        if len(gaps) > 5:
            table.add_row("  ...", f"({len(gaps) - 5} more)")

    # Time
    table.add_section()
    table.add_row("Time", f"{elapsed:.1f}s")

    console.print()
    console.print(table)


@app.command()
def evaluate(
    context: str = typer.Argument(help="Path to context directory (contains generated .md files)"),
    output_dir: str = typer.Option("./tribal-knowledge-output", help="Output directory with Phase 1 data"),
    repo: str = typer.Option(".", help="Path to the repo (for file path validation)"),
    use_llm: bool = typer.Option(False, "--llm", help="Include LLM-as-judge usefulness scoring"),
) -> None:
    """Evaluate the quality of generated context files."""
    import json as json_mod

    from tribal_knowledge.evaluation.evaluator import evaluate_all
    from tribal_knowledge.evaluation.gap_filler import CoverageGap, detect_gaps
    from tribal_knowledge.evaluation.llm_judge import judge_usefulness
    from tribal_knowledge.evaluation.models import DocumentEvaluation, EvaluationReport
    from tribal_knowledge.serialization import load_manifest

    start_time = time.time()
    context_path = Path(context)
    output_path = Path(output_dir)
    repo_path = Path(repo)

    if not context_path.is_dir():
        console.print(f"[red]Error:[/red] Context directory does not exist: {context}")
        raise typer.Exit(code=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # 1. Load knowledge graph + analysis from output_dir
        task = progress.add_task("Loading Phase 1 data...", total=None)
        graph = load_graph(output_path)
        analysis = load_analysis(output_path)
        progress.update(task, completed=1, total=1)

        # 2. Collect all relative file paths from the repo
        task = progress.add_task("Scanning repo file paths...", total=None)
        repo_file_paths: set[str] = set()
        repo_resolved = repo_path.resolve()
        for item in repo_resolved.rglob("*"):
            if item.is_file():
                try:
                    rel = str(item.relative_to(repo_resolved))
                    repo_file_paths.add(rel)
                except ValueError:
                    pass
        progress.update(task, completed=1, total=1)
        console.print(f"  Found [bold]{len(repo_file_paths)}[/bold] files in repo")

        # 3. Run evaluate_all
        task = progress.add_task("Evaluating context documents...", total=None)
        report: EvaluationReport = evaluate_all(
            context_dir=context_path,
            knowledge_graph=graph,
            analysis=analysis,
            repo_file_paths=repo_file_paths,
        )
        progress.update(task, completed=1, total=1)
        console.print(
            f"  Evaluated [bold]{len(report.documents)}[/bold] documents, "
            f"average composite: [bold]{report.average_composite:.4f}[/bold]"
        )

        # 4. Optional LLM-as-judge scoring
        if use_llm:
            task = progress.add_task("Running LLM-as-judge...", total=None)
            for doc_eval in report.documents:
                md_file = context_path / doc_eval.name
                if md_file.exists():
                    doc_text = md_file.read_text(encoding="utf-8")
                    module_name = doc_eval.name.removesuffix(".md")
                    score, _reasoning = judge_usefulness(doc_text, module_name)
                    doc_eval.llm_usefulness = round(score, 4)
            progress.update(task, completed=1, total=1)

        # 5. Detect coverage gaps
        task = progress.add_task("Detecting coverage gaps...", total=None)
        # Build module_docs dict from context directory
        module_docs: dict[str, str] = {}
        for md_file in sorted(context_path.glob("*.md")):
            try:
                module_docs[md_file.stem] = md_file.read_text(encoding="utf-8")
            except OSError:
                pass

        # Load extractions for gap detection
        extractions_path = output_path / ".tribal-knowledge" / "extractions.json"
        extractions_list: list[FileExtraction] = []
        if extractions_path.exists():
            raw_extractions = extractions_path.read_text(encoding="utf-8")
            for item in json_mod.loads(raw_extractions):
                extractions_list.append(FileExtraction.model_validate(item))

        gaps: list[CoverageGap] = detect_gaps(
            module_docs=module_docs,
            knowledge_graph=graph,
            analysis=analysis,
            extractions=extractions_list,
        )
        progress.update(task, completed=1, total=1)
        console.print(f"  Detected [bold]{len(gaps)}[/bold] coverage gaps")

        # 6. Write evaluation-report.json
        task = progress.add_task("Writing evaluation report...", total=None)
        report_payload = report.model_dump()
        # Include composite_score per document (it's a @property, not in model_dump)
        for i, doc_eval in enumerate(report.documents):
            report_payload["documents"][i]["composite_score"] = round(
                doc_eval.composite_score, 4
            )
        # Include gap summary
        report_payload["coverage_gaps"] = [g.model_dump() for g in gaps]

        report_file = context_path / "evaluation-report.json"
        report_file.write_text(
            json_mod.dumps(report_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        progress.update(task, completed=1, total=1)

    elapsed = time.time() - start_time
    _print_evaluate_summary(report, gaps, elapsed)
    console.print(f"\n  Report written to [bold]{report_file}[/bold]")


# ── Refresh command ──────────────────────────────────────────────────────


def _print_refresh_summary(
    staleness: "StalenessReport",
    elapsed: float,
) -> None:
    """Print a Rich summary table after refresh completes."""
    table = Table(title="tribal-knowledge refresh", show_header=False, border_style="yellow")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Stale files", str(len(staleness.stale_files)))
    table.add_row("New files", str(len(staleness.new_files)))
    table.add_row("Deleted files", str(len(staleness.deleted_files)))
    table.add_row("Stale modules", str(len(staleness.stale_modules)))
    table.add_row("Broken references", str(len(staleness.broken_references)))
    table.add_row("Commit distance", str(staleness.commit_distance))

    table.add_section()
    if staleness.needs_full_reparse:
        table.add_row("Action", "[red]Full reparse[/red]")
    elif staleness.needs_partial_reparse:
        table.add_row("Action", "[yellow]Partial reparse[/yellow]")
    elif staleness.needs_doc_refresh_only:
        table.add_row("Action", "[cyan]Doc refresh only[/cyan]")
    else:
        table.add_row("Action", "[green]Up to date[/green]")

    table.add_section()
    table.add_row("Time", f"{elapsed:.1f}s")

    console.print()
    console.print(table)


@app.command()
def refresh(
    repo: str = typer.Argument(help="Path to the repository"),
    output: str = typer.Option("./tribal-knowledge-output", help="Output directory"),
) -> None:
    """Refresh context files based on repo changes since last parse."""
    from tribal_knowledge.refresh.detector import StalenessReport, detect_staleness
    from tribal_knowledge.refresh.reparser import selective_reparse
    from tribal_knowledge.serialization import load_manifest

    start_time = time.time()
    repo_path = Path(repo)
    output_path = Path(output)

    if not repo_path.is_dir():
        console.print(f"[red]Error:[/red] Repository path does not exist: {repo}")
        raise typer.Exit(code=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # 1. Load manifest
        task = progress.add_task("Loading manifest...", total=None)
        manifest = load_manifest(output_path)
        progress.update(task, completed=1, total=1)

        if manifest is None:
            console.print(
                "[red]Error:[/red] No manifest found. "
                "Run [bold]tribal-knowledge parse[/bold] first."
            )
            raise typer.Exit(code=1)

        # 2. Detect staleness
        task = progress.add_task("Detecting staleness...", total=None)
        context_dir = output_path / "context"
        staleness: StalenessReport = detect_staleness(
            repo_path=repo_path,
            manifest=manifest,
            context_dir=context_dir,
        )
        progress.update(task, completed=1, total=1)

        console.print(
            f"  [bold]{len(staleness.stale_files)}[/bold] stale, "
            f"[bold]{len(staleness.new_files)}[/bold] new, "
            f"[bold]{len(staleness.deleted_files)}[/bold] deleted"
        )

        # 3. Run selective reparse
        task = progress.add_task("Running selective reparse...", total=None)
        selective_reparse(
            repo_path=repo_path,
            output_dir=output_path,
            staleness=staleness,
        )
        progress.update(task, completed=1, total=1)

    elapsed = time.time() - start_time
    _print_refresh_summary(staleness, elapsed)


# ── Run command ──────────────────────────────────────────────────────────


@app.command()
def run(
    source: str = typer.Argument(help="GitHub URL or local path to repo"),
    output: str = typer.Option("./tribal-knowledge-output", help="Output directory"),
    branch: str | None = typer.Option(None, help="Branch to clone"),
    languages: str | None = typer.Option(None, help="Comma-separated language filter"),
    skip_evaluate: bool = typer.Option(False, "--skip-evaluate", help="Skip evaluation after generation"),
) -> None:
    """Parse + generate + evaluate in one shot."""
    console.print("[bold blue]Step 1/3:[/bold blue] Parsing repository...")
    parse(source=source, output=output, branch=branch, languages=languages)

    console.print("\n[bold blue]Step 2/3:[/bold blue] Generating context files...")
    generate(source=source, output=output)

    if not skip_evaluate:
        console.print("\n[bold blue]Step 3/3:[/bold blue] Evaluating generated docs...")
        context_dir = str(Path(output) / "context")
        evaluate(context=context_dir, output_dir=output, repo=source, use_llm=False)
    else:
        console.print("\n[bold blue]Step 3/3:[/bold blue] Skipped evaluation (--skip-evaluate)")

    console.print("\n[bold green]Done![/bold green] Full pipeline complete.")
