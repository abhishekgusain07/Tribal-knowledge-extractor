<p align="center">
  <img src="assets/banner.png" alt="Tribal Knowledge" width="720" />
</p>

<h1 align="center">Tribal Knowledge</h1>

<p align="center">
  <strong>Point it at any repo. Get context files that make AI tools 10x more effective.</strong>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#output">Output</a> &bull;
  <a href="#docker">Docker</a> &bull;
  <a href="#evaluation">Evaluation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/languages-6-green" />
  <img src="https://img.shields.io/badge/tests-235_passing-brightgreen" />
  <img src="https://img.shields.io/badge/cost-%3C%242%2Frepo-orange" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
</p>

---

New engineers spend **3-6 weeks** building a mental model of a codebase. Not because the code is unreadable — because the *why* behind decisions, the implicit conventions, and critical data flows live only in senior engineers' heads.

**Tribal Knowledge** extracts that knowledge automatically. It parses your repo, builds a knowledge graph, and uses a 5-agent AI pipeline to generate concise **"compass" context files** — inspired by [Meta's approach](https://engineering.fb.com/) to codebase documentation.

The output works with **AGENTS.md** (the new cross-tool standard), **CLAUDE.md**, **.cursorrules**, and any AI coding tool.

---

## Quickstart

```bash
pip install tribal-knowledge
```

```bash
# One command — parse, generate, evaluate
tribal-knowledge run ./my-project --output ./context/

# Or step by step
tribal-knowledge parse ./my-project
tribal-knowledge generate ./my-project
tribal-knowledge evaluate ./context/context --repo ./my-project
```

```
Generated 7 context files:
  context/OVERVIEW.md           project-level compass
  context/auth.md               authentication lifecycle
  context/api-layer.md          API routing & middleware
  context/CONVENTIONS.md        naming, patterns, gotchas
  context/AGENTS.md             cross-tool AI context
  context/dependency-index.md   what depends on what
  context/quality-report.json   scores & cost breakdown

Average score: 0.87  |  Cost: $0.43  |  Time: 4m 12s
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # Required — Claude Sonnet for analysis & writing
export GOOGLE_API_KEY=...               # Optional — Gemini Flash for exploration (cheaper)
export VOYAGE_API_KEY=...               # Optional — Voyage Code 3 for embeddings
```

> Works without API keys using fallback paths (heuristic analysis, built-in embeddings) — just with lower quality.

---

## How It Works

```
                     ┌─────────────────┐
                     │   Your Repo     │
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  Parse   │   │  Graph   │   │  Embed   │
        │ 6 langs  │   │ PageRank │   │ ChromaDB │
        │tree-sitter│   │ Louvain  │   │ Voyage   │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             └───────────────┼───────────────┘
                             ▼
        ┌────────────────────────────────────────┐
        │         5-Agent LangGraph Pipeline      │
        │                                        │
        │  Explorer → Analyst → Writer → Critic  │
        │                           ↑       │    │
        │                           └─Fixer─┘    │
        │              (loop until score ≥ 0.85)  │
        └────────────────────┬───────────────────┘
                             ▼
                     ┌───────────────┐
                     │ Context Files │
                     │  AGENTS.md    │
                     │  OVERVIEW.md  │
                     │  modules/*.md │
                     └───────────────┘
```

### The Pipeline

| Phase | What | How |
|-------|------|-----|
| **Parse** | Extract every function, class, import, type across 6 languages | Tree-sitter AST parsing |
| **Graph** | Build a knowledge graph of entities + relationships | NetworkX + PageRank + Louvain communities |
| **Embed** | Index code chunks for semantic retrieval | Voyage Code 3 + ChromaDB |
| **Explore** | Identify logical modules and tech stack | Gemini Flash (cheap, 1M context) |
| **Analyze** | Extract tribal knowledge per module using Meta's Five Questions | Claude Sonnet |
| **Write** | Generate 25-35 line compass docs (not encyclopedias) | Claude Sonnet |
| **Critique** | Score on accuracy, coverage, conciseness, usefulness | Claude Sonnet |
| **Fix** | Surgical edits to address critique (not rewrites) | Claude Sonnet |
| **Synthesize** | Produce OVERVIEW, CONVENTIONS, AGENTS.md | Claude Sonnet |

### Supported Languages

| Language | Parser | Status |
|----------|--------|--------|
| Python | tree-sitter-python | Full |
| TypeScript/TSX | tree-sitter-typescript | Full |
| JavaScript/JSX | tree-sitter-javascript | Full |
| Go | tree-sitter-go | Full |
| Rust | tree-sitter-rust | Full |
| Java | tree-sitter-java | Full |

---

## Output

### Philosophy: Compass, Not Encyclopedia

Every generated line passes this test: **"Could a developer figure this out in under 60 seconds by reading the code?"** If yes, it's cut.

Each module doc is **25-35 lines** with exactly 4 sections:

```markdown
# Auth

## Quick Commands
- To add a new auth method: add handler in `login.py`, register in AUTH_METHODS
- To debug token issues: check `verify_token()` in `src/auth/token.py`

## Key Files
- `src/auth/login.py` — Entry point for all authentication flows
- `src/auth/token.py` — JWT creation and verification with Ed25519
- `src/auth/models.py` — User model with repository pattern

## Non-Obvious Patterns
- bcrypt rounds must stay at 12 — perf regression at 14 breaks SLAs
- verify_token checks expiry BEFORE signature (cheaper to reject expired)
- Token cache TTL must match JWT expiry or stale tokens pass verification

## See Also
- [db.md](db.md) — get_connection() called during every login
- [api.md](api.md) — handle_request() calls verify_token() on every request
```

### Generated Files

| File | Purpose | Length |
|------|---------|--------|
| `AGENTS.md` | Cross-tool AI context (Claude Code, Cursor, Copilot, Gemini CLI, Codex, Windsurf, Aider, Zed) | ~1,000 tokens |
| `OVERVIEW.md` | Project compass: what, why, how, module map | ~1,000 tokens |
| `{module}.md` | Per-module: Quick Commands, Key Files, Non-Obvious Patterns, See Also | ~1,000 tokens each |
| `CONVENTIONS.md` | Naming, error handling, import patterns, gotchas | ~1,000 tokens |
| `dependency-index.md` | Cross-module dependency map + change impact | varies |
| `quality-report.json` | Per-doc scores, cost, token usage | JSON |

**Total context budget:** All files together consume **< 0.1%** of a model's context window.

---

## CLI Reference

```bash
tribal-knowledge parse <repo>          # Phase 1: AST parse + knowledge graph + embeddings
tribal-knowledge generate <repo>       # Phase 2: 5-agent pipeline → context files
tribal-knowledge evaluate <context/>   # Score generated docs (4 automated + LLM-as-judge)
tribal-knowledge refresh <repo>        # Update stale docs (selective re-parse)
tribal-knowledge run <repo>            # All-in-one: parse + generate + evaluate
```

| Flag | Command | Description |
|------|---------|-------------|
| `--output` | parse, generate, run | Output directory (default: `./tribal-knowledge-output`) |
| `--languages` | parse, run | Filter languages: `python,typescript` |
| `--branch` | parse, run | Git branch to clone |
| `--llm` | evaluate | Include LLM-as-judge usefulness scoring |
| `--skip-evaluate` | run | Skip evaluation step |

---

## Docker

```bash
# Build
docker build -t tribal-knowledge .

# Run on a local repo
docker run -v $(pwd):/repo \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  tribal-knowledge run /repo --output /repo/context/

# Run on a GitHub repo
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  tribal-knowledge run https://github.com/user/repo
```

---

## Evaluation

Every generated document is scored on 4 dimensions:

| Dimension | What It Measures | Method |
|-----------|-----------------|--------|
| **Accuracy** | Do referenced files/functions exist? | Cross-check against knowledge graph |
| **Coverage** | Are top PageRank entities mentioned? | Automated entity lookup |
| **Conciseness** | Under 35 lines? No encyclopedia content? | Pattern detection |
| **Usefulness** | Would a new dev learn something? | LLM-as-judge + heuristics |

Quality gate: documents loop through Critic → Fixer until score **>= 0.85** or max 4 iterations.

### Self-Refresh

Context files go stale. The `refresh` command detects what changed and surgically updates:

```bash
tribal-knowledge refresh ./my-project
```

```
Commits since last parse     14
Files changed                6 / 42
Stale modules                1 (auth)
Broken references            3

Action: Partial re-parse
  Re-generated: auth.md
  Fixed refs:   3
  Quality:      0.89 (was 0.87)
  Time:         12.4s
```

---

## Architecture

```
src/tribal_knowledge/
├── parser/              # Tree-sitter AST extraction (6 languages)
├── graph/               # NetworkX knowledge graph + PageRank + Louvain
├── embeddings/          # Voyage Code 3 + ChromaDB vector store
├── pipeline/            # LangGraph 5-agent generation pipeline
│   └── agents/          #   Explorer, Analyst, Writer, Critic, Fixer, Synthesizer
├── evaluation/          # 4-dimension scoring + LLM-as-judge + gap detection
├── refresh/             # Staleness detection + selective re-parse
├── benchmark/           # Multi-repo benchmark runner
└── cli.py               # Typer CLI with Rich output
```

**47 Python modules | 12,170 lines | 235 tests**

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| AST Parsing | tree-sitter + language-pack | 6 languages, <1ms/file, battle-tested |
| Knowledge Graph | NetworkX | Zero infrastructure, built-in PageRank + Louvain |
| Embeddings | Voyage Code 3 + ChromaDB | Best code embeddings + zero-infra vector store |
| Agent Orchestration | LangGraph | State machine with cycles, conditional routing |
| Primary LLM | Claude Sonnet | Strongest reasoning for code analysis |
| Routing LLM | Gemini Flash | $0.075/1M tokens, 1M context window |
| CLI | Typer + Rich | Type-safe CLI with beautiful terminal output |

---

## Cost

Typical run on a 5-10 module codebase:

| Agent | Model | Cost |
|-------|-------|------|
| Explorer | Gemini Flash | ~$0.01 |
| Analyst | Claude Sonnet | ~$0.30 |
| Writer + Critic + Fixer | Claude Sonnet | ~$0.60 |
| Synthesis | Claude Sonnet | ~$0.09 |
| **Total** | | **< $2.00** |

---

## Development

```bash
git clone https://github.com/abhishekgusain07/tribal-knowledge
cd tribal-knowledge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

<p align="center">
  Built on <a href="https://engineering.fb.com/">Meta's codebase context approach</a> &bull;
  Powered by <a href="https://www.anthropic.com/claude">Claude</a> + <a href="https://ai.google.dev/">Gemini</a> + <a href="https://www.trychroma.com/">ChromaDB</a> + <a href="https://tree-sitter.github.io/">Tree-sitter</a>
</p>
