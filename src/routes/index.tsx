import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/')({ component: Home })

function Home() {
  return (
    <main className="page-wrap px-4 pb-20">
      {/* Hero */}
      <section className="rise-in pt-20 pb-16 sm:pt-28 sm:pb-20">
        <span className="mb-6 inline-block rounded-full border border-[var(--pill-border)] bg-[var(--pill-bg)] px-3 py-1 text-[11px] font-semibold tracking-[0.08em] uppercase text-[var(--pill-text)]">
          Coming Soon
        </span>
        <h1 className="mb-5 max-w-[620px] font-serif text-[clamp(2.2rem,5.5vw,3.5rem)] leading-[1.08] font-medium tracking-tight text-[var(--ink)]">
          Your codebase has tribal knowledge.{' '}
          <span className="text-[var(--ink-3)]">We extract it.</span>
        </h1>
        <p className="mb-8 max-w-[480px] text-base leading-relaxed text-[var(--ink-2)]">
          Point it at any GitHub repo. Get concise context files that make AI
          tools — and new engineers — actually understand the code.
        </p>
        <div className="flex flex-wrap gap-3">
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-md border border-[var(--border-strong)] bg-[var(--bg-raised)] px-4 py-2 text-sm font-medium text-[var(--ink)] no-underline transition-colors hover:border-[var(--accent-border)] hover:text-[var(--accent)]"
          >
            <svg
              viewBox="0 0 16 16"
              width="16"
              height="16"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
            </svg>
            Star on GitHub
          </a>
        </div>
      </section>

      {/* Terminal */}
      <section
        className="rise-in overflow-hidden rounded-lg border border-[var(--border-strong)] shadow-[0_2px_20px_rgba(0,0,0,0.06)]"
        style={{ animationDelay: '120ms' }}
      >
        <div className="flex items-center gap-1.5 border-b border-[rgba(255,255,255,0.06)] bg-[#1e1d1b] px-4 py-2.5">
          <span className="h-2.5 w-2.5 rounded-full bg-[#ec6a5e]" />
          <span className="h-2.5 w-2.5 rounded-full bg-[#f4bf4f]" />
          <span className="h-2.5 w-2.5 rounded-full bg-[#61c554]" />
          <span className="ml-2 text-[11px] text-[#6b6860]">terminal</span>
        </div>
        <div className="overflow-x-auto bg-[#1a1918] px-5 py-5 font-mono text-[13px] leading-[1.7] text-[#c8c4b8] sm:text-sm">
          <pre className="m-0 whitespace-pre">
            <span className="text-[#8b8780]">$</span>{' '}
            <span className="text-[#e7e5e0]">
              tribal-knowledge
            </span>{' '}
            <span className="text-[#8b8780]">
              https://github.com/acme/webapp
            </span>
            {'\n\n'}
            <span className="text-[#8b8780]">
              Parsing 847 files across 3 languages...
            </span>
            {'\n'}
            <span className="text-[#8b8780]">
              Building knowledge graph
            </span>
            <span className="text-[#6b6860]">
              {' ··· '}
            </span>
            <span className="text-[#c8c4b8]">
              2,341 nodes · 8,102 edges
            </span>
            {'\n'}
            <span className="text-[#8b8780]">
              Running PageRank
            </span>
            <span className="text-[#6b6860]">
              {' ··· '}
            </span>
            <span className="text-[#c8c4b8]">
              top 20 entities identified
            </span>
            {'\n\n'}
            <span className="text-[#e7e5e0]">
              Generating with 5-agent pipeline
            </span>
            {'\n\n'}
            {'  '}
            <span className="text-[#7ec699]">{'✓'}</span>
            {'  Explorer    '}
            <span className="text-[#8b8780]">
              identified 6 modules
            </span>
            {'\n'}
            {'  '}
            <span className="text-[#7ec699]">{'✓'}</span>
            {'  Analyst     '}
            <span className="text-[#8b8780]">
              extracted tribal knowledge
            </span>
            {'\n'}
            {'  '}
            <span className="text-[#7ec699]">{'✓'}</span>
            {'  Writer      '}
            <span className="text-[#8b8780]">
              produced compass docs
            </span>
            {'\n'}
            {'  '}
            <span className="text-[#7ec699]">{'✓'}</span>
            {'  Critic      '}
            <span className="text-[#8b8780]">
              quality score
            </span>
            {' '}
            <span className="text-[#e7e5e0]">0.91</span>
            {'\n\n'}
            <span className="text-[#e7e5e0]">
              Generated 5 context files
            </span>
            <span className="text-[#6b6860]"> → </span>
            <span className="text-[#8b8780]">./context/</span>
            {'\n\n'}
            {'  '}
            <span className="text-[#c8c4b8]">OVERVIEW.md</span>
            {'       '}
            <span className="text-[#6b6860]">
              project-level compass
            </span>
            {'\n'}
            {'  '}
            <span className="text-[#c8c4b8]">api-layer.md</span>
            {'      '}
            <span className="text-[#6b6860]">
              routing, middleware, auth
            </span>
            {'\n'}
            {'  '}
            <span className="text-[#c8c4b8]">data-model.md</span>
            {'     '}
            <span className="text-[#6b6860]">
              schemas, relationships
            </span>
            {'\n'}
            {'  '}
            <span className="text-[#c8c4b8]">auth-flow.md</span>
            {'      '}
            <span className="text-[#6b6860]">
              authentication lifecycle
            </span>
            {'\n'}
            {'  '}
            <span className="text-[#c8c4b8]">CONVENTIONS.md</span>
            {'    '}
            <span className="text-[#6b6860]">
              patterns, naming, gotchas
            </span>
            {'\n\n'}
            <span className="text-[#7ec699]">Accuracy</span>
            <span className="text-[#e7e5e0]"> 0.91</span>
            <span className="text-[#6b6860]"> · </span>
            <span className="text-[#7ec699]">Coverage</span>
            <span className="text-[#e7e5e0]"> 0.87</span>
            <span className="text-[#6b6860]"> · </span>
            <span className="text-[#7ec699]">Conciseness</span>
            <span className="text-[#e7e5e0]"> 0.89</span>
            {'\n'}
            <span className="text-[#8b8780]">
              Cost $0.43 · Time 4m 12s
            </span>
          </pre>
        </div>
      </section>

      {/* Value props */}
      <section className="mt-16 grid gap-px overflow-hidden rounded-lg border border-[var(--border-strong)] bg-[var(--border-strong)] sm:grid-cols-3">
        {[
          {
            title: 'Compass, not encyclopedia',
            desc: '25–35 lines per file. Only what you can\'t figure out in 60 seconds of reading the code.',
          },
          {
            title: 'Multi-agent pipeline',
            desc: 'Five agents — explore, analyze, write, critique, fix — iterate until quality hits 0.85.',
          },
          {
            title: 'Works everywhere',
            desc: 'AGENTS.md-compatible output. Claude Code, Cursor, Copilot, Gemini CLI, and more.',
          },
        ].map((item, i) => (
          <article
            key={item.title}
            className="rise-in bg-[var(--bg-raised)] p-6"
            style={{ animationDelay: `${240 + i * 80}ms` }}
          >
            <h3 className="mb-2 text-sm font-semibold text-[var(--ink)]">
              {item.title}
            </h3>
            <p className="m-0 text-[13px] leading-relaxed text-[var(--ink-3)]">
              {item.desc}
            </p>
          </article>
        ))}
      </section>

      {/* The problem */}
      <section
        className="rise-in mt-16 rounded-lg border border-[var(--border)] bg-[var(--bg-raised)] p-6 sm:p-8"
        style={{ animationDelay: '400ms' }}
      >
        <p className="mb-4 text-[11px] font-semibold tracking-[0.08em] uppercase text-[var(--ink-3)]">
          The problem
        </p>
        <p className="m-0 max-w-[560px] text-[15px] leading-relaxed text-[var(--ink-2)]">
          Engineers spend 58% of their time reading code, not writing it. When
          senior engineers leave, their mental model of the codebase — the{' '}
          <em>why</em> behind decisions, the non-obvious patterns, the things
          that break silently — leaves with them. No existing tool generates
          this context automatically.{' '}
          <span className="text-[var(--ink-3)]">We're building the one that does.</span>
        </p>
      </section>

      {/* CTA */}
      <section
        className="rise-in mt-16 text-center"
        style={{ animationDelay: '500ms' }}
      >
        <p className="mb-1 font-serif text-xl font-medium text-[var(--ink)]">
          Shipping soon.
        </p>
        <p className="m-0 text-sm text-[var(--ink-3)]">
          Open source, free to use, built for real codebases.
        </p>
      </section>
    </main>
  )
}
