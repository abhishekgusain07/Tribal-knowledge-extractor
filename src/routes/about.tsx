import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/about')({ component: About })

function About() {
  return (
    <main className="page-wrap px-4 py-20">
      <h1 className="mb-4 font-serif text-2xl font-medium text-[var(--ink)]">
        About
      </h1>
      <p className="max-w-[480px] text-[15px] leading-relaxed text-[var(--ink-2)]">
        Tribal Knowledge is an open-source tool that extracts the undocumented
        context from any codebase — the patterns, conventions, and decisions
        that live only in senior engineers' heads — and turns it into concise,
        useful context files for AI tools and humans alike.
      </p>
    </main>
  )
}
