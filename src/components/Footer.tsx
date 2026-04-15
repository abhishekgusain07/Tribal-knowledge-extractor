export default function Footer() {
  return (
    <footer className="border-t border-[var(--border)] px-4 py-8">
      <div className="page-wrap flex flex-col items-center justify-between gap-3 text-xs text-[var(--ink-3)] sm:flex-row">
        <p className="m-0">Open source. Built for engineers who hate stale docs.</p>
        <p className="m-0">
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="text-[var(--ink-3)] no-underline hover:text-[var(--ink-2)]"
          >
            GitHub
          </a>
        </p>
      </div>
    </footer>
  )
}
