import { Link } from '@tanstack/react-router'
import ThemeToggle from './ThemeToggle'

export default function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-[var(--border)] bg-[var(--bg)] px-4 backdrop-blur-sm">
      <nav className="page-wrap flex items-center justify-between py-3.5">
        <Link
          to="/"
          className="text-[15px] font-semibold tracking-tight text-[var(--ink)] no-underline"
        >
          tribal knowledge
        </Link>
        <ThemeToggle />
      </nav>
    </header>
  )
}
