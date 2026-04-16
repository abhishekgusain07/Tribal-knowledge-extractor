# Marketing Website

## Quick Commands
- `cd apps/marketing && npm run dev` to start local development server
- Update release info in `lib/releases.ts` then rebuild for new version announcements
- Replace `public/screenshot.jpeg` when app UI changes significantly
- Deploy independently via `npm run build` without touching main application

## Key Files
- `src/pages/index.astro` - Landing page content
- `src/pages/download.astro` - Download page pulling from releases.ts
- `src/layouts/Layout.astro` - Shared wrapper for all marketing pages
- `lib/releases.ts` - Single source of truth for version/release data
- `public/screenshot.jpeg` - Hero image requiring manual updates

## Non-Obvious Patterns
- Zero cross-module dependencies enable independent deployment cycles from main app
- Layout.astro changes affect all pages simultaneously - test thoroughly before deploying
- Favicon files in public/ must maintain exact naming/dimensions for mobile icons to work
- releases.ts must be updated synchronously with actual app releases to avoid stale download links
- Screenshot.jpeg becomes misleading if not updated when app UI changes significantly
- Astro's static generation means no server-side logic - all dynamic content comes from build-time data

## See Also
- Release Information Centralization pattern for version management
- Astro Static Generation Pattern for component architecture
- Isolated Marketing Site Pattern for deployment independence