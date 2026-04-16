# Cursor IDE — Codebase Context

## What This Is
AI-powered code editor built on Electron, providing intelligent code completion and chat-based programming assistance for developers.

## Tech Stack
Electron + React frontend + Node.js backend + Astro marketing site + GitHub Actions CI/CD.

## How It Works (30-Second Version)
User opens the desktop app, which spawns a local Node.js backend server. The React frontend connects via WebSocket for real-time AI chat and code operations. Git integration enables context-aware suggestions. Authentication flows through dedicated services while analytics track usage patterns.

## Module Map
- **Desktop Application** - Electron wrapper managing backend lifecycle and cross-platform compatibility (see desktop-application.md)
- **Web Frontend** - React UI with logic-view separation, browser-specific components, and WebSocket real-time sync (see web-frontend.md)
- **Backend Server** - Node.js API with layers/services architecture, distributed tracing, and provider abstractions (see backend-server.md)
- **Marketing Website** - Independent Astro static site for product information and downloads (see marketing-website.md)
- **CI/CD Pipelines** - GitHub Actions for automated testing, releases, and issue management (see ci-cd-pipelines.md)

## Key Decisions
**Backend-First Architecture**: Desktop app spawns backend server before UI loads, ensuring all services are ready and preventing blank screen issues during startup.

**Logic-View Separation**: Frontend business logic lives in `.logic.ts` files separate from React components, enabling comprehensive testing without DOM rendering complexity.