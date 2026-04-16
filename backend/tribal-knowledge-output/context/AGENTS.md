# Cursor IDE

## Overview
AI-powered code editor built on Electron, providing intelligent code completion and chat-based programming assistance. Desktop app spawns local Node.js backend server with React frontend connecting via WebSocket for real-time AI operations.

## Architecture
- **Desktop Application** - Electron wrapper managing backend lifecycle and cross-platform compatibility
- **Web Frontend** - React UI with logic-view separation, browser-specific components, and WebSocket real-time sync
- **Backend Server** - Node.js API with layers/services architecture, distributed tracing, and provider abstractions
- **Marketing Website** - Independent Astro static site for product information and downloads
- **CI/CD Pipelines** - GitHub Actions for automated testing, releases, and issue management

## Key Conventions
- Business logic extracted to .logic.ts files separate from React components for comprehensive testing
- All main process modules export typed APIs for preload script consumption
- Backend communication requires health check pattern before UI load
- Browser-specific implementations use .browser.tsx suffix and conditional loading
- Error types centralized in domain-specific Errors.ts files
- Component state hierarchy: local state → custom hooks → React Query → WebSocket updates
- All domains follow Layers/Services pattern for dependency injection
- Provider implementations extend makeManagedServerProvider pattern
- React Query hooks centralized in lib/*ReactQuery.ts files with consistent naming

## Gotchas
- Backend server must be fully ready before renderer loads - race conditions cause blank screens
- Shell environment sync critical on macOS for PATH inheritance from GUI vs terminal
- Electron security context isolation means renderer cannot directly access node APIs
- React Query cache keys must follow specific patterns or git operations show stale data
- Diff operations must use worker pool pattern to prevent UI blocking
- Update channel switching requires app restart - cannot change channels mid-session
- TraceSink intentionally drops invalid records during serialization to prevent cascade failures