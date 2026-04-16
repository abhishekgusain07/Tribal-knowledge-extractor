# Desktop Application

## Quick Commands
- `npm run dev` - Start desktop app in development mode with hot reload
- `npm run build:electron` - Build production desktop app for current platform
- `npm run test:electron` - Run desktop-specific test suite

## Key Files
- `electron-launcher.mjs` - Entry point that launches the main Electron process
- `backendReadiness.ts` - Monitors backend server readiness before showing UI
- `updateChannels.ts` - Determines auto-update release channel (stable/beta)
- `runtimeArch.ts` - Cross-platform runtime detection and backend binary selection
- `syncShellEnvironment.ts` - Shell environment synchronization for backend dependencies
- `backendPort.ts` - Backend server port management and conflict detection

## Non-Obvious Patterns
- **Backend Server Lifecycle Management**: Complex coordination between process spawning, port detection, and health checking - backend must be fully ready before UI loads or you get blank screens
- **Secure Renderer-Main Communication**: Preload scripts with `contextIsolation` safely expose main process APIs without compromising security sandbox - breaks easily with Electron updates
- **Auto-Update State Machine**: Complex state management for download/install/restart cycle - update channel switching requires full app restart, can't change mid-session
- **Cross-Platform Runtime Detection**: Backend server binary selection and environment setup varies significantly across macOS/Windows/Linux - port detection timing differs by platform
- Shell environment sync critical for backend finding dependencies - platform-specific propagation issues cause silent failures
- Port conflicts cause silent failures - occupied ports not handled gracefully in backend startup

## See Also
- **Backend Server** - Desktop app spawns and manages the backend process lifecycle
- **Web Frontend** - Served by backend and loaded in Electron renderer process
- **CI/CD Pipelines** - Electron packaging and distribution workflows