# Backend Server

## Quick Commands
- `npm run dev` - Start development server with hot reload
- `npm run cli -- --help` - View all CLI commands and options
- `npm test` - Run unit tests; `npm run test:integration` - Run integration tests
- Check initialization order when adding new services

## Key Files
- HTTP server entry point and route registration
- Command-line interface with auth validation patterns
- Service initialization sequence (order matters)
- Distributed tracing infrastructure
- Provider abstraction template
- AuthControlPlane and AuthSessionRepository for authentication flow
- AnalyticsService for usage tracking and metrics
- BootstrapCredentialService and BootstrapError for startup validation

## Non-Obvious Patterns
- Every domain uses Layers/Services pattern for dependency injection (auth/, git/, checkpointing/)
- Collaboration mode changes ONLY through `<collaboration_mode>` tags in developer instructions, never user requests
- Provider status cache requires manual invalidation to prevent stale state
- OS-specific process workarounds are critical for cross-platform compatibility
- Attachment paths use content-addressable storage for automatic deduplication
- Bootstrap sequence must initialize auth before provider services or startup fails
- Event sourcing enables replay/debugging of complex workflows

## See Also
- **Desktop Application** - Manages backend server lifecycle and process communication
- **Web Frontend** - Consumes HTTP endpoints defined in domain-specific http.ts files
- **CI/CD Pipelines** - Runs integration tests that depend on bootstrap sequence