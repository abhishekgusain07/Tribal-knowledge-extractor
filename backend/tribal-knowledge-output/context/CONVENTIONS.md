# Conventions

## Import Aliases
- All main process modules export typed APIs for preload script consumption
- Backend communication goes through health check pattern before UI load
- All main process modules use .ts extension, scripts use .mjs

## File Naming
- Test files follow .test.ts naming convention with comprehensive coverage
- Business logic extracted to .logic.ts files with corresponding .logic.test.ts tests
- Browser-specific implementations use .browser.tsx suffix and are conditionally loaded
- Integration tests use .integration.ts suffix in integration/ directory
- Migration files follow timestamp naming in persistence/Migrations/

## Error Handling
- Error types are centralized in domain-specific Errors.ts files
- Confirmation dialogs use native Electron dialogs via confirmDialog.ts
- TraceSink intentionally drops invalid records during serialization to prevent cascade failures

## State Management
- Component state follows: local state → custom hooks → React Query → WebSocket updates
- React Query hooks centralized in lib/*ReactQuery.ts files with consistent naming
- Settings use clientPersistence.ts abstraction rather than direct file I/O

## Architecture Patterns
- All domains follow Layers/Services pattern for dependency injection
- Provider implementations extend makeManagedServerProvider pattern
- Diff operations must use worker pool pattern to prevent UI blocking
- Environment-specific code isolated in src/environments/ with clear interface boundaries

## Gotchas
- Backend server must be fully ready before renderer loads - race conditions cause blank screens
- Shell environment sync critical on macOS for PATH inheritance from GUI vs terminal
- Electron security context isolation means renderer cannot directly access node APIs
- React Query cache keys must follow specific patterns or git operations show stale data
- Update channel switching requires app restart - can't change channels mid-session