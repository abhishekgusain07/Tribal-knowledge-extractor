# Web Frontend

## Quick Commands
- `npm run dev` - Start development server with hot reload
- `npm run test:logic` - Run tests for .logic.ts files only
- `npm run build:browser` - Build with browser-specific components included
- `npm run storybook` - Launch component development environment

## Key Files
- `ChatMarkdown.browser.tsx` - Browser-specific markdown rendering with SSR handling
- `ChatView.browser.tsx` - Browser-specific chat interface implementation
- `GitActionsControl.browser.tsx` - Browser-specific git operation controls
- `ThreadTerminalDrawer.tsx` - Terminal session management with intentional dependency omissions
- `WebSocketConnectionSurface.tsx` - Real-time connection handling without exposed state
- `lib/*ReactQuery.ts` - Git operation state management and cache invalidation
- `mockServiceWorker.js` - Auto-generated MSW file with disabled eslint

## Non-Obvious Patterns
- Logic-view separation: Business logic in `.logic.ts` files enables testing without React rendering
- Browser-specific loading: `.browser.tsx` files handle SSR vs client differences through conditional imports
- Environment isolation: `src/environments/` contains runtime-specific implementations for local/remote/runtime contexts
- React Query cache keys must follow git operation patterns or stale data appears after commits
- WebSocket failures cause silent desync - UI works but changes aren't saved
- Command registration order determines keyboard shortcut priority conflicts
- Worker pool exhaustion freezes UI when processing too many large diffs simultaneously

## See Also
- **Backend API** - React Query hooks depend on consistent API contracts
- **Git Operations** - Browser-specific git controls require git service availability
- **WebSocket Service** - Real-time message handling for state synchronization
- **Desktop Integration** - Desktop update mechanisms for cross-platform compatibility
- **Terminal Service** - Terminal session management for ThreadTerminalDrawer.tsx