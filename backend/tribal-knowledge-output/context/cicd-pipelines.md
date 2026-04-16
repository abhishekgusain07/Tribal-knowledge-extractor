# CI/CD Pipelines

## Quick Commands
- Trigger release: Push git tag matching version pattern to main branch
- Debug workflow failure: Check `.github/workflows/{workflow}.yml` and Actions tab for specific job logs
- Add new CI check: Modify `ci.yml` jobs section and update branch protection rules
- Override PR size limit: Add `pr-vouch.yml` approval comment from maintainer

## Key Files
- `.github/workflows/ci.yml` - Main build/test pipeline triggered on pushes and PRs
- `.github/workflows/release.yml` - Automated release creation and artifact publishing
- `.github/workflows/pr-size.yml` - PR size validation and blocking for large changes
- `.github/workflows/pr-vouch.yml` - Maintainer override system for PR requirements
- `.github/workflows/issue-labeler.yml` - Automatic issue categorization and triage

## Non-Obvious Patterns
- Release workflow only triggers on version tags, not manual dispatch - check tag format requirements in `release.yml`
- PR workflows run with restricted permissions for fork contributions - secrets unavailable for external contributors
- Workflow dependencies use `needs:` keyword - `release.yml` waits for `ci.yml` success before deploying
- Branch protection bypasses require specific status check names - must match job IDs in workflow files exactly
- Issue labeler uses path-based rules - modifying file locations breaks automatic categorization in `issue-labeler.yml`
- Concurrent releases are prevented through GitHub's tag creation atomicity, not explicit workflow locking

## See Also
- **Backend Server** - CI pipeline builds and tests server components before release
- **Desktop Application** - Release workflow packages and distributes desktop builds
- **Web Frontend** - CI validates frontend builds and deploys to staging environments