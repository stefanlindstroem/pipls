# Decision 0091: create snapshots from a clean committed Git tree

## Status

Accepted.

## Context

The snapshot helper copied the complete worktree with `rsync` and determined cleanliness from
tracked staged and unstaged differences only. Nonignored untracked files could therefore enter an
archive while `SNAPSHOT_INFO` reported `dirty: false`. Ignored generated documentation figures
could also enter snapshots because exclusion depended on a manually maintained `rsync` list.

Repository snapshots are authoritative inputs for later patch work. Their contents and recorded
commit must therefore agree without relying on local cleanup conventions.

## Decision

Create snapshots from `git archive HEAD`, not from a worktree copy.

1. Snapshot creation requires a Git worktree and a valid `HEAD` commit.
2. The helper refuses to run when `git status --porcelain` reports tracked changes, staged changes,
   or nonignored untracked files.
3. Ignored files do not block snapshot creation because they cannot enter `git archive` output.
4. `SNAPSHOT_INFO` is regenerated after extracting the committed tree and always records
   `dirty: false`.
5. Archive paths remain relative to the repository root, and the generated metadata replaces the
   committed handoff metadata inside the archive.
6. Relative output paths are resolved against the caller's current directory before temporary
   staging begins.

Tests exercise clean, modified, staged, nonignored-untracked, and ignored-file cases in isolated
Git repositories. CI no longer installs or depends on `rsync` for snapshot creation.

## Consequences

A successful snapshot is a faithful committed-tree handoff for one recorded commit. Local caches,
generated tutorial figures, ignored results, and other ignored artifacts are excluded by
construction. Contributors must commit or remove every nonignored change before running
`make snapshot`; the helper no longer produces diagnostic dirty snapshots.
