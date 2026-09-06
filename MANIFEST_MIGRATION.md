# Manifest migration — dropped git submodule

Prior submodule at `ggml` is now populated via the weftspun-keypoint manifest as a repo project. Direct-clone consumers (outside `repo sync`) should check the manifest for the pinned SHA and clone that path manually.

Migrated 2026-09-06 as part of the workspace-wide submodule→manifest sweep (CLAUDE.md blocklist row on git submodules).
