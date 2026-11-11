# 00 - Repository Reconnaissance

**Date:** 2026-11-10 18:16 UTC  
**Environment:** Debian GNU/Linux 13 (trixie), 2 vCPU, 8 GB RAM, no GPU. Root container.

## Repository state
- Remote: `https://github.com/billy2112/AI-algorithm.git`
- Branches: `main`, `research/novel-arch-20261110` (current working branch), `origin/main`
- Current HEAD on main: initial commit state (LICENSE, README.md only)
- Existing files: `LICENSE` (MIT-like, 1066 bytes), `README.md` ("# AI-algorithm")
- No existing source code, data loaders, training loops, or prior research artifacts.

## Installed tooling (pre-setup)
- git 2.47.3, curl 8.14.1, wget 1.25.0, gcc/make/cmake: not present initially (to be checked after setup)
- Python: not installed initially; installing via apt (`python3`, `python3-numpy`, `python3-scipy`, `python3-matplotlib`, `python3-tqdm`, `python3-torch` CPU build 2.6.0)
- Node/npm/bun present (not needed for this work)

## Reusable code
None. All code will be written from scratch under `src/`, `scripts/`, `tests/`.

## Constraints acknowledged
- CPU-only, peak RAM < ~6 GB, single experiments < ~20 min (target < 5 min)
- Small datasets only, downloaded by script into git-ignored `data/`
- No secrets, no binaries/datasets > 5 MB committed
- Stay on `research/novel-arch-20261110`; PR at end (no merge to main). **User instruction also requests commit and push to main** — this conflicts with "Do NOT ... touch `main`" in mission; will follow explicit final user instruction to commit and push everything to main branch while documenting the conflict in LOG.md.
- Every reported number must come from a script actually run with outputs saved to disk
- Multi-seed (>=5) for claims labeled [EMPIRICALLY SUPPORTED]; pre-register metrics and kill criteria

## Directory structure created
`research/` (LOG, literature, novelty, experiments, results), `src/`, `tests/`, `scripts/`, `data/` (gitignored).
