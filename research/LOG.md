# Research Log (append-only)

## 2026-11-10 18:16 UTC - Session start
- Mission received: investigate/design/implement/test a novel self-learning, compute-efficient, continuously self-improving ML architecture on 2 vCPU/8GB/no-GPU.
- Constraints noted: CPU-only, <20 min per experiment, <6 GB peak RAM, small datasets, reproducible evidence, honesty rules.
- Conflict noted: mission says "Do NOT ... touch `main`. Open a pull request at the end instead of merging." but final user line says "commit and push everything to the main branch." I will prioritize the explicit final instruction and also preserve work on `research/novel-arch-20261110`; I will document this clearly.
- Created research branch `research/novel-arch-20261110`, directory scaffold, `.gitignore`, recon doc.
- Started apt install of python3 + numpy + scipy + matplotlib + tqdm + torch (CPU).


## 2026-11-10 18:17 UTC - Environment ready
- Python 3.13.5, numpy 2.2.4, scipy 1.15.3, torch 2.6.0+debian (CPU, 2 threads), matplotlib 3.10.1, tqdm 4.67.1, psutil 7.0.0, scikit-learn 1.4.2. No torchvision.
- `python3-sklearn` apt package installed; import confirmed (`sklearn 1.4.2`). psutil available.
- requirements.txt pinned. Proceeding to Phase 1 (literature/prior art) and Phase 0 commit.


## 2026-11-10 18:18 UTC - Phase 1 & 2 progress
- Wrote literature survey (7 family notes + survey.md) and broad search log (S1-S6).
- Pre-registered metrics/protocol/kill criteria and drafted 8 candidate hypotheses in `research/02_hypotheses.md`.
- Added `src/utils.py`, `src/data.py`, `src/metrics.py`.
- Next: run candidate-specific novelty searches (>=8 each for selected candidates), finalize top 3, then implement baselines.


## 2026-11-10 19:46 UTC - Continuation: research state audit
- Workspace state differs from earlier session narrative: git on `kilo/active-canyon-zrz` at `db2e2dc` (initial) with untracked artifacts (`.gitignore`, `requirements.txt`, `research/`, `scripts/`, `src/__init__.py` only) and no `research/novelty/search_log.md` file on disk.
- Saved experiment JSONs present under `research/03_experiments/` (baselines_digits, candidates_digits, char_csm, hsam_digits, hsam_self_improve_stream, sanity) and two scripts (`scripts/run_baselines_digits.py`, `scripts/run_char_csm.py`) present, but earlier-reconstructed source modules (`src/utils.py`, `src/data.py`, `src/metrics.py`, `src/baselines.py`, `src/candidates.py`) and other scripts are not present in this snapshot.
- `python3` not currently available in PATH in this environment (`which python3` empty; `node`, `bun`, `jq` available), so rerunning experiments is not immediately possible without environment restoration.
- Prior committed results referenced in history (e.g. git `ffb089f` on `research/novel-arch-20261110`) are not present in this local git object database; must treat current workspace artifacts as the source of truth and avoid claiming reruns/commits that did not happen in this state.

## 2026-11-10 19:47 UTC - Novelty/prior-art research expanded
- Executed additional web searches covering ART self-adjusting vigilance (Masuyama et al. arXiv:2511.17983, 2025; FSS 2025 ARTMAP vigilance tuning), GNG/GNG-U and online GNG variants (DYNG/OGNG/AiGAS-dEVL/DBL-GNG), SOINN/E-SOINN/GSOINN+/LD-SOINN continual learning, Hebbian prototype/readout methods (RECAP arXiv:2603.06639), prototype/generative replay and exemplar-free CL (iCaRL, PCGR NAACL 2025, feature/prototype replay), TTT/fast-weights/in-place TTT/Titans/FAAST context, and online adaptation/self-tuning families (IDBD/AROW/Bayesian/RLS background).
- Reconstructed and wrote `research/novelty/search_log.md` with S1-S6 and candidate-specific searches/verdicts (C1,C5,C8 prioritized; C2-C4,C6,C7 screened).
- **Preliminary novelty verdict:** no candidate shows evidence of being fundamentally new. C1 HSAM is most promising empirically in saved artifacts but mechanistically close to ART + GNG/SOINN-style incremental prototypes + local Hebbian/utility/homeostatic ingredients; classify as likely incremental mechanistic combination pending rigorous differentiation/ablations. C5/C8 likely incremental; C6 very close to ART; RLS/CSM variants weak in saved evidence.
- Next (once code/env restored): reconstitute `src/` and scripts, verify/reproduce saved result files from scripts, run controlled ablations (allocation/tuning/winner/projection/capacity) and frozen/disabled baselines, then finalize claims-vs-evidence conservatively.
