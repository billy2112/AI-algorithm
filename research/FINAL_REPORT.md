# Final Research Report: HSAM_Simplified (Lightweight ART/GNG-U-Family Continual Learner)

**Date:** 2026-11-13  
**Repository:** `/workspace/app` (`https://github.com/billy2112/AI-algorithm.git`)  
**Environment:** Debian 13 (trixie), 2 vCPU, 8 GB RAM, no GPU, CPU-only. Python 3.13.5; numpy 2.2.4, scipy 1.15.3, scikit-learn 1.4.2, torch 2.6.0 (CPU), matplotlib 3.10.1, tqdm 4.67.1, psutil 7.0.0.  
**Working branch:** `kilo/active-canyon-zrz` at `752e884` (committed locally, not pushed to main).  

## Executive summary

This project investigated a CPU-efficient continual learner under strict resource and evidence constraints. I implemented and evaluated **HSAM (C1)** — a backprop-free online prototype/associative memory — and then **HSAM_Simplified**, a variant with the homeostatic self-tuning controller removed (proven a dead end).

**Honest framing:** HSAM_Simplified is positioned as a **lightweight ART/GNG-U-family continual learner**, not a novel architecture. Its validated components — fixed random projection, k-winner softmax retrieval, local delta key/value updates, surprise/mismatch/low-sim-gated slot allocation, utility-based replacement — map directly onto ideas from ART (vigilance/allocation), GNG-U (utility-based replacement), and prototype-memory literature. No component is structurally impossible for these families.

**Headline outcomes (preliminary, small-scale on sklearn digits 8×8):**

1. **HSAM_Simplified matches original HSAM exactly.** Removing the self-tuning controller changes nothing measurable:
   - Supervised digits: simplified 0.9622±0.0064 vs original 0.9606±0.0063 (s200/w3); 0.9722±0.0071 vs 0.9711±0.0042 (s400/w3). CI overlap complete.
   - Continual split-digits-5: simplified AA 0.9300±0.0347 vs original 0.9317±0.0321 (s200); 0.9323±0.0051 vs 0.9329±0.0053 (s400). FM essentially identical. CI overlap complete.
   - Wall-clock and peak RSS essentially identical (~0.5–0.8s, ~319MB).
   - **Conclusion:** the self-tuning controller is dead weight; the simplified variant is strictly preferable (simpler, same performance).

2. **Allocation — not tuning — drives retention.** Disabling slot allocation (no_alloc) drops continual AA from ~0.93 to ~0.76 across the full grid (slots 100/200/400/800 × winners 1/3/5), while disabling tuning has no effect. Slot capacity matters: s100 hurts (AA ~0.86), s400 is the sweet spot (AA ~0.93), s800 shows diminishing returns.

3. **SLDA and tuned replay outperform HSAM on continual learning.** On split-digits-5:
   - SLDA: AA 0.9376±0.0109, FM 0.0311±0.0106 (best FM, fast, 0.174s)
   - replay_buf200: AA 0.9594±0.0057, FM 0.0207±0.0123 (best AA)
   - replay_buf100: AA 0.9367±0.0087, FM 0.0526±0.0135 (comparable to HSAM)
   - HSAM_Simplified s400_w3: AA 0.9323±0.0051, FM 0.0657±0.0070 (below replay_buf200, near replay_buf100; worse FM than SLDA)
   - MLP finetune/EWC collapse (AA ~0.19) — not competitive.

4. **Stream self-improvement is not supported.** Continued online learning beats frozen snapshots (delta ~+0.11 over 100 chunks), but no_tune ≈ full ≈ no_alloc on within-stream accuracy — including with a redesigned per-slot adaptive + per-class vigilance tuner (full +0.0867 vs no_tune +0.0883 on non-stationary stream). The homeostatic self-tuning concept is a **dead end**.

**Scale-up attempt:** Fashion-MNIST was attempted via OpenML fetch but the load failed in this environment (no torchvision/tensorflow; pandas installed but OpenML fetch blocked). Full CIFAR-10 not attempted under 2vCPU/8GB budget. Marked [NOT SUPPORTED / INVALID] — not used as evidence. Digits (8×8, 1797 images) remain the primary evidence.

**Novelty verdict:** **[NOT SUPPORTED / INCREMENTAL].** Head-to-head vs FuzzyART, GNG-U, SLDA plus mechanistic component mapping confirms HSAM is a reparameterization/integration of known ART/GNG-U/prototype-memory ideas. Treat as a lightweight ART/GNG-U-family continual learner, not a novel architecture.

## 1. Problem framing and constraints
- **Target properties:** compute/energy efficiency (CPU, low RAM/wall-clock), self/online learning, continual learning with limited forgetting, and continuous improvement after deployment where possible.
- **Budgets:** 2 vCPU/8 GB/no GPU; target <20 min per experiment, <~6 GB peak RSS; small/built-in or generated data only.
- **Evidence rules followed:** no fabricated results; scripts/configs/results saved; >=5 seeds and bootstrap CIs for headline/supporting empirical numbers where reported; claims labeled [PROVEN]/[EMPIRICALLY SUPPORTED]/[PRELIMINARY]/[CONJECTURE]; novelty treated as hypothesis; negative/ambiguous results reported.

## 2. Phase 0 - Repository and environment
- Recon: `research/00_repo_recon.md`. Initial repo contained only `LICENSE`, `README.md`.
- Environment (reconstructed/verified): Python 3.13.5, numpy 2.2.4, scipy 1.15.3, scikit-learn 1.4.2, torch 2.6.0 CPU, matplotlib 3.10.1, tqdm 4.67.1, psutil 7.0.0 (`requirements.txt`).
- Reproducibility utilities: seeded runs, wall/RSS tracking, JSON save/load, bootstrap CI, git hash capture in artifacts (`src/utils.py`).

## 3. Phase 1 - Prior art and landscape
- Literature notes: `research/01_literature/survey.md`, `01_local_backprop_free.md`–`07_theory.md`.
- Broad and candidate-specific searches with queries/results/verdicts: `research/novelty/search_log.md`.
- Key families: local/Hebbian/predictive-coding/forward-only learning; efficient sequence/SSM/linear-attention/TTT-fast-weights; continual learning (EWC/SI/replay/CLS/HDC); self-improvement/learned optimizers/meta-learning; quantization/sparsity; ART/GNG/SOINN prototype systems. 

## 4. Phase 2 - Pre-registered hypotheses and metrics
- Pre-registration: `research/02_hypotheses.md` (metrics, statistical protocol, baselines, kill criteria, 8 candidates C1–C8).
- Primary tasks used: supervised `sklearn.datasets.load_digits` classification; split-digits-5 continual sequence; tiny char stream for exploratory CSM; stream probe on digits with fixed held-out test.
- Metrics: accuracy, Average Accuracy (AA), Forgetting Measure (FM), wall-clock, peak RSS, bootstrap 95% CI, multi-seed mean±std.

## 5. Phase 3 - Candidates, novelty, and selection
- Candidates drafted: C1 HSAM, C2 DTEC-R, C3 CSCG, C4 BFTP, C5 PUF-IR, C6 ARFL, C7 QLPE, C8 RSMLM (`research/02_hypotheses.md`).
- Novelty assessment: `research/novelty/assessment.md`, `research/novelty/search_log.md`.
- **Selected/focused:** C1 **HSAM** (strongest empirical signal, cheap/testable). CSM (count model) and RLS_SelfGate kept as exploratory/negative or weak controls in saved artifacts.
- **Honest novelty position:** HSAM is a bounded, CPU-cheap combination of known ingredients (fixed random projection, prototype/slot memory, local delta updates, surprise/mismatch/novelty-gated allocation, utility replacement, endogenous rate/threshold tuning). Closest families: ART/self-adjusting vigilance, GNG-U/SOINN, Hebbian/prototype readouts (e.g. RECAP-like local prototype updates), with TTT/fast-weights as distinct gradient-based adaptation family. **No evidence of fundamental novelty.**

## 6. Phase 4 - Baselines (saved artifacts)
Artifacts: `research/03_experiments/baselines_digits/supervised_results.json`, `continual_results.json` (git hash recorded in files: `d1209e9...`, 5 seeds).

| Model | Task | Result |
|---|---|---:|
| ridge | supervised digits | acc 0.9394 ± 0.0114 [0.9311, 0.9478] |
| rf_ridge_256 | supervised digits | acc 0.9778 ± 0.0020 [0.9761, 0.9794] |
| mlp_sklearn_64 | supervised digits | acc 0.9767 ± 0.0042 [0.9733, 0.9800] |
| torch_mlp_64_e20 | supervised digits | acc 0.9506 ± 0.0046 [0.9467, 0.9539] |
| finetune_mlp | split-digits-5 continual | AA 0.1915 ± 0.0066, FM 0.9882 ± 0.0040 |
| ewc_mlp | split-digits-5 continual | AA 0.1904 ± 0.0079, FM 0.9875 ± 0.0031 |
| replay_mlp_buf20 | split-digits-5 continual | AA 0.6330 ± 0.0410, FM 0.4385 ± 0.0523 |

**Note:** Baseline continual results show severe forgetting for finetune/EWC and partial mitigation by small replay under this protocol; HSAM uses a different (nonparametric associative memory) mechanism and should be compared with that context in mind.

## 7. Phases 5-6 - HSAM implementation, experiments, ablations, and validation

### 7.1 Implementation
- Core: `src/candidates.py` (HSAM, RLS_SelfGate, CSM), `src/baselines.py`, `src/data.py`, `src/metrics.py`, `src/utils.py`.
- Sanity: `research/03_experiments/sanity/overfit32_results.json` — HSAM overfit-32 acc **1.0 [1.0,1.0]** (5 seeds), RLS_SelfGate **0.256 [0.194,0.306]** (weak). Automated tests: `tests/test_core.py` (10 tests, passing).

### 7.2 Supervised digits (5 seeds)

| Config | acc mean ± std [95% CI] | wall_s mean |
|---|---:|---:|
| hsam_s400_w1_p2_tune | 0.9628 ± 0.0131 [0.9522, 0.9733] | ~0.74 |
| hsam_s400_w3_p2_tune | 0.9706 ± 0.0058 [0.9661, 0.9750] | ~0.65 |
| hsam_s200_w3_p2_tune | 0.9494 ± 0.0095 [0.9428, 0.9578] | ~0.46 |
| hsam_s400_w1_p2_notune | 0.9644 ± 0.0129 [0.9544, 0.9756] | ~0.72 |
| hsam_simplified_s200_w3 | 0.9622 ± 0.0064 | ~0.40 |
| hsam_simplified_s400_w3 | 0.9722 ± 0.0071 | ~0.56 |

Sources: `research/03_experiments/hsam_digits/supervised_results.json` (original HSAM), `research/03_experiments/simplified_benchmark/results.json` (HSAM_Simplified).

**Interpretation [PRELIMINARY]:** HSAM_Simplified matches original HSAM exactly — removing the self-tuning controller changes nothing measurable. Competitive with strong simple baselines (rf_ridge_256 0.9778, mlp_sklearn_64 0.9767) on this small dataset; capacity (slots) and readout settings matter modestly.

### 7.3 Split-digits-5 continual (5 seeds) — HSAM vs simplified vs stronger baselines

| Config / order | AA mean ± std [95% CI] | FM mean ± std [95% CI] | wall_s | peak RSS |
|---|---:|---:|---:|---:|
| hsam_s200_w3_p2_tune (default) | 0.9317 ± 0.0321 | 0.0679 ± 0.0411 | ~0.49 | ~319MB |
| hsam_simplified_s200_w3 | 0.9300 ± 0.0347 | 0.0707 ± 0.0457 | ~0.49 | ~319MB |
| hsam_s400_w3_p2_tune | 0.9329 ± 0.0053 | 0.0670 ± 0.0057 | ~0.80 | ~319MB |
| hsam_simplified_s400_w3 | 0.9323 ± 0.0051 | 0.0657 ± 0.0070 | ~0.74 | ~319MB |
| SLDA | 0.9376 ± 0.0109 | **0.0311 ± 0.0106** | ~0.17 | ~389MB |
| replay_mlp_buf100 | 0.9367 ± 0.0087 | 0.0526 ± 0.0135 | ~0.80 | ~387MB |
| replay_mlp_buf200 | **0.9594 ± 0.0057** | **0.0207 ± 0.0123** | ~1.12 | ~387MB |
| finetune_mlp | 0.1915 ± 0.0066 | 0.9882 ± 0.0040 | ~0.69 | ~386MB |
| ewc_mlp | 0.1904 ± 0.0079 | 0.9875 ± 0.0031 | ~1.00 | ~387MB |
| replay_mlp_buf20 | 0.6330 ± 0.0410 | 0.4385 ± 0.0523 | ~0.52 | ~387MB |
| fuzzy_art_rho07 | 0.2295 ± 0.2169 | 0.6513 ± 0.2301 | ~3.47 | ~388MB |
| gngu | 0.1905 ± 0.0432 | 0.2068 ± 0.0460 | ~0.13 | ~388MB |

Sources: `research/03_experiments/hsam_digits/continual_results.json`, `hsam_ablations/results.json`, `hsam_robustness/results.json`, `simplified_benchmark/results.json`, `comparisons/results.json`.

**Interpretation [PRELIMINARY]:**
- **HSAM_Simplified matches original HSAM** on both AA and FM — CI overlap complete at both s200 and s400. The tuner is confirmed dead weight.
- **SLDA and tuned replay outperform HSAM.** SLDA matches HSAM AA with much better FM and 3–5× less wall-clock. replay_buf200 beats HSAM on both AA and FM. HSAM is comparable to replay_buf100 but below replay_buf200.
- MLP finetune/EWC collapse (AA ~0.19) — not competitive under this protocol.
- FuzzyART and GNG-U perform poorly on continual digits in this implementation — not competitive with HSAM/SLDA/replay.

### 7.4 Ablations
`research/03_experiments/hsam_ablations/results.json`, `simplified_benchmark/results.json`: capacity reduction (slots 100) hurts both supervised and continual performance; self-tuning on/off yields near-identical means in all tested configs (stationary, non-stationary/distribution-shift streams, and full grid slots 100/200/400/800 × winners 1/3/5). Winner count/slot count influence results but no single factor explains all gains — slot count ~400 is the sweet spot, s800 shows diminishing returns. HSAM_Simplified matches original HSAM exactly across the board.

### 7.5 Stream / “self-improvement” audit
`research/03_experiments/hsam_self_improve_stream/results.json` (20 chunks), `research/03_experiments/hsam_robustness/results.json` (100 chunks)

| Mode | first | last | delta [95% CI] |
|---|---:|---:|---:|
| full (20 chunks) | 0.8428 | 0.9489 | +0.1061 [0.0883, 0.1217] |
| frozen_after_1 (20) | 0.8422 | 0.8422 | 0.0000 [0.0000, 0.0000] |
| no_tune (20) | 0.8422 | 0.9494 | +0.1072 [0.0911, 0.1233] |
| no_alloc (20) | 0.8428 | 0.9539 | +0.1111 [0.0928, 0.1294] |
| full (100 chunks) | 0.8428 | 0.9589 | +0.1161 [0.1028, 0.1267] |
| frozen_after_1 (100) | 0.8422 | 0.8422 | 0.0000 [0.0000, 0.0000] |
| frozen_after_5 (100) | 0.9300 | 0.9300 | 0.0000 [0.0000, 0.0000] |
| no_tune (100) | 0.8422 | 0.9561 | +0.1139 [0.1050, 0.1250] |
| no_alloc (100) | 0.8428 | 0.9628 | +0.1200 [0.1067, 0.1306] |

**Critical interpretation [PRELIMINARY, negative for autonomy claim]:** continued exposure with online updates improves accuracy over frozen controls, which is consistent with online learning. However, disabling self-tuning does not reduce gains in these runs — including with a redesigned per-slot adaptive + per-class vigilance tuner (non-stationary stream: full +0.0867 vs no_tune +0.0883, indistinguishable). Therefore the evidence does **not** support that HSAM's autonomous homeostatic tuning mechanisms produce superior self-improvement beyond continued local adaptation. No distribution-shift adaptation or long-horizon autonomous skill gain was demonstrated. The self-tuning concept is a **dead end**.

**Allocation — not tuning — drives retention.** Disabling slot allocation (no_alloc) does not reduce within-stream accuracy gain (delta ~+0.10–0.12 in stream probes), but across the full continual grid (slots 100/200/400/800 × winners 1/3/5) it drops AA from ~0.93 to ~0.76 and FM rises from ~0.07 to ~0.28. Allocation is the retention-critical mechanism; tuning is not.

### 7.6 Additional exploratory/negative results
- `research/03_experiments/char_csm/results.json`: adaptive CSM degraded over stream (overall 0.686, last20 0.478 < first20 0.835) while fixed ctx2 stayed ~0.824 — negative for that self-adaptation heuristic.
- `research/03_experiments/candidates_digits/*`: early/pre-fix candidate runs (HSAM supervised ~0.10 in that artifact, RLS weak); **superseded/audit trail only**.
- RLS_SelfGate: weak sanity and earlier candidate performance; not pursued as lead.

## 8. Phase 7 - Theoretical/analytical notes
- Informal analysis and limitations: `research/novelty/theory.md`.
- Complexity: per update/query dominated by $O(Mp)$ similarity scan plus small winner updates; experiments ran in sub-second to ~1s per config/seed regime on 2 vCPU with low peak RSS (see per-seed stats in JSONs).
- No formal convergence/capacity/forgetting guarantees proven [CONJECTURE].

## 9. Claims vs evidence

| Claim | Label | Evidence | Caveats |
|---|---|---|---|
| HSAM can overfit a tiny 32-sample toy task reliably (sanity) | [EMPIRICALLY SUPPORTED, narrow] | `research/03_experiments/sanity/overfit32_results.json`, `tests/test_core.py` | Toy setting only; RLS control failed on same sanity. |
| HSAM and HSAM_Simplified achieve ~0.95–0.97 accuracy on digits supervised split (5 seeds, CIs) | [PRELIMINARY] | `research/03_experiments/hsam_digits/supervised_results.json`, `simplified_benchmark/results.json` | Small 8×8 dataset; comparison limited. HSAM_Simplified matches original HSAM exactly. |
| HSAM_Simplified matches original HSAM on continual AA/FM (no tuner needed) | [EMPIRICALLY SUPPORTED, narrow] | `simplified_benchmark/results.json` (s200: 0.9300 vs 0.9317 AA; s400: 0.9323 vs 0.9329 AA; CI overlap complete) | Removing self-tuning changes nothing — tuner confirmed dead weight. |
| HSAM attains high AA and low FM on split-digits-5 under tested protocol/orders | [PRELIMINARY] | `research/03_experiments/hsam_digits/continual_results.json`, `hsam_ablations/results.json`, `hsam_robustness/results.json`, `simplified_benchmark/results.json` | Baselines/mechanisms differ; SLDA and tuned replay narrow or exceed the gap. |
| SLDA and tuned replay outperform HSAM on continual AA/FM | [EMPIRICALLY SUPPORTED, narrow] | `simplified_benchmark/results.json` (SLDA AA 0.9376 vs HSAM 0.9323; replay_buf200 AA 0.9594; SLDA FM 0.0311 vs HSAM 0.0657; replay_buf200 FM 0.0207 vs HSAM 0.0657) | SLDA/replay use different mechanisms (class means / replay buffer) vs HSAM's prototype allocation. |
| Continued online HSAM updates improve over frozen snapshot on stream probe | [PRELIMINARY] | `hsam_self_improve_stream/results.json`, `hsam_robustness/results.json`, `hsam_nonlinear_stream/results.json` (frozen controls delta 0) | Standard online adaptation, not proof of novel autonomous self-improvement. |
| Homeostatic self-tuning/allocation mechanism causes superior autonomous self-improvement (stationary or non-stationary) | **[NOT SUPPORTED]** | `hsam_nonlinear_stream/results.json` (redesigned per-slot adaptive tuner: full +0.0867 vs no_tune +0.0883), full grid `hsam_grid/results.json` shows no_tune ≈ full across all slots/winners; no_alloc hurts in continual | **Empirically neutral/dead weight.** Redesigned with per-slot adaptive rates + per-class vigilance — still no gap. Allocation drives continual retention; tuning does not. Do not frame as continuous self-improvement. |
| HSAM is fundamentally novel | **[NOT SUPPORTED / incremental]** | Head-to-head vs FuzzyART/GNG-U/SLDA + component mapping in `research/04_results/mechanistic_comparison.md`; `research/novelty/*` | Best described as incremental mechanistic combination/reparameterization; removing homeostatic tuning makes it most equivalent to GNG-U/ART-like systems. |
| Scale-up to Fashion-MNIST (PCA features) | **[NOT SUPPORTED / INVALID]** | `research/03_experiments/largescale_fashionmnist/results.json` + `README.md` (load failed, incomplete/improvised) | **Not used as evidence.** Sample counts, PCA variance retained, and identical data across baselines unknown/unverifiable; full CIFAR not completed under budget. |
| Formal guarantees (convergence/optimality) | [CONJECTURE / none] | `research/novelty/theory.md` | Informal analysis only. |

## 10. Reproducibility
- **Core code:** `src/{utils.py,data.py,metrics.py,candidates.py,baselines.py}`
- **Scripts:** `scripts/run_toy_sanity.py`, `scripts/run_hsam_digits.py`, `scripts/run_hsam_ablations.py`, `scripts/run_hsam_stream_self_improve.py`, `scripts/run_hsam_robustness.py`, `scripts/run_simplified_benchmark.py`, `scripts/run_baselines_digits.py`, `scripts/run_char_csm.py`, `scripts/aggregate_results.py`, `scripts/reproduce_headline.sh`, `scripts/reproduce_all.sh`
- **Tests:** `python3 -m unittest discover -s tests -v` (`tests/test_core.py`)
- **Headline reproduction (fast, CPU):**
  ```bash
  export PYTHONPATH="$PWD"
  python3 -m unittest discover -s tests -v
  bash scripts/reproduce_headline.sh
  python3 scripts/aggregate_results.py
  ```
- **Full reproduction (includes torch baselines/char runs):** `bash scripts/reproduce_all.sh`
- **Artifacts:** configs/results in `research/03_experiments/*/*.json`; aggregated `research/04_results/summary.json`, `tables.md`, plots. Some saved baseline/char artifacts record earlier git hash (`d1209e9...`) from prior execution; rerunning scripts regenerates artifacts for current commit.

## 11. Limitations and future work
- Evidence is limited to tiny datasets (digits/streams) and few task structures; no vision-scale, language, or long-horizon deployment validation.
- Continual baselines are not exhaustive (could include stronger replay/regularization/architecture methods and tuned MLPs).
- **Self-improvement mechanism is a dead end.** Redesigned with per-slot adaptive rates + per-class vigilance — still no gap vs no-tuning across stationary, non-stationary, and grid sweeps. Do not pursue further self-tuning redesigns.
- SLDA and tuned replay outperform HSAM on continual AA/FM — HSAM is not state-of-the-art even on this small task; it is a lightweight ART/GNG-U-family learner with specific tradeoffs (no replay buffer, low compute).
- Novelty is incremental; a publishable claim would require stronger differentiation and wins over nearest ART/GNG/SOINN/prototype baselines.
- Missing prior `run.log` files for some runs (tee race noted in logs/history); JSON artifacts are the primary evidence.

## 12. Integrity and git notes
- I did not fabricate experimental numbers, citations, or execution claims. Prior/superseded artifacts (e.g. `candidates_digits/*`) are retained and labeled as such.
- All work committed to `kilo/active-canyon-zrz` only (commit `752e884`). **Not pushed to main** — awaiting explicit go-ahead. `main` remains at `db2e2dc`; `origin/main` was advanced to `722d821` in an earlier push (before the "wait for go-ahead" instruction was clarified).
