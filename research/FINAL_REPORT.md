# Final Research Report: HSAM (Homeostatic Surprise-Gated Associative Memory)

**Date:** 2026-11-12 (updated)  
**Repository:** `/workspace/app` (`https://github.com/billy2112/AI-algorithm.git`)  
**Environment:** Debian 13 (trixie), 2 vCPU, 8 GB RAM, no GPU, CPU-only. Python 3.13.5; numpy 2.2.4, scipy 1.15.3, scikit-learn 1.4.2, torch 2.6.0 (CPU), matplotlib 3.10.1, tqdm 4.67.1, psutil 7.0.0.  
**Working branch in this workspace:** `kilo/active-canyon-zrz` at `e019a8c`; pushed to `origin/main`, `origin/kilo/active-canyon-zrz`, `origin/research/novel-arch-20261110`.

## Executive summary
This project investigated a novel, CPU-efficient, self-learning/continuously adapting architecture under strict resource and evidence constraints. I implemented and evaluated **HSAM (C1)**, a backprop-free online prototype/associative memory with fixed random projection, k-winner local delta updates, surprise/mismatch/low-similarity-gated slot allocation, utility replacement, and bounded endogenous homeostatic tuning.

**Headline outcomes (preliminary, small-scale on sklearn digits 8×8):**
- On `sklearn` digits (80/20 stratified split, 5 seeds), HSAM configurations reached **~0.949–0.971 test accuracy** (`research/03_experiments/hsam_digits/supervised_results.json`), competitive with RF-ridge/MLP baselines.
- On split-digits-5 continual learning (2-class tasks sequentially, no replay buffer, 5 seeds), HSAM reached **AA ~0.860–0.941, FM ~0.053–0.092** across tested configs/orders (`hsam_digits/continual_results.json`, `hsam_ablations/results.json`, `hsam_robustness/results.json`), substantially higher AA / lower FM than saved MLP finetune/EWC baselines and competitive with tuned replay in places; **SLDA** (streaming LDA, prototype/class-mean based) is the fairest closest comparator and narrows the gap.
- **Stream probes (stationary and non-stationary with distribution shift):** Continued online learning improves over frozen snapshots, but **ablations show no_tune and no_alloc are statistically indistinguishable from full HSAM** across stationary streams, non-stationary/corrupted streams, and the full hyperparameter grid (slots 100/200/400/800 × winners 1/3/5). The homeostatic self-tuning/allocation mechanism is **empirically neutral/dead weight** for any "continuous self-improvement" advantage.

**Scale-up attempt:** Fashion-MNIST was attempted via PCA features (64 components) due to OpenML/environment constraints (pandas/parser issues); full CIFAR-10 feature embeddings were not completed within tight 2vCPU/8GB budget - exact compromise documented in `research/03_experiments/largescale_fashionmnist/results.json`. At PCA-reduced Fashion-MNIST scale the margin appears to narrow/compress relative to strong prototype baselines (see artifacts).

**Novelty verdict (backed by head-to-head runs + mechanistic analysis):** **[LIKELY INCREMENTAL MECHANISTIC COMBINATION, not fundamentally new]**. Head-to-head comparisons with Fuzzy ART, GNG-U, and SLDA show HSAM shares prototype competition, incremental allocation/pruning ideas with ART/GNG-U and is close to SLDA class-mean learning. Component mapping (see `research/04_results/mechanistic_comparison.md`) shows no component is structurally impossible for these families; removing homeostatic tuning makes HSAM most equivalent to a GNG-U/ART-like prototype system. Treat as integration/reparameterization rather than a novel mechanism.

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
`research/03_experiments/hsam_digits/supervised_results.json`

| Config | acc mean ± std [95% CI] | wall_s mean |
|---|---:|---:|
| hsam_s400_w1_p2_tune | 0.9628 ± 0.0131 [0.9522, 0.9733] | ~0.74 |
| hsam_s400_w3_p2_tune | 0.9706 ± 0.0058 [0.9661, 0.9750] | ~0.65 |
| hsam_s200_w3_p2_tune | 0.9494 ± 0.0095 [0.9428, 0.9578] | ~0.46 |
| hsam_s400_w1_p2_notune | 0.9644 ± 0.0129 [0.9544, 0.9756] | ~0.72 |

**Interpretation [PRELIMINARY]:** competitive with strong simple baselines on this small dataset; capacity (slots) and readout settings matter modestly.

### 7.3 Continual split-digits-5 (5 seeds)
`research/03_experiments/hsam_digits/continual_results.json`, `hsam_ablations/results.json`, `hsam_robustness/results.json`

| Config / order | AA mean ± std [95% CI] | FM mean ± std [95% CI] |
|---|---:|---:|
| hsam_s200_w3_p2_tune (default order) | 0.9322 ± 0.0269 [0.9140, 0.9566] | 0.0672 ± 0.0349 [0.0360, 0.0925] |
| hsam_s200_w3_p2_notune | 0.9300 ± 0.0347 [0.9063, 0.9610] | 0.0707 ± 0.0457 [0.0293, 0.1034] |
| hsam_s100_w3_p2_tune | 0.8605 ± 0.0362 [0.8329, 0.8902] | 0.0922 ± 0.0635 [0.0400, 0.1497] |
| hsam_s400_w3_p2_tune | 0.9268 ± 0.0194 [0.9122, 0.9438] | 0.0747 ± 0.0286 [0.0529, 0.0976] |
| ablation w1_s200 (continual) | 0.9412 ± 0.0121 [0.9303, 0.9491] | 0.0525 ± 0.0153 [0.0394, 0.0639] |
| task order reverse | 0.9322 ± 0.0183 [0.9163, 0.9460] | 0.0677 ± 0.0256 [0.0461, 0.0901] |
| task order interleaved | 0.9396 ± 0.0156 [0.9279, 0.9526] | 0.0599 ± 0.0249 [0.0422, 0.0826] |

**Interpretation [PRELIMINARY]:** strong retention/plasticity balance on this protocol across tested orders and several configs; tuning vs no-tuning difference is small in these runs. Comparison to MLP finetune/EWC/replay baselines favors HSAM under this setup, but baselines are not exhaustive and mechanisms differ.

### 7.4 Ablations
`research/03_experiments/hsam_ablations/results.json`: capacity reduction (slots 100) hurts both supervised and continual performance; self-tuning on/off yields near-identical means in tested configs; winner count/slot count influence results but no single factor explains all gains.

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

**Critical interpretation [PRELIMINARY, negative for autonomy claim]:** continued exposure with online updates improves accuracy over frozen controls, which is consistent with online learning. However, disabling self-tuning or allocation-after-fill (as implemented in probe) does not reduce gains in these runs; therefore the evidence does **not** support that HSAM’s autonomous homeostatic/allocation mechanisms produce superior self-improvement beyond continued local adaptation. No distribution-shift adaptation or long-horizon autonomous skill gain was demonstrated.

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
| HSAM achieves ~0.95–0.97 accuracy on digits supervised split in tested configs (5 seeds, CIs) | [PRELIMINARY] | `research/03_experiments/hsam_digits/supervised_results.json`, `hsam_grid/results.json`, rerunnable via scripts | Small 8×8 dataset; comparison limited. |
| HSAM attains high AA and low FM on split-digits-5 under tested protocol/orders relative to saved MLP finetune/EWC/replay baselines | [PRELIMINARY] | `research/03_experiments/hsam_digits/continual_results.json`, `hsam_ablations/results.json`, `hsam_robustness/results.json`, `comparisons*/results.json` | Baselines/mechanisms differ; tuned replay and SLDA narrow the gap substantially. |
| Continued online HSAM updates improve over frozen snapshot on stream probe | [PRELIMINARY] | `hsam_self_improve_stream/results.json`, `hsam_robustness/results.json`, `hsam_nonlinear_stream/results.json` (frozen controls delta 0) | Standard online adaptation, not proof of novel autonomous self-improvement. |
| Homeostatic self-tuning/allocation mechanism causes superior autonomous self-improvement (stationary or non-stationary) | **[NOT SUPPORTED]** | `hsam_nonlinear_stream/results.json` (full/no_tune/no_alloc indistinguishable under distribution shift, incl. redesigned per-slot adaptive tuner: full +0.0867 vs no_tune +0.0883), full grid `hsam_grid/results.json` shows no_tune ≈ full across all slots/winners; no_alloc hurts in continual | **Empirically neutral/dead weight.** Redesigned with per-slot adaptive rates + per-class vigilance — still no gap. Do not frame as continuous self-improvement. |
| HSAM is fundamentally novel | **[NOT SUPPORTED / incremental]** | Head-to-head vs FuzzyART/GNG-U/SLDA + component mapping in `research/04_results/mechanistic_comparison.md`; `research/novelty/*` | Best described as incremental mechanistic combination/reparameterization; removing homeostatic tuning makes it most equivalent to GNG-U/ART-like systems. |
| Scale-up to Fashion-MNIST (PCA features) | **[NOT SUPPORTED / INVALID]** | `research/03_experiments/largescale_fashionmnist/results.json` + `README.md` (load failed, incomplete/improvised) | **Not used as evidence.** Sample counts, PCA variance retained, and identical data across baselines unknown/unverifiable; full CIFAR not completed under budget. |
| Formal guarantees (convergence/optimality) | [CONJECTURE / none] | `research/novelty/theory.md` | Informal analysis only. |

## 10. Reproducibility
- **Core code:** `src/{utils.py,data.py,metrics.py,candidates.py,baselines.py}`
- **Scripts:** `scripts/run_toy_sanity.py`, `scripts/run_hsam_digits.py`, `scripts/run_hsam_ablations.py`, `scripts/run_hsam_stream_self_improve.py`, `scripts/run_hsam_robustness.py`, `scripts/run_baselines_digits.py`, `scripts/run_char_csm.py`, `scripts/aggregate_results.py`, `scripts/reproduce_headline.sh`, `scripts/reproduce_all.sh`
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
- Self-improvement mechanism not validated; controller may need redesign, richer signals, or different regimes.
- Novelty is incremental; a publishable claim would require stronger differentiation and wins over nearest ART/GNG/SOINN/prototype baselines.
- Missing prior `run.log` files for some runs (tee race noted in logs/history); JSON artifacts are the primary evidence.

## 12. Integrity and git notes
- I did not fabricate experimental numbers, citations, or execution claims. Prior/superseded artifacts (e.g. `candidates_digits/*`) are retained and labeled as such.
- Mission instructed not to touch `main` and to open a PR; user explicitly requested “commit and push everything to the main branch”. I will follow the explicit user request, preserve work on a research branch/clear history, and document this conflict in `research/LOG.md`.
