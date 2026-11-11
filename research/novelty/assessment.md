# Novelty and Candidate Assessment (2026-11-10)

## Scope and honesty
This assessment is based on in-session web searches (see `search_log.md`), saved experiment artifacts in `research/03_experiments/`, and project hypotheses in `research/02_hypotheses.md`. It does **not** claim exhaustive prior-art coverage. Mechanistic similarity judgments are preliminary where full papers were not read end-to-end.

## Closest prior-art families to HSAM (current lead candidate)
1. **Adaptive Resonance Theory (ART/ARTMAP/topological ART):** vigilance/match criterion controls whether to update/resonate with an existing category or create/commit a new one; local prototype updates; stability-plasticity via resonance. Recent work includes self-adjusting/adaptive vigilance (e.g. Masuyama et al., arXiv:2511.17983, 2025; ARTMAP adaptive vigilance tuning, J-Stage FSS 2025).
2. **Growing/incremental prototype networks:** GNG/GNG-U (Fritzke; utility-based deletion), DCS/GCS, OGNG/DYNG, and SOINN/E-SOINN/adjusted SOINN/GSOINN+ families use similarity-based insertion/adaptation and node/edge utility/pruning for online topology/prototype learning and lifelong settings.
3. **Backprop-free Hebbian prototype/readout memories:** fixed/random-feature front-ends with local or closed-form readouts (ESN/ELM) and local Hebbian prototype learning (e.g. RECAP, arXiv:2603.06639) show online-updatable prototype associations without backprop.
4. **Continual-learning prototype/replay methods:** iCaRL/prototype classifiers, exemplar/statistics/generative replay, and exemplar-free prototype-conditioned replay address forgetting differently (often supervised batches/replay buffers/distillation), but overlap conceptually in prototype maintenance.

## HSAM differentiation (mechanistic, preliminary)
HSAM implementation concept (from hypotheses and saved experiment intent): fixed random projection encoder, fixed-capacity key/value slots, k-winner retrieval, local delta updates to keys/values, allocation triggered by surprise/error and/or mismatch/low-similarity, utility accumulation/decay and replacement, and bounded endogenous tuning of rates/thresholds from error/improvement EMAs.

- **Potentially distinguishing combination:** (a) no learned graph/topology edges (unlike GNG/SOINN graph methods), (b) associative key->value slots for direct prediction/class targets with local delta updates, (c) allocation driven by a composite of surprise, label mismatch, and low similarity, (d) autonomous homeostatic adjustment using endogenous improvement signal. 
- **Not distinguishing on its own:** vigilance-like novelty gating (ART), prototype allocation/pruning by utility (GNG-U/SOINN), local Hebbian/delta updates, random projection/front-end, or self-tuning thresholds (recent ART and adaptive systems literature).
- **Verdict:** **[LIKELY INCREMENTAL MECHANISTIC COMBINATION]**. Any novelty claim must be narrowly scoped to the exact algorithmic combination and supported by ablations + explicit comparison/positioning against ART/self-adjusting vigilance and incremental prototype networks.

## Evidence status from saved artifacts (workspace snapshot)

| Artifact | What it shows | Evidence strength / caveats |
|---|---|---|
| `research/03_experiments/baselines_digits/supervised_results.json` | Digits supervised (5 seeds): ridge 0.939±0.011, rf_ridge_256 0.978±0.002, mlp_sklearn_64 0.977±0.004, torch_mlp_64_e20 0.951±0.005 | **[EMPIRICALLY SUPPORTED within artifact]** for those configs/seeds; script/source modules not present in current snapshot so direct rerun reproduction from this state is blocked. |
| `research/03_experiments/baselines_digits/continual_results.json` | Split-digits-5 continual (5 seeds): finetune AA 0.192 FM 0.988, EWC AA 0.190 FM 0.988, replay_buf20 AA 0.633 FM 0.438 | Same caveat; strong forgetting for naive/EWC in this setup, replay moderate. |
| `research/03_experiments/hsam_digits/supervised_results.json` | HSAM supervised (5 seeds): 0.949–0.971 across configs (best s400_w3_p2_tune 0.971±0.006) | **Promising but [PRELIMINARY/UNREPRODUCED in current snapshot]**: implementation code missing from snapshot; must be rerun from restored source to support claims. Competitive with RF-ridge on this small task. |
| `research/03_experiments/hsam_digits/continual_results.json` | HSAM continual (5 seeds): AA ~0.860–0.932, FM ~0.067–0.092 across configs (best shown s200_w3_p2_tune AA 0.932±0.027, FM 0.067±0.035) | **Promising but [PRELIMINARY/UNREPRODUCED in current snapshot]**. If reproduced, materially better AA/FM than finetune/EWC/replay(20) on this protocol; still needs ablations, task-order/seed robustness, and stronger baselines. |
| `research/03_experiments/hsam_self_improve_stream/results.json` | Stream probe (20 chunks, fixed test): full Δ +0.106, frozen_after_1 Δ 0.000, no_tune Δ +0.107, no_alloc Δ +0.111 (5 seeds) | **[PRELIMINARY, interpretation caution]**. Shows continued improvement with ongoing updates vs frozen-after-chunk-1, but no_tune/no_alloc also improved similarly in this run — so improvement cannot be attributed to homeostatic self-tuning or allocation mechanism from this artifact alone. "Self-improvement" claim is not supported as autonomous meta-improvement beyond continued online learning; needs redesigned controls/ablations and longer-stream evidence. |
| `research/03_experiments/char_csm/results.json` | Adaptive CSM overall 0.686, last20 0.478 vs first20 0.835; fixed ctx2 0.824/0.828/0.823 | **Negative signal** for adaptive self-improvement in that setting. |
| `research/03_experiments/candidates_digits/*` | Early/pre-fix or exploratory candidate runs (HSAM supervised ~0.10 in file; RLS weak) | **Superseded/invalid for fixed HSAM**; retain as audit trail only. |
| `research/03_experiments/sanity/overfit32_results.json` | HSAM overfit-32 acc 1.0; RLS 0.256 | Sanity pass for HSAM, fail/weak for RLS candidate in that form. |

## Implications for claims
- **[PROVEN]:** none yet.
- **[EMPIRICALLY SUPPORTED]:** none that can be responsibly claimed from this snapshot alone, because key implementation/reproduction scripts for lead results are missing and reruns were not executed in the current environment.
- **[PRELIMINARY]:** saved JSONs suggest HSAM can be competitive on tiny digits and can reduce forgetting vs simple MLP continual baselines under this protocol, but this must be reproduced from committed code/configs with matched baselines and ablations.
- **[CONJECTURE]:** HSAM's specific combination may offer a useful CPU-cheap local-memory tradeoff; autonomous "self-improvement" beyond standard online adaptation is unproven and contradicted in part by the stream ablation pattern.

## Required next validation steps
1. Restore/reconstruct `src/utils.py`, `src/data.py`, `src/metrics.py`, `src/baselines.py`, `src/candidates.py`, and experiment scripts; pin environment and verify outputs reproduce saved JSON summaries.
2. Run ablations: no-allocation, frozen thresholds/tuning, 1-vs-k winners, slot-capacity sweep, no-random-projection/raw input, utility-reset, and matched-parameter/readout baselines.
3. Self-improvement audit: longer stream (>=10x), frozen-after-T, disabled-adaptation, disabled-meta-controller, distribution-shift segments, and leakage checks; claim only if online/meta mechanism beats continued-learning controls with CIs excluding 0.
4. Continual-learning robustness: multiple task orders/seeds, stronger CL baselines if feasible (e.g. better replay/regularization configs) under equal budget.
5. Novelty writeup: position explicitly against ART/self-adjusting vigilance, GNG-U/SOINN, and Hebbian prototype readouts; label contribution as incremental unless clearly demonstrated otherwise.
