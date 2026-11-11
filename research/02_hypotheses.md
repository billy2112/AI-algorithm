# Phase 2 - Pre-registered Hypotheses, Metrics, Protocol, and Kill Criteria

**Date:** 2026-11-10 (pre-experiment registration)  
**Status:** Registered before running comparative experiments. Any deviations will be logged in `research/LOG.md`.

## 1. Target properties and quantitative success criteria

### 1.1 Efficiency (compute/energy proxy, CPU-only)
- **Metrics recorded per run:** wall-clock seconds, peak RSS (MB) via `psutil`/`resource` where available, analytical FLOPs estimate per training/update/inference step and total (counted in code; approximations documented), parameter count, memory footprint of state/buffers.
- **Primary comparison:** quality achieved within a fixed budget (target <= 300s, peak RAM < 1 GB for core experiments) versus baselines under equal-or-greater budget and equal tuning effort.
- **Success criterion:** [EMPIRICALLY SUPPORTED] if proposed method reaches >= baseline best quality with >=5x fewer analytical training/update FLOPs or >=5x less wall-clock at equal quality; OR matches within 2% relative quality at >=10x lower training cost. Smaller effects/single-seed are [PRELIMINARY].

### 1.2 Learns by itself (minimal/no labels, online/self-supervised)
- **Tasks:** online next-step prediction on character/symbol streams and self-supervised associative prediction; label-efficiency via frozen/online linear probe on <=1% labels where applicable. 0-label stream learning is primary.
- **Metrics:** prediction accuracy / bits-per-character proxy / MSE vs examples seen (data-efficiency curves); probe accuracy with 0%/1% labels.
- **Success criterion:** [EMPIRICALLY SUPPORTED] if at 0% or <=1% labels method achieves non-trivial performance (>= chance + meaningful margin, e.g. >1.5x random on classification or >=20% relative error reduction vs naive predictor) and competitive data-efficiency vs baselines on same unlabeled stream.

### 1.3 Continual learning (no catastrophic forgetting)
- **Tasks:** Split-Digits-5 stream from `sklearn.datasets.load_digits` (8x8 images, 10 classes) split into 5 disjoint two-class tasks in fixed order; Permuted-Digits (>=3 permutations) as secondary. No replay buffer, no task IDs at test time for proposed method unless intrinsic and disclosed. Baselines: fine-tuning, EWC, small replay.
- **Metrics:** per-task accuracy after each task, Average Accuracy (AA) at end, Forgetting Measure FM = mean_t (max_{k<=T} acc_{t,k} - acc_{t,T}), current-task plasticity.
- **Protocol:** fixed task order/splits/seeds before runs.
- **Success criterion:** [EMPIRICALLY SUPPORTED] if AA >= fine-tuning by >=10 pp and FM >=25% lower than fine-tuning, with no replay and no test-time task IDs; comparisons to EWC/replay reported honestly.

### 1.4 Continuous self-improvement after deployment
- **Definition:** After initial phase, a snapshot is taken; the model continues on a stream using only its own autonomous mechanism (no human intervention, no external re-tuning, no label supervision unless intrinsically generated from self-supervised prediction error — specified per method).
- **Metrics:** quality vs experience/time over >=10x more stream length than initial phase. Compare to (a) frozen snapshot, (b) self-modification disabled, (c) standard online baseline where fair.
- **Success criterion:** [EMPIRICALLY SUPPORTED] if statistically significant improvement over frozen control (paired across seeds/segments, bootstrap 95% CI of difference excludes 0) sustained in late segments and not explained by external supervised training.

### 1.5 Intelligence / generalization
- **Tasks:** synthetic algorithmic streams (copy/repeat, associative recall key-value, parity, modular addition/mod counting), length extrapolation (train/adapt short, test longer), OOD symbol/order variations.
- **Metrics:** accuracy/MSE on held-out IID and OOD/longer splits; generalization gap.
- **Success criterion:** [EMPIRICALLY SUPPORTED] if above chance and above naive memorization/majority baselines on >=2 algorithmic tasks with multi-seed significance, including some OOD/extrapolation gain.

## 2. Statistical protocol
- Minimum **5 random seeds** per headline configuration; 1-4 seeds labeled [PRELIMINARY].
- Report **mean ± std** and **95% bootstrap confidence intervals** (>=1000 resamples) for key metrics; paired bootstrap/differences where applicable.
- Improvement claimed only if CI of difference excludes 0 or effect clearly exceeds seed variance (documented).
- Count and report every variation tried per candidate (anti p-hacking). Equal tuning budget per method.
- Train/val/test splits and stream order fixed **before** comparative runs; seeds recorded. **No tuning on test set.**

## 3. Baselines (fair, equal-budget tuning)
- **Supervised/reference:** small MLP, small LSTM/GRU, small Transformer (1-4 layers, ~0.1-1M params feasible), linear/SGD baseline.
- **Efficient:** ridge/online linear readout on fixed random features (ELM/reservoir-style readout) where applicable.
- **Continual:** naive fine-tuning, EWC, small replay buffer.
- Small documented LR/size sweep with equal tuning budget; select on validation, evaluate on test with 5 seeds.

## 4. Kill criteria (pre-registered)
1. **No sanity signal:** cannot overfit 32 samples or beat trivial/random baseline on toy task after debugging.
2. **No competitiveness:** after 3 honest iterations and fair tuning, >20% relative worse (or >10 pp) than strongest simple baseline on primary metric with no efficiency advantage.
3. **Budget infeasible:** cannot run meaningful experiment within 20 min / <6 GB RAM.
4. **Novelty already known:** mechanism essentially identical to prior work found in searches — drop or relabel incremental.
5. **Self-improvement invalid:** gain disappears vs frozen/disabled controls or explained by leakage/bug.

## 5. Data and reproducibility
- Datasets: `sklearn` digits (built-in), synthetic generated streams, tiny char data generated or scripted download into git-ignored `data/` only if small (<5MB). Fallback: generated text.
- Every run saves: config, `results.json`, stdout log, git commit hash, seed, env info, wall-clock, peak RAM.
- Plots from saved results by script. Single-command reproduction scripts in `scripts/`.

## 6. Claim labels
- **[PROVEN]:** formal derivation checked and/or exhaustive verification in stated scope.
- **[EMPIRICALLY SUPPORTED]:** >=5 seeds, baselines/ablations as relevant, CI/significance reported, reproducible by script.
- **[PRELIMINARY]:** 1-4 seeds, small scale, or limited comparison.
- **[CONJECTURE]:** plausible but not demonstrated.
All claims in FINAL_REPORT carry one label and evidence pointer.

---

# Phase 3 - Hypothesis Generation (8 candidates)

**Generated:** 2026-11-10 before candidate-specific experiments. Each candidate specifies state/parameters, forward computation, learning rule, self-improvement mechanism, cost rationale, and expected failure modes. Novelty assessed against Phase 1 searches; per-candidate search details appended to `novelty/search_log.md`.

## Candidate 1: Homeostatic Surprise-Gated Associative Memory (HSAM)
- **Core idea:** A fixed-capacity associative memory of prototype slots with local Hebbian write/recall and an autonomous homeostatic controller that adapts vigilance/plasticity from internal prediction statistics.
- **State/parameters:** $M$ slots, each slot $i$: key $k_i \in \mathbb{R}^d$, value $v_i \in \mathbb{R}^o$, utility $u_i \in \mathbb{R}$, age $a_i$. Global stats: running error EMA $\mu_e, \sigma_e^2$, running prediction-improvement EMA $\Delta$. Fixed random projection $P$ (optional) for inputs.
- **Forward computation:** Encode input $x_t \to z_t=\phi(Px_t)$ (normalized). Similarities $s_i = \mathrm{sim}(z_t,k_i)$. Retrieve top-1 or top-r with weights $w_i=\mathrm{softmax}(s_i/\tau)$ over winners $W_t$. Prediction $\hat{y}_t = \sum_{i\in W_t} w_i v_i$.
- **Learning rule (local, online, no backprop):** Observe target $y_t$ (self-supervised next token/value or provided label in supervised mode). Error $e_t=\|y_t-\hat{y}_t\|$. Surprise $\sigma_t = e_t/(\mu_e+\epsilon)$. 
  - If $\sigma_t \le \theta_{low}$ (familiar): consolidate winners $k_i \leftarrow \mathrm{norm}((1-\eta_k)k_i+\eta_k z_t)$, $v_i \leftarrow v_i + \eta_v w_i (y_t-v_i)$, $u_i \leftarrow \gamma u_i + w_i(1-\sigma_t)$.
  - If $\sigma_t \ge \theta_{high}$ (novel/surprising): allocate new slot at least-utility index $j=\arg\min u_i$: $k_j=z_t, v_j=y_t, u_j=u_0$ (structural plasticity).
  - Else: small interpolated updates. Age/decay $u_i \leftarrow \lambda u_i$ each step for forgetting stale slots.
- **Self-improvement mechanism:** Autonomous meta-controller updates $(\theta_{low},\theta_{high},\eta_k,\eta_v)$ from internal signals only: if recent prediction-improvement $\Delta>0$, tighten thresholds/reduce rates to stabilize good memories; if $\Delta\le0$ or stagnation, increase plasticity/allocation sensitivity within bounds. This is a bounded homeostatic rule on endogenous error statistics (no external tuning after deployment).
- **Cost:** $O(Md)$ per step (dot products over $M$ slots) + tiny scalar updates; memory $O(M(d+o))$. Extremely cheap for small $M,d$.
- **Expected failure modes:** capacity saturation/interference for many patterns; similarity metric mismatch; instability if thresholds oscillate; poor feature space without learned encoder (random projection may be weak).

## Candidate 2: Dual-Timescale Error-Correcting Reservoir (DTEC-R)
- **Core idea:** Fixed random reservoir + two readouts at different adaptation rates plus an autonomous error-correction memory; fast readout adapts online, slow readout consolidates stable correlations, correction memory stores residual prototypes.
- **State/parameters:** Reservoir state $r_t\in\mathbb{R}^{N_r}$ with fixed weights $W_{in},W_r$. Fast linear readout $W_f$, slow readout $W_s$ (EMA of $W_f$), error-correction table $C=\{(r_j,\delta_j,c_j)\}$ of bounded size.
- **Forward computation:** $r_t=(1-\alpha)r_{t-1}+\alpha\tanh(W_{in}x_t+W_r r_{t-1})$. Base prediction $\hat{y}_b = W_s r_t$. Correction $\hat{y}_c = \sum_j \kappa(r_t,r_j)\delta_j$. Output $\hat{y}=\hat{y}_b + g_t\hat{y}_c$ where gate $g_t$ from confidence.
- **Learning rule:** Self-supervised target $y_t$. Fast update: online least-squares/SGD step $W_f \leftarrow W_f + \eta_f (y_t - W_f r_t)r_t^T$ (local in output). Slow: $W_s \leftarrow (1-\beta)W_s + \beta W_f$, $\beta\ll\eta_f$. Residual $\delta_t=y_t-\hat{y}$ stored in $C$ if large; entries merged/decayed by usage count $c_j$.
- **Self-improvement mechanism:** Model autonomously adjusts $\beta, g_t$ threshold, and correction-memory pruning based on residual trend (if residuals systematically decrease, increase consolidation; if spikes, increase correction reliance) — endogenous control only.
- **Cost:** Reservoir step $O(N_r^2)$ fixed but small; readout/correction $O(N_r o + |C|N_r)$.
- **Expected failure modes:** Fixed reservoir may lack needed features; resembles ESN+online readout+memory (incremental risk); stability of online updates.

## Candidate 3: Compression-Guided Sparse Code Grower (CSCG)
- **Core idea:** Online dictionary/sparse-coding learner minimizing reconstruction/prediction error (MDL-flavored proxy) with local Hebbian updates and autonomous neuron/code growth-pruning driven by utilization and surprise.
- **State/parameters:** Dictionary $D\in\mathbb{R}^{d\times K}$ (columns atoms), bias/gain per atom, utilization counters $n_k$, sparse code $a_t$.
- **Forward computation:** Encode $x_t$ by greedy matching pursuit/thresholding: $a_t = \mathrm{top\text{-}s}(D^T x_t)$ with soft threshold, reconstruction $\hat{x}_t=Da_t$ (or prediction variant).
- **Learning rule:** Local Oja/Hebbian-like update $D_k \leftarrow D_k + \eta a_{k,t}(x_t - \hat{x}_t)$ with normalization; only active atoms updated. Prediction readout from $a_t$ via local delta rule.
- **Self-improvement mechanism:** If sustained surprise/compression error high and capacity underused, grow new atom initialized from residual; if atom utilization low over window, prune/merge. Sparsity $s$ and thresholds self-adjust to target reconstruction/utilization bands (homeostatic).
- **Cost:** $O(dK)$ per step, sparse active set; dynamic $K$ bounded.
- **Expected failure modes:** Greedy sparse coding unstable; growth/prune oscillation; may reduce to online k-SVD/sparse coding variants.

## Candidate 4: Bilinear Fast-Trace Predictor (BFTP)
- **Core idea:** Gradient-free fast-weights via bilinear Hebbian traces: context writes a temporary key-value transform used for next-step prediction, with a slow consolidated matrix and surprise-gated trace decay.
- **State/parameters:** Slow weights $W_s\in\mathbb{R}^{o\times d}$, fast trace $W_f\in\mathbb{R}^{o\times d}$, decay/trace rates.
- **Forward computation:** $\hat{y}_t = (W_s + \alpha_t W_f) z_t$ for encoded input $z_t$.
- **Learning rule:** After observing $y_t$, fast Hebbian trace $W_f \leftarrow \lambda_t W_f + \eta_t (y_t-\hat{y}_t) z_t^T$ (local outer-product delta). Slow consolidation $W_s \leftarrow W_s + \beta g(W_f)$ periodically/conditionally.
- **Self-improvement mechanism:** $\lambda_t,\eta_t,\alpha_t$ modulated by prediction confidence/surprise so reliable contexts strengthen traces and noisy ones decay faster; consolidation triggered when fast trace consistently reduces error (self-validated).
- **Cost:** $O(od)$ per step.
- **Expected failure modes:** Interference in single matrix; related to fast weights/linear attention/Hebbian delta rules (novelty risk high).

## Candidate 5: Prototype Utility Field with Internal Replay Synthesis (PUF-IR)
- **Core idea:** Online prototype learner that synthesizes its own "replay" by rehearsing high-utility prototypes when surprise indicates potential forgetting, without storing raw data.
- **State/parameters:** Prototypes $(p_i, y_i, u_i, \Sigma_i)$ (centroid, label/value, utility, spread estimate).
- **Forward computation:** Nearest/probabilistic prototype prediction.
- **Learning rule:** Move nearest prototypes toward sample by local delta rule; create new prototype on high surprise; update utility by prediction contribution.
- **Self-improvement mechanism:** When incoming error on previously-good regions rises (detected internally by utility drops), model self-generates synthetic rehearsal samples from stored prototypes and performs consolidation updates — autonomous anti-forgetting loop using only internal state.
- **Cost:** $O(Pd)$ plus occasional rehearsal bursts bounded in count.
- **Expected failure modes:** Synthetic rehearsal may distort (model collapse in miniature); prototype statistics noisy; resembles generative replay/prototype rehearsal conceptually.

## Candidate 6: Adaptive Resonant Feature Lattice (ARFL)
- **Core idea:** A hierarchy-free lattice of feature units with resonance (match) criterion: inputs that resonate update units, mismatches trigger new units; resonance threshold self-tunes by prediction success.
- **State/parameters:** Units with weight vectors, activation counts, predictive links to outputs.
- **Forward computation:** Activate best-matching units above resonance; predict via predictive links.
- **Learning rule:** Resonant units move toward input (Hebbian); predictive links updated by local delta; mismatch allocates unit if capacity allows else replaces lowest-utility.
- **Self-improvement mechanism:** Vigilance/resonance threshold adjusted online to maintain target match rate and prediction accuracy (homeostatic), enabling regime shifts from exploration to exploitation autonomously.
- **Cost:** $O(Ud)$ for $U$ units.
- **Expected failure modes:** Very close to ART family (Grossberg) — high prior-art risk; must be distinguished mechanistically or killed/relabelled.

## Candidate 7: Quantized Local-Prediction Ensemble (QLPE)
- **Core idea:** Ensemble of tiny ternary/low-bit predictors trained by sign/local perceptron-like updates on different random feature subspaces; aggregation weights self-updated by recent correctness.
- **State/parameters:** $E$ predictors with weights in {-1,0,1} or small ints, random masks/projections, per-predictor reliability scores.
- **Forward computation:** $\hat{y}_e = f_e(m_e \odot \phi(x))$, aggregate by reliability-weighted vote/average.
- **Learning rule:** On error/surprise, update only mispredicting predictors with local sign/Hebbian-like rule; reinforce correct predictors' reliability.
- **Self-improvement mechanism:** Reliability scores and sparse participation adapted online; weak predictors replaced/reseeded from residual-driven random features when persistently wrong (self-repair).
- **Cost:** Very low arithmetic (ternary/additive), $O(E d_s)$.
- **Expected failure modes:** Limited capacity/expressivity; quantization noise; ensemble management heuristic.

## Candidate 8: Recursive Self-Modeling Linear Map (RSMLM)
- **Core idea:** Online linear predictor augmented with a learned "self-model" vector predicting its own future error from current state; updates allocate learning rate and feature reweighting based on predicted error (simple self-referential control), using only recursive least-squares-like local updates.
- **State/parameters:** Predictor weights $W$, error-model weights $w_e$, feature scaling vector $s$, running covariance/precision approximations (diagonal/low-rank).
- **Forward computation:** $\hat{y}=W(s\odot x)$; predicted error $\hat{e}=w_e^T \psi(x,\hat{y},h)$.
- **Learning rule:** RLS/online gradient-free ridge-style update for $W$ on prediction residual; local delta update for $w_e$ on observed vs predicted error; $s$ updated to up/down-weight features correlated with reduced error.
- **Self-improvement mechanism:** Self-model gates plasticity and feature scaling so the system learns when/where it is unreliable and autonomously reallocates adaptation budget — a bounded self-referential loop.
- **Cost:** $O(d^2)$ worst case but $O(d)$ with diagonal approximations.
- **Expected failure modes:** Self-model may be inaccurate/misleading; resembles meta-learned/adaptive LR heuristics; limited nonlinear capacity.

## Initial scoring (pre-implementation, to be revised after novelty checks)

| Candidate | Novelty risk | Plausibility | Impl. cost (2 vCPU) | Testability (<20m) | Expected upside | Notes |
|---|---:|---:|---:|---:|---:|---|
| C1 HSAM | Medium (ART/prototype/associative memory family) | Medium | Low | High | Medium | Clear autonomous loop, easy to test; must distinguish from ART/growing prototypes. |
| C2 DTEC-R | High (ESN+readout+memory incremental) | Medium | Low | High | Low-Med | Strong baseline-like; likely incremental. |
| C3 CSCG | Medium-High (online sparse coding/growing dictionaries known) | Medium | Medium | Medium | Medium | Interesting but trickier to stabilize. |
| C4 BFTP | High (fast weights/Hebbian outer products well known) | Medium | Low | High | Medium | Novelty likely low. |
| C5 PUF-IR | Medium | Medium | Medium | Medium | Medium | Self-synthesized rehearsal is distinctive angle but collapse risk. |
| C6 ARFL | Very High (ART very close) | Medium | Low | High | Low | Probably drop/relabel after check. |
| C7 QLPE | Medium | Low-Med | Low | High | Low-Med | Efficiency angle, limited capacity. |
| C8 RSMLM | Medium | Medium | Low | High | Medium | Simple self-referential control; linear capacity limit. |

**Plan:** Run candidate-specific novelty searches (>=8 each) for C1,C3,C4,C5,C6,C7,C8,C2 as prioritized; select top 3 implementable with best novelty/plausibility/testability balance. Provisional top-3 for deep implementation: **C1 HSAM**, **C5 PUF-IR** (if distinct enough), **C8 RSMLM** — pending search verdicts; will justify final choice in writing.
