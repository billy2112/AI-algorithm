# Theory and Mechanistic Analysis (HSAM, preliminary)

**Status:** [CONJECTURE] / informal analysis unless stated otherwise. No formal guarantees are proven here.

## 1. Model summary
HSAM (Homeostatic Surprise-Gated Associative Memory) maintains $M$ slots, each with key $k_i \in \mathbb{R}^p$ (unit-normalized), value $v_i \in \mathbb{R}^o$ (or scalar), utility $u_i$, and fill mask. Input $x$ is encoded by a fixed random projection and normalization: $z = \mathrm{norm}(P x) \in \mathbb{R}^p$. Retrieval uses top-$k_w$ cosine similarities over filled slots with softmax weights $w_i \propto \exp(s_i)$, prediction $\hat y = \sum_{i\in W} w_i v_i$.

Learning is local and online:
- Compute error $e_t = \|y_t - \hat y_t\|$ and endogenous statistics: $\bar e_t = (1-\alpha)\bar e_{t-1} + \alpha e_t$, surprise $\sigma_t = e_t / (\bar e_t + \epsilon)$, improvement EMA $\Delta_t$ from error changes.
- **Allocation/growth:** if no filled slots, or $\sigma_t > \theta_{high}$, or predicted label mismatches target (classification), or max similarity below a low-similarity threshold, allocate/write a slot (empty slot or lowest-utility replacement) with $k_j \leftarrow z_t, v_j \leftarrow y_t, u_j \leftarrow u_0$.
- **Consolidation/update:** otherwise apply local delta/Hebbian-like updates to winners:
  $$k_i \leftarrow \mathrm{norm}((1-\eta_k w_i)k_i + \eta_k w_i z_t), \quad v_i \leftarrow v_i + \eta_v w_i (y_t - v_i),$$
  and increase utility proportional to winner weight and low-surprise confidence.
- **Homeostatic tuning (bounded):** periodically adjust $(\eta_k,\eta_v,\theta_{low/high})$ using $\Delta_t$: positive improvement tends to stabilize (reduce rates/tighten in limited way), negative/stagnant improvement increases plasticity/sensitivity within bounds. Utilities also decay globally.

## 2. Why this can learn (intuition)
- **Associative/prototype interpolation:** with normalized keys and softmax-weighted value readout, HSAM acts like a local kernel/prototype regressor-classifier in projected space. Delta updates move values toward targets and keys toward inputs for active regions, reducing local prediction error when the same region is revisited.
- **Capacity allocation:** surprise/mismatch/novelty gating creates new slots for poorly explained or novel inputs, analogous in role (not identical in mechanism) to ART vigilance commitment or GNG insertion, helping plasticity for new patterns.
- **Forgetting control (structural):** in class-incremental split-digits tasks, allocating/keeping class-specific or region-specific prototypes can reduce interference compared to a single shared MLP updated by gradient descent without replay, because past slots need not be overwritten unless utility-based replacement is forced.
- **Efficiency:** per-step cost is dominated by similarity scan and winner updates, $O(Mp + k_w(p+o))$, with small fixed $M,p,o,k_w$ in experiments (CPU-fast, low memory).

## 3. Why "self-improvement" is not established mechanistically
The implementation includes an endogenous controller (error/improvement EMAs modulating rates/thresholds), which is a form of autonomous parameter adaptation. However, empirical stream ablations (`research/03_experiments/hsam_self_improve_stream/results.json`, `hsam_robustness/results.json`) show:
- `full` improves over stream vs `frozen_after_1`, but `no_tune` and `no_alloc` variants show similar or slightly larger deltas in the tested settings.
- This indicates the dominant effect is **continued online local learning/updating from additional samples**, not demonstrably the homeostatic meta-controller or allocation policy.

Therefore, claims should distinguish: (1) online continual adaptation [supported in limited tests] from (2) autonomous meta-level self-improvement of learning rules/thresholds yielding superior long-horizon gains [not supported by current ablations].

## 4. Failure modes and risks
- **Interference under capacity pressure:** when slots saturate, lowest-utility replacement can discard useful prototypes (capacity-quality tradeoff; see ablation slot sweeps).
- **Metric/representation dependence:** fixed random projection may be insufficient for complex structure; performance can degrade on tasks requiring learned hierarchical features.
- **Allocation pathologies:** excessive allocation on noise, or insufficient allocation for subtle novelty, depending on thresholds/statistics.
- **Controller instability/irrelevance:** homeostatic updates are heuristic and bounded but not proven to improve convergence; may be neutral or harmful in some regimes.
- **Continual-learning confounds:** strong split-digits results may partly reflect separability and task structure; must be validated across task orders, harder datasets, and stronger CL baselines before broad claims.

## 5. Relation to prior art (positioning)
- **ART:** shared notion of match/vigilance-like gating for update vs new commitment and local prototype adaptation; HSAM differs in exact similarity/readout, surprise+error-EMA controller, utility decay, and k-winner soft assignment, but overlap is substantial.
- **GNG/SOINN family:** shared incremental prototype allocation and utility/pruning ideas; HSAM lacks explicit topological edge graph and uses associative value prediction slots.
- **ELM/reservoir readouts:** fixed random projection is shared in spirit, but readout is prototype-memory with local online updates rather than closed-form/global linear solve.
- **TTT/fast weights:** both adapt at test/stream time; TTT/fast-weights are typically gradient-based parametric updates (often meta-learned), HSAM uses local non-gradient associative slot updates.

## 6. Testable predictions / next theory steps
- Derive or empirically characterize conditions where prototype allocation reduces interference vs shared-weight SGD on split tasks.
- Analyze sensitivity to $(M,k_w,\theta,\eta)$ and stability of endogenous controller.
- Compare against an ART-like vigilance-only baseline and nearest-prototype (1-NN memory) controls to isolate contributions.
- Formalize memory-capacity/forgetting bounds only if a simplified variant admits analysis [not done].
