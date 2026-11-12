# Mechanistic comparison: HSAM vs ART/Fuzzy ART vs GNG-U vs SLDA (digits)

**Status:** Attempted full head-to-head runs; some runs were resource/timeout constrained in this environment. Results below are aggregated from completed artifacts where available, plus mechanistic diff.

## Mechanistic diff (concrete)

| Aspect | HSAM | Fuzzy ART (implemented) | GNG-U (implemented) | SLDA (streaming) |
|---|---|---|---|---|

| Representation | Fixed random projection + unit-normalized keys in slots | Complement-coded input, category prototypes (weights) | Prototype vectors per unit (neural gas) | Class means + regularized pooled covariance |

| Learning rule | Local delta to winner keys/values; utility updates | Fuzzy ART learning: `w = beta * min(xcc,w) + (1-beta)*w` on resonance | Hebbian/competitive update to nearest units; error-based utility; insertion between high-error units | Online mean updates per class; running scatter/covariance update |

| Allocation/growth | Surprise/mismatch/low-sim gated allocation to empty/least-utility slot | Vigilance test `|x ∩ w|/|x| >= rho` → resonate else create new category | Insert node periodically between worst-error unit and neighbor if under max nodes; utility-based deletion heuristic | No structural growth; fixed model structure (means per seen class) |

| Forgetting/competition | k-winner softmax retrieval + utility decay/replacement | Choice function + vigilance competition across categories | Nearest-neighbor competition; age/utility heuristics | Nearest class-mean in Mahalanobis/LDA space; no explicit forgetting mechanism beyond online means |

| Homeostasis/self-tuning | Endogenous EMA of error/improvement modulates eta/theta (bounded) | Fixed rho/beta/alpha in this impl (no strong self-tuning) | Fixed rates/heuristics | Fixed shrinkage |


## Verdict on novelty vs these baselines

**Mechanistic assessment:** HSAM is *not* structurally impossible for ART/GNG-U/SLDA — all are prototype/associative memories with some form of competition and adaptation. The specific combination (fixed random projection + k-winner value readout + surprise/mismatch/low-sim composite gate + utility replacement + endogenous homeostatic controller) is a particular design point. However, the gating/insertion logic is closely analogous to ART vigilance/resonance and GNG insertion/deletion principles. **If forced to choose:** this reads as a reparameterization/integration of known ideas rather than a mechanism these families *structurally cannot* express. The "homeostatic self-improvement" framing is not supported by the ablation evidence shown earlier (no_tune/no_alloc perform comparably in tested streams). I would treat the novel contribution as *incremental at best* unless future tests show a clear, mechanism-attributable advantage under distribution shift.

## Non-stationary (distribution-shift) stream test

Stream: noise corruption + partial label/class mixing mid-stream (chunks 10-40), fixed held-out test (5 seeds).

| mode | first acc | last acc | delta | peak acc |
|---|---:|---:|---:|---:|
| frozen_after_1 | 0.8472±0.0373 | 0.8472±0.0373 | 0.0000±0.0000 | 0.8472±0.0373 |
| full | 0.8472±0.0373 | 0.9367±0.0141 | 0.0894±0.0403 | 0.9556±0.0083 |
| no_alloc | 0.8472±0.0373 | 0.9456±0.0105 | 0.0983±0.0413 | 0.9506±0.0087 |
| no_tune | 0.8472±0.0373 | 0.9356±0.0223 | 0.0883±0.0322 | 0.9528±0.0081 |

**Raw result summary (bootstrap):** full delta +0.0894, no_tune +0.0883, no_alloc +0.0983, frozen_after_1 0.0. **Gap still does not open meaningfully** — mechanism remains neutral/indistinguishable under this distribution-shift test. Mechanism should not be framed as the novel/causal factor; treat as dead weight for "self-improvement" claims.

## Component-by-component mapping (falsifiable)

| HSAM component | Closest in Fuzzy ART | Closest in GNG-U | Closest in SLDA | Identical vs differs |
|---|---|---|---|---|
| **Fixed random projection** | None (inputs used directly / complement-coded). ART operates on input patterns (often complement-coded). | None / raw input features; GNG-U uses input space directly for distances. | None / raw/feature space; SLDA uses features directly in LDA space. | **Different.** HSAM adds a fixed random projection (JL-like) before keying; ART/GNG-U/SLDA do not require this. This is a preprocessing choice, not inherently novel. |
| **k-winner delta update (keys/values)** | Category weight update on resonance: `w = beta*min(xcc,w) + (1-beta)*w` (updates prototype to match). Single winning category typically updates. | Hebbian/competitive update to nearest (and sometimes second-nearest) units: `w += eps*(x-w)`. | Class mean online update: `mu_y = (mu_y*(n-1)+x)/n`. | **Partly similar.** All do local prototype updates to winners; HSAM does *k*-winner soft updates to both keys and values with a delta rule + utility weighting. ART/GNG-U update weights/positions; SLDA updates means. The multi-winner + value-associative form is a small variation. |
| **Surprise-gated allocation** | Vigilance test: if `match >= rho` resonate/update existing category, else create new category. Allocation gated by match/vigilance. | Node insertion when error/age criteria met (e.g. every lambda steps insert between high-error unit and neighbor) and utility-based deletion. | No structural allocation; fixed class set (or grows by adding classes on first sight). | **Very close.** Functionally analogous to ART vigilance/resonance and GNG insertion heuristics. HSAM uses surprise/error-EMA + mismatch + low-similarity as composite gate vs ART's match/vigilance or GNG's error-driven insertion. **Not clearly distinct mechanism.** |
| **Utility replacement** | No explicit utility-based replacement of categories; categories persist once created (with weight updates). | Utility-based deletion of low-utility units; also age/edge pruning heuristics. | No replacement mechanism. | **Closest to GNG-U.** HSAM replaces least-utility filled slot on allocation pressure; GNG-U deletes low-utility nodes. Similar idea, different policy details. |
| **Homeostatic tuning** | Fixed vigilance/params in basic ART (adaptive vigilance variants exist in literature, but not the canonical form). | Fixed rates/thresholds in standard GNG-U. | Fixed shrinkage. | **Weakest link.** Bounded adaptation of learning rates/thresholds from error/improvement EMAs. Literature has adaptive-vigilance ART variants (see novelty search), so not unprecedented. Empirically **neutral/dead weight** across tested streams/grids (no_tune ≈ full). | 

**If someone removed the single most novel-looking component, which baseline would HSAM become equivalent to?**  

If we removed **homeostatic tuning** (treat as fixed), HSAM becomes effectively a prototype/slot memory with random projection, k-winner local updates, surprise/vigilance-like allocation, and utility replacement - most equivalent to a **GNG-U-like** or **ART-like** incremental prototype classifier (leaning toward GNG-U due to explicit utility replacement). If we also stripped down allocation heuristics to pure nearest-prototype updates, it would become closer to a **nearest-class-mean / SLDA-like** prototype learner. The random projection is cosmetic/preprocessing; the core is **prototype competition + incremental allocation/pruning**. Given this, the honest novelty ceiling is **low** - HSAM is best viewed as an integration/reparameterization of ART/GNG-U ideas rather than a mechanism these families structurally cannot express.
