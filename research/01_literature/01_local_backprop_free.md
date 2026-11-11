# Backprop-free / Local Learning

## Scope
Forward-Forward, predictive coding, equilibrium propagation, target propagation, feedback alignment, Hebbian/STDP/BCM, e-prop, reservoir computing/ESN, extreme learning machines (ELM).

## What it does
- **Hebbian/STDP/BCM:** synaptic updates based on local pre/post activity correlations/timing; biologically inspired, no global error transport. STDP common in SNNs (Safa 2024 survey links Hebbian/STDP with sparse/predictive coding for continual learning).
- **Predictive coding (PC):** hierarchical generative model minimizing local prediction errors via iterative inference and local weight updates; can approximate backprop in some settings but implemented with local errors (Rao & Ballard; Whittington & Bogacz; recent surveys/discussions arXiv:2308.07870).
- **Forward-Forward (FF):** Hinton (2022) replaces forward/backward passes with two forward passes (positive/negative data) and local goodness objectives per layer.
- **Equilibrium propagation:** energy-based; learns by comparing free/ nudged equilibria, yielding local updates approximating gradients (Scellier & Bengio 2017).
- **Target propagation / difference target propagation:** computes layer-wise targets via approximate inverses instead of gradients (Lee et al.; O'Shea & Rajendran 2025 discuss relation in Forward Projection).
- **Feedback alignment / direct feedback alignment:** uses fixed/random feedback weights to convey error signals, avoiding symmetric weight transport (Lillicrap et al.; Nøkland).
- **e-prop:** eligibility propagation for online/local learning in recurrent/spiking networks using local eligibility traces and learning signals.
- **Reservoir computing / ESN / LSM:** fixed recurrent/random dynamical reservoir, train only linear readout (often ridge regression) – extremely cheap training, one-pass/closed-form possible.
- **ELM:** random hidden layer + analytic linear solve for output weights.

## Cost profile
Local rules reduce need to store activations for global backprop; can be memory-efficient and parallelizable per-layer/neuron. ESN/ELM: training cost dominated by readout solve (small matrix ops) – near-zero relative to deep backprop for small tasks. Equilibrium/iterative PC require settling iterations.

## Known weaknesses
- Often lower accuracy than backprop on complex benchmarks; depth credit assignment remains challenging.
- Stability, hyperparameter sensitivity, and scaling issues.
- ESN/ELM: representation fixed/random, limited feature learning; performance depends on reservoir design.

## Key references (selected)
1. Hinton, G. (2022). The Forward-Forward Algorithm. arXiv:2212.13345.
2. Scellier, B., & Bengio, Y. (2017). Equilibrium Propagation. Frontiers in Computational Neuroscience.
3. Lillicrap, T. et al. (2016). Random synaptic feedback weights support error backpropagation. Nature Communications.
4. Whittington, J., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network. Neural Computation.
5. Safa, A. (2024). Continual learning with Hebbian plasticity in sparse and predictive coding networks: a survey and perspective. Neuromorphic Computing and Engineering, 4(4). arXiv:2407.17305.
6. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. GMD Report.
7. Huang, G.-B. et al. (2006). Extreme learning machine. Neurocomputing.
8. O'Shea, R., & Rajendran, B. (2025). Closed-Form Feedback-Free Learning with Forward Projection. arXiv:2501.16476.
