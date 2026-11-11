# Test-Time Learning, Fast Weights, Meta-Learning, and In-Context Mechanisms

## What it does
- **Test-Time Training (TTT):** adapts model parameters at inference using a self-supervised objective on the test/context sequence (Sun et al. 2020). TTT layers (Sun et al. 2024, arXiv:2407.04620 "Learning to (Learn at Test Time)") replace/augment sequence-model hidden states with fast weights updated by gradient steps during forward/test; end-to-end TTT/meta-learning variants (arXiv:2512.23675) learn initializations for test-time adaptation.
- **Fast weight programmers:** maintain rapidly changing "fast" weights (associations) modulated/programmed by a "slow" network (Schmidhuber 1992; Schlag et al.). Linear attention can be viewed as fast-weight memory; modern variants unify attention/RNN/SSM views.
- **Modern Hopfield networks:** continuous/differentiable associative memories with exponential storage capacity in dimension; attention-like retrieval (Ramsauer et al. 2020).
- **Meta-learning:** MAML (Finn et al.) learns initialization enabling few gradient steps for new tasks; Reptile uses simpler first-order updates. Enables rapid adaptation but requires meta-training distribution.
- **In-context learning (ICL):** Transformers adapt behavior from prompt examples without weight updates; mechanistic interpretations link to implicit gradient descent/mesa-optimization in some settings.

## Cost profile
TTT/fast-weights incur extra compute at test time for inner-loop updates (can be chunked/parallelized in some implementations). Meta-training (MAML) expensive (second-order/through-optimization) but amortizes adaptation. Hopfield retrieval efficient for moderate sizes.

## Known weaknesses
- Stability and catastrophic forgetting during test-time updates; choice of self-supervised objective critical.
- Compute overhead and need to manage update frequency/learning rate at test time.
- MAML/meta-learning assumes task distribution; may not generalize to arbitrary streams.

## Key references
1. Sun, Y. et al. (2020). Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. ICML.
2. Sun, Y. et al. (2024). Learning to (Learn at Test Time): RNNs with Expressive Hidden States. arXiv:2407.04620.
3. Tandon et al. / Sun et al. (2025). End-to-End Test-Time Training for Long Context. arXiv:2512.23675.
4. Behrouz, A., Zhong, P., & Mirrokni, V. (2024). Titans: Learning to Memorize at Test Time. arXiv:2501.00663.
5. Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. ICML.
6. Ramsauer, H. et al. (2020). Hopfield Networks is All You Need. ICLR.
7. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning (MAML). ICML.
8. Nichol, A., Achiam, J., & Schulman, J. (2018). On First-Order Meta-Learning Algorithms (Reptile). arXiv:1803.02999.
