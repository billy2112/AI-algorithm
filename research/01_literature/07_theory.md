# Theoretical Grounding

## What it does
- **Free-energy principle / active inference (Friston):** agents minimize variational free energy (surprise bound) via perception and action; unifies inference and learning under Bayesian/thermodynamic-inspired objectives.
- **Minimum Description Length (MDL) / compression:** model selection and learning as data compression (Rissanen); Solomonoff induction as idealized Bayesian compression (incomputable in general). Practical proxies: predictive coding length, regularization.
- **Information bottleneck (Tishby):** representations should compress input while preserving relevant information about targets.
- **NTK / mean-field theory:** analyze neural network training dynamics in infinite-width or other limits; provide convergence/expressivity insights but often far from practical finite regimes.
- **Scaling laws:** empirical power-law relationships between loss and model/data/compute (Kaplan et al.; Hoffmann/Chinchilla).

## Cost profile
Theoretical frameworks guide objectives; operationalization varies. Compression/prediction-error objectives can be computed online cheaply; exact Bayesian/Solomonoff methods infeasible.

## Known weaknesses
- Gaps between asymptotic theory (NTK/mean-field) and finite, practical networks.
- Free-energy/active inference implementations require modeling choices and approximations.
- MDL optimal codes unknown for complex data; proxies may not guarantee desired generalization.
- Scaling laws descriptive, not mechanistic guarantees for small/custom architectures.

## Key references
1. Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience.
2. Rissanen, J. (1978). Modeling by shortest data description. Automatica.
3. Tishby, N., Pereira, F., & Bialek, W. (1999). The Information Bottleneck Method. arXiv:physics/0004057.
4. Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural Tangent Kernel. NeurIPS.
5. Kaplan, J. et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.
6. Hoffmann, J. et al. (2022). Training Compute-Optimal Large Language Models (Chinchilla). arXiv:2203.15556.
