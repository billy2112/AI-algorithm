# Self-Improvement

## What it does
- **Self-play / self-distillation:** agent improves from games against itself (AlphaZero) or distills its own predictions/ensembles.
- **Evolutionary / neuroevolution:** population-based search over weights/architectures (NEAT, evolution strategies); quality-diversity (MAP-Elites) seeks diverse high-performing solutions.
- **Open-ended search:** POET co-evolves environments and agents for continual novelty.
- **Gödel-machine / self-referential improvement:** theoretical self-modifying systems that rewrite themselves when they can prove improvement (Schmidhuber); practical approximations use empirical validation (Darwin Gödel Machine, Zhang et al. 2025 arXiv:2505.22954; Gödel Agent, Yin et al. 2024 arXiv:2410.04444).
- **Learned optimizers / meta-learned update rules:** learn update functions that generalize across tasks.
- **Neural architecture search (NAS):** automated search over architectures (RL, evolutionary, gradient-based).
- **"AI Scientist" / agentic loops:** LLM agents that propose/run/analyze experiments and iterate (often heavy external tooling/compute).

## Cost profile
Search/population/LLM-agent methods are typically compute-intensive (many evaluations, large models). In-weight self-improvement (a small model modifying itself online) can be cheap if the mechanism is local/simple, but demonstrating reliable improvement is hard.

## Known weaknesses
- Outer-loop search expensive; sample inefficiency.
- Defining reliable self-improvement signals without human labels/reward hacking.
- Provably beneficial self-modification intractable in general; empirical self-modification risks instability/regression.
- Open-ended systems may diverge or produce unproductive novelty.

## Key references
1. Schmidhuber, J. (2003). Gödel Machines: Fully Self-Referential Optimal Universal Self-Improvers.
2. Wang, R. et al. (2019). Paired Open-Ended Trailblazer (POET). GECCO.
3. Mouret, J.-B., & Clune, J. (2015). Illuminating search spaces by mapping elites (MAP-Elites). arXiv:1504.04909.
4. Zhang, J. et al. (2025). Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents. arXiv:2505.22954.
5. Yin, X. et al. (2024/2025). Gödel Agent: A Self-Referential Agent Framework for Recursively Self-Improvement. arXiv:2410.04444.
6. Andrychowicz, M. et al. (2016). Learning to learn by gradient descent by gradient descent. NeurIPS.
