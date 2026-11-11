# Efficiency

## What it does
- **Quantization:** reduce numeric precision (8/4/2/1-bit). Binary/ternary networks replace multiplications with adds/XNOR-popcount; BitNet b1.58 (Ma et al. 2024, arXiv:2402.17764) uses ternary {-1,0,1} weights matching FP performance in reported LLM settings with major latency/memory/energy gains on specialized kernels.
- **Pruning / Lottery Ticket Hypothesis:** remove redundant weights/structures; sparse subnetworks can train effectively (Frankle & Carbin).
- **Mixture-of-Experts (MoE):** activate sparse subset of experts per input, increasing capacity with limited active compute.
- **Early exit / adaptive computation:** stop inference early when confident.
- **Spiking Neural Networks (SNNs):** event-driven sparse computation, temporal coding; efficient on neuromorphic hardware but training/accuracy gaps on conventional hardware.

## Cost profile
Quantization reduces memory bandwidth and can simplify arithmetic (especially 1/1.58-bit on supporting hardware). Sparsity reduces FLOPs if exploited. SNNs reduce active operations when spike rates low; on CPUs without event-driven execution, benefits may not materialize.

## Known weaknesses
- Accuracy degradation at extreme low-bit without careful QAT/architecture design.
- Hardware/software support required for real speed/energy gains (CPU kernels for bitnets exist but specialized).
- SNN training (surrogate gradients, conversion) can be complex; latency/accuracy tradeoffs.

## Key references
1. Ma, S. et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58). arXiv:2402.17764.
2. Wang, H. et al. (2023). BitNet: Scaling 1-bit Transformers for Large Language Models. arXiv:2310.11453.
3. Frankle, J., & Carbin, M. (2019). The Lottery Ticket Hypothesis. ICLR.
4. Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers. JMLR.
5. Roy, K., Jaiswal, A., & Panda, P. (2019). Towards spike-based machine intelligence with neuromorphic computing. Nature.
