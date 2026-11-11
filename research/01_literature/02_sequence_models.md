# Sequence Models

## Scope
Transformers, linear attention, state-space models (S4, Mamba), RWKV, RetNet, xLSTM, Hyena, liquid/neural ODE networks.

## What it does
- **Transformers:** self-attention over all tokens; strong modeling but O(T^2) time/memory in sequence length for standard attention.
- **Linear attention / kernelized attention:** reformulate attention to achieve linear complexity via associative feature maps / recurrent state (Katharopoulos et al.; Performer; GLA Yang et al. 2024 ICML).
- **State-space models (SSM):** continuous-time state dynamics discretized for sequences. S4 (Gu et al.) uses structured HiPPO matrices for long-range dependencies; Mamba/S6 (Gu & Dao 2023/24, arXiv:2312.00752) adds input-dependent selective parameters and hardware-aware scan, enabling content-based reasoning with linear-time inference.
- **RWKV:** RNN-like reformulation with parallelizable training and efficient inference, using time-mixing/channel-mixing with receptance/gating.
- **RetNet:** retention mechanism with parallel, recurrent, and chunkwise forms; linear inference.
- **xLSTM:** extended LSTM with exponential gating/memory mixing to improve scaling/long context.
- **Hyena:** implicit long convolutions + gating, subquadratic.
- **Liquid/Neural ODE networks:** continuous-time dynamics; adaptive time constants (Liquid Time-Constant networks) but training/inference can be costly due to ODE solvers.

## Cost profile
SSM/linear-attention/RWKV/RetNet: linear scaling in sequence length during training in efficient implementations, constant memory/state for recurrent inference (no KV cache growth like Transformers). Mamba shows strong throughput on GPUs; on CPU, small models still efficient per step but absolute speed depends on implementation.

## Known weaknesses
- Standard training is gradient-based/backprop through time/sequence (high data/compute for large models).
- Long exact-copy/associative recall can be challenging for fixed-state recurrent/SSM models vs attention (active research area).
- Selective SSMs improve content selection but remain within conventional training paradigms.

## Key references
1. Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
2. Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces (S4). ICLR.
3. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
4. Peng, B. et al. (2023). RWKV: Reinventing RNNs for the Transformer Era. arXiv:2305.13048.
5. Sun, Y. et al. (2023). Retentive Network: A Successor to Transformer. arXiv:2307.08621.
6. Yang, S. et al. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML.
7. Beck, M. et al. (2024). xLSTM: Extended Long Short-Term Memory. arXiv:2405.04517.
8. Poli, M. et al. (2023). Hyena Hierarchy. ICML.
