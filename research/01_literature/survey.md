# Phase 1 - Literature and Prior-Art Survey

**Date:** 2026-11-10  
**Method:** Web search tool queries (recorded in `research/novelty/search_log.md`) plus general domain knowledge. I cite sources I actually retrieved in search results; where I rely on well-known background not directly returned, I note it. I do not claim exhaustive coverage.

## 1. Backprop-free / local learning
**What it does:** Replaces end-to-end backpropagation with local updates (Hebbian/STDP, predictive coding error minimization, forward-only losses, contrastive local objectives, closed-form/local regression). Examples: Forward-Forward (Hinton), predictive coding networks (Rao & Ballard lineage; recent CL surveys e.g. Safa 2024, arXiv:2407.17305 / DOI 10.1088/2634-4386/ada08b), equilibrium propagation (Scellier & Bengio), target/difference target propagation, feedback/direct feedback alignment, e-prop for recurrent/spiking nets, reservoir computing / echo state networks / extreme learning machines (fixed random reservoir + linear readout). Recent: Forward Projection (O'Shea & Rajendran, arXiv:2501.16476), Mono-Forward (arXiv:2501.09238).  
**Cost profile:** Often lower memory than backprop-through-time/graph (no activation storage for global gradients) and can be parallel/local; some methods (equilibrium prop) require settling dynamics. ESN/ELM extremely cheap training (closed-form/one-pass readout) but limited adaptation of representation.  
**Known weaknesses:** Accuracy gaps on hard tasks vs backprop; stability/credit assignment over depth/time; scalability; hyperparameter sensitivity. Local rules can struggle with hierarchical credit assignment.  
**Key refs (from searches/background):** Safa 2024 (Hebbian/PC/STDP CL survey); O'Shea & Rajendran 2025 arXiv:2501.16476; Scellier & Bengio 2017 equilibrium propagation; Lillicrap et al. feedback alignment; Hinton 2022 Forward-Forward; Bellec et al. e-prop; Jaeger ESN / Huang ELM.  

## 2. Sequence models
**What it does:** Transformers (attention, quadratic in context), linear attention/RNN reformulations, state-space models S4/S5/S6/Mamba (selective state spaces, linear-time inference) (Gu & Dao, arXiv:2312.00752), RWKV, RetNet, xLSTM, Hyena, liquid/neural ODE networks.  
**Cost profile:** Transformers high memory/compute for long context; SSM/linear-attention families offer linear/near-linear scaling and constant-time recurrent inference in some forms (Mamba, RWKV).  
**Known weaknesses:** Training usually backprop/large data; long-range exact recall tradeoffs; Mamba selectivity helps but still standard gradient training.  
**Key refs:** Vaswani et al. 2017 Transformer; Gu & Dao 2023/24 Mamba; Peng et al. RWKV; Sun et al. RetNet; Yang et al. 2024 Gated Linear Attention (ICML 2024); Poli et al. Hyena.  

## 3. Test-time learning and fast weights
**What it does:** Adapts parameters at inference: test-time training/adaptation (Sun et al. TTT layers arXiv:2407.04620; end-to-end TTT arXiv:2512.23675), fast weight programmers (Schmidhuber lineage), modern Hopfield networks, meta-learning (MAML, Reptile) to learn good initializations/update rules, in-context learning in Transformers. Titans (Behrouz et al. arXiv:2501.00663) adds neural long-term memory.  
**Cost profile:** Inner-loop gradient updates add test-time compute/memory; meta-learning expensive at train time but can yield cheap adaptation if designed well.  
**Known weaknesses:** Stability/forgetting during adaptation, choice of self-supervised objective, compute overhead at test time, need for meta-training in many setups.  
**Key refs:** Sun et al. 2020/2024 TTT; Behrouz et al. 2024 Titans; Ramsauer et al. modern Hopfield; Finn et al. MAML; Nichol et al. Reptile; Schlag et al. fast weight programmers.  

## 4. Memory / continual learning
**What it does:** Mitigates catastrophic forgetting: regularization (EWC, SI, MAS), replay/generative replay, parameter isolation/masks, complementary learning systems / dual memory (CLS-inspired, e.g. CLS-ER), sparse distributed representations, hyperdimensional computing (HDC/VSA), Numenta HTM/thousand-brains ideas, progressive/modular networks.  
**Cost profile:** Replay stores data (memory/privacy cost); regularization stores Fisher/importance; modular methods grow parameters. HDC can be efficient with simple algebraic ops.  
**Known weaknesses:** Task-ID dependence, replay buffer needs, plasticity-stability tradeoff, scalability. Recent works (UPGD, ICLR 2024 arXiv:2404.00781) target plasticity+forgetting jointly.  
**Key refs:** Kirkpatrick et al. EWC; Zenke et al. SI; Rolnick et al. replay; McClelland et al. CLS; Kanerva HDC; Hawkins HTM; Elsayed & Mahmood 2024 UPGD.  

## 5. Efficiency
**What it does:** Reduce compute/memory/energy: quantization (binary/ternary, BitNet b1.58 arXiv:2402.17764), pruning/lottery tickets, sparsity/MoE, early exit, event-driven/spiking networks.  
**Cost profile:** 1-bit/ternary can drastically cut multiply cost and memory bandwidth; sparsity/event-driven reduces active ops. Gains depend on hardware/software support.  
**Known weaknesses:** Accuracy at extreme quantization, training stability, hardware mismatch on CPUs/GPUs, conversion/training complexity for SNNs.  
**Key refs:** Ma et al. 2024 BitNet b1.58; Frankle & Carbin lottery ticket; Fedus et al. Switch/MoE; Roy et al. spiking survey.  

## 6. Self-improvement
**What it does:** Systems that improve via self-play, self-distillation, evolutionary/neuroevolution, open-ended search (POET, MAP-Elites/quality-diversity), Gödel-machine/self-referential ideas, learned optimizers, neural architecture search, "AI Scientist"/agent self-improvement (e.g. Darwin Gödel Machine arXiv:2505.22954, Gödel Agent arXiv:2410.04444).  
**Cost profile:** Often very high (population/search/LLM calls); self-distillation moderate. True in-situ self-improvement of a small model is rare and constrained.  
**Known weaknesses:** Search cost, safety/validation, defining improvement signal without external labels, collapse/instability.  
**Key refs:** Schmidhuber Gödel machine; Wang et al. POET; Mouret & Clune MAP-Elites; Zhang et al. 2025 DGM; Yin et al. 2024/25 Gödel Agent; Andrychowicz et al. learned optimizers.  

## 7. Theoretical grounding
**What it does:** Principles: free-energy principle / active inference (Friston), minimum description length/compression (MDL/Solomonoff-inspired), information bottleneck, NTK/mean-field analyses, scaling laws.  
**Cost profile:** N/A (frameworks); compression objectives can be operationalized cheaply in restricted forms.  
**Known weaknesses:** Many principles hard to instantiate exactly at scale; approximations may lose guarantees.  
**Key refs:** Friston free-energy principle; Rissanen MDL; Tishby information bottleneck; Jacot et al. NTK; Kaplan/Hoffmann scaling laws.  

## Implications for this project
CPU-only, <20 min/experiment, <6GB RAM strongly favors small models, online/local/one-pass or few-pass updates, simple objectives (prediction error, reconstruction, local Hebbian/least-squares), and tiny datasets/streams. A credible novelty claim must be mechanistic and checked against closest families: TTT/fast weights (gradient inner loops), ELM/reservoir+readout, prototype/ART/Hebbian associative memories, HDC, predictive coding local rules, and self-modifying/meta-learned update rules. Any "self-improvement" must be demonstrated as autonomous post-deployment improvement from the model's own mechanism without external retuning.
