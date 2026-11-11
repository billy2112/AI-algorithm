# Memory and Continual Learning

## What it does
- **Regularization-based:** constrain important parameters from changing. EWC (Kirkpatrick et al.) uses Fisher information; SI (Zenke et al.) tracks online importance; MAS uses output sensitivity. GEAR (Kong et al.) analyzes Hessian eigenvalues.
- **Replay:** store/replay past samples (experience replay) or generate pseudo-samples (generative replay). Effective but needs memory/buffers; privacy concerns.
- **Parameter isolation / dynamic architectures:** allocate/task-specific masks or grow modules (PackNet, progressive networks, HAT). Can avoid interference but may require task IDs/capacity growth.
- **Complementary Learning Systems (CLS):** fast hippocampal-like episodic learning + slow neocortical consolidation; inspires dual-memory methods (e.g. CLS-ER, dual semantic memories with EMA).
- **Sparse distributed / associative memory:** sparse coding, Hopfield/SDM (Kanerva), prototype memories; hyperdimensional computing (HDC/VSA) uses high-dimensional random vectors with algebraic bind/bundle operations for robust, efficient associative storage.
- **HTM / Thousand Brains (Numenta):** sparse distributed representations, sequence memory, sensorimotor models; biologically motivated but specialized architectures.

## Cost profile
Regularization cheap in storage (importance vectors/matrices approximations) but may limit plasticity. Replay adds memory and compute for rehearsing. HDC operations can be lightweight (bitwise/XOR/bundling) and one-pass. Modular growth increases parameter count.

## Known weaknesses
- Stability-plasticity dilemma: preventing forgetting can reduce ability to learn new tasks (loss of plasticity, Elsayed & Mahmood UPGD ICLR 2024 arXiv:2404.00781 addresses both).
- Task boundaries/IDs often assumed; task-free/online class-incremental settings harder.
- Replay buffer size/selection critical; generative replay adds model complexity.

## Key references
1. Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks (EWC). PNAS.
2. Zenke, F., Poole, B., & Ganguli, S. (2017). Continual Learning Through Synaptic Intelligence. ICML.
3. Rolnick, D. et al. (2019). Experience Replay for Continual Learning. NeurIPS.
4. McClelland, J., McNaughton, B., & O'Reilly, R. (1995). Why there are complementary learning systems. Psychological Review.
5. Arani, E., Sarfraz, F., & Zonooz, B. (2022). Learning Fast, Learning Slow: CLS-ER. ICLR.
6. Kanerva, P. (1988/2009). Sparse Distributed Memory; Hyperdimensional Computing.
7. Hawkins, J. et al. (2016+). A Framework for Intelligence and Cortical Function (HTM/Thousand Brains).
8. Elsayed, M., & Mahmood, A. R. (2024). Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning (UPGD). ICLR. arXiv:2404.00781.
