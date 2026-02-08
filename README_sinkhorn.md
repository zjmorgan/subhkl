# Memory-Efficient Sinkhorn Indexing in subHKL

The `indexer_sinkhorn_jax` algorithm provides a differentiable, high-performance method for indexing diffraction peaks against a large pool of theoretical HKL reflections. It is designed to handle wide-bandwidth data and unknown orientations without the discrete "walls" of traditional indexers.

## Core Features

### 1. The Observer Rotation Trick
To avoid the $O(N_{obs} 	imes N_{hkl})$ memory cost of materializing a full cost matrix, we project the observer vectors into the crystal frame:
$$ 	ext{Cost}_{ij} = \hat{k}_{obs, i} \cdot (UB \, \mathbf{h}_j) = (\hat{k}_{obs, i} \, UB) \cdot \mathbf{h}_j $$
This allows us to compute dot products using a single matrix multiplication against the HKL pool, which is processed in efficient chunks using `jax.lax.scan`.

### 2. Multi-Scale Angular Kernel
To resolve the "vanishing gradient" problem common in high-precision indexing, we employ a multi-scale kernel that combines:
*   **Narrow Peak (vMF-like):** High precision for final refinement.
*   **Heavy-Tailed Background (Log-Cauchy):** Ensures a non-zero gradient even when the orientation is >10° off, providing a massive "capture range".

### 3. Log-Stable Softmax (Log-Sinkhorn)
Instead of hard-matching peaks to HKLs, we use a softmax-based probability distribution. This is equivalent to a one-iteration Sinkhorn step in log-space, which maintains differentiability and prevents numerical underflow in the tails of the distribution.

### 4. Soft-Masking and Penalties
Experimental constraints (wavelength bandwidth and resolution limits) are implemented as continuous soft penalties rather than hard cuts:
*   **Wavelength:** Gaussian penalty for reflections requiring $\lambda$ outside $[\lambda_{min}, \lambda_{max}]$.
*   **Resolution:** Penalties for reflections outside the $d_{min}/d_{max}$ range.
These penalties guide the optimizer back into the valid physical regime without creating discontinuous "dead zones".

### 5. Resolution-Aware Top-K
To prevent the indexer from becoming "blind" to high-resolution data (the Resolution Wall), the HKL pool size is dynamically determined by the observed scattering vectors. The `top_k` selection logic includes a resolution-based tie-breaker to ensure a diverse set of candidate reflections are considered for matching.

## Optimization Flow
1. **HKL Pool Generation:** Robustly determined from cell parameters and peak resolution.
2. **Block-wise Selection:** Finds the top $K$ candidates per peak using the rotation trick.
3. **Soft-Matching:** Computes Log-Sinkhorn probabilities using multi-scale kernels.
4. **Scoring:** The final loss is the negative weighted log-likelihood of the matches, suitable for Gradient Descent.
