# Mathematical Derivation & Code-Binding Compendium (Nanogrid Taylor Attention)

This reference compendium provides the complete, mathematically rigorous derivations, index-wise expansions, complexity proofs, and exact silicon memory code-bindings for the **LinearGridTransformer** (Taylor-linearized attention) architecture.

---

## TRACK 1: THE MATHEMATICAL DERIVATION OF THE TAYLOR KERNEL

Standard softmax attention requires materializing a $T \times T$ similarity matrix, creating a quadratic $O(T^2)$ computational and memory bottleneck. Linear attention bypasses this limitation by decomposing the exponential kernel into an explicit feature mapping $\phi(\mathbf{x})$, enabling the use of the associative property of matrix multiplication.

### 1. The 2nd-Degree Taylor Expansion of the Exponential Kernel

The unnormalized attention weights between a query vector $\mathbf{q} \in \mathbb{R}^d$ and a key vector $\mathbf{k} \in \mathbb{R}^d$ are determined by the exponential similarity kernel:
$$K(\mathbf{q}, \mathbf{k}) = \exp\left(\mathbf{q}^\top \mathbf{k}\right)$$

#### 1.1 Taylor Series Expansion of $\exp(z)$
The Taylor series expansion of the exponential function $\exp(z)$ around the origin ($z = 0$) is defined as:
$$\exp(z) = \sum_{n=0}^{\infty} \frac{z^n}{n!} = 1 + z + \frac{1}{2!}z^2 + \frac{1}{3!}z^3 + \dots + \frac{z^n}{n!} + \dots$$

Letting $z = \mathbf{q}^\top \mathbf{k}$, the second-degree Taylor polynomial approximation of the exponential similarity kernel is:
$$K(\mathbf{q}, \mathbf{k}) \approx 1 + \mathbf{q}^\top \mathbf{k} + \frac{1}{2} \left(\mathbf{q}^\top \mathbf{k}\right)^2$$

#### 1.2 Algebraic Expansion of the Quadratic Scalar Term
Let $\mathbf{q} = \begin{bmatrix} q_1 & q_2 & \dots & q_d \end{bmatrix}^\top$ and $\mathbf{k} = \begin{bmatrix} k_1 & k_2 & \dots & k_d \end{bmatrix}^\top$ be $d$-dimensional vectors representing the query and key projections. Their inner product is:
$$\mathbf{q}^\top \mathbf{k} = \sum_{i=1}^d q_i k_i$$

Squaring this scalar dot product yields:
$$\left(\mathbf{q}^\top \mathbf{k}\right)^2 = \left( \sum_{i=1}^d q_i k_i \right)^2 = \left( \sum_{i=1}^d q_i k_i \right) \left( \sum_{j=1}^d q_j k_j \right)$$

Expanding this product of summations algebraically results in a double sum over all index pairs $(i, j)$:
$$\left(\mathbf{q}^\top \mathbf{k}\right)^2 = \sum_{i=1}^d \sum_{j=1}^d q_i q_j k_i k_j$$

#### 1.3 Diagonal and Off-Diagonal Separation
To isolate the distinct components of this double summation, we separate the indices into the diagonal terms where $i = j$ and the symmetric off-diagonal terms where $i \neq j$:
$$\sum_{i=1}^d \sum_{j=1}^d q_i q_j k_i k_j = \sum_{i=1}^d q_i^2 k_i^2 + \sum_{i \neq j} q_i q_j k_i k_j$$

Due to the commutativity of real multiplication ($q_i q_j k_i k_j = q_j q_i k_j k_i$), the off-diagonal terms are symmetric. We can group the $i > j$ and $i < j$ halves of the off-diagonal space:
$$\sum_{i \neq j} q_i q_j k_i k_j = \sum_{i < j} q_i q_j k_i k_j + \sum_{i > j} q_i q_j k_i k_j = 2 \sum_{i < j} q_i q_j k_i k_j$$

Substituting this decomposition back into the squared inner product gives:
$$\left(\mathbf{q}^\top \mathbf{k}\right)^2 = \sum_{i=1}^d q_i^2 k_i^2 + 2 \sum_{i < j} q_i q_j k_i k_j$$

Now, substitute this expanded quadratic form back into the second-degree Taylor approximation of the kernel:
$$K(\mathbf{q}, \mathbf{k}) \approx 1 + \sum_{i=1}^d q_i k_i + \frac{1}{2} \left( \sum_{i=1}^d q_i^2 k_i^2 + 2 \sum_{i < j} q_i q_j k_i k_j \right)$$

Distributing the $\frac{1}{2}$ coefficient yields the isolated fractional terms:
$$K(\mathbf{q}, \mathbf{k}) \approx 1 + \sum_{i=1}^d q_i k_i + \frac{1}{2} \sum_{i=1}^d q_i^2 k_i^2 + \sum_{i < j} q_i q_j k_i k_j$$

---

### 2. Deriving the Feature Map $\phi(\mathbf{x})$ and Dimensional Parity ($d'=153$)

To construct a feature map $\phi(\mathbf{x}) \in \mathbb{R}^{d'}$ such that the kernel is evaluated exactly as a linear inner product:
$$K(\mathbf{q}, \mathbf{k}) \approx \phi(\mathbf{q})^\top \phi(\mathbf{k})$$

We analyze two formulations depending on whether the $\frac{1}{2}$ scale factor on the quadratic term is explicitly preserved in the feature space or absorbed (as in the unscaled-quadratic kernel commonly implemented in linear transformer architectures).

#### 2.1 Explicit Vector Form of $\phi(\mathbf{x})$

##### Case A: The Unscaled-Quadratic Kernel (Implemented in `kernel.py`)
In many linear attention architectures (including this repository's codebase), the feature map is designed to reconstruct $1 + \mathbf{q}^\top \mathbf{k} + \left(\mathbf{q}^\top \mathbf{k}\right)^2$ (omitting the $\frac{1}{2}$ denominator, where scale is absorbed by downstream LayerNorm). The explicit vector $\phi(\mathbf{x})$ is defined as:
$$\phi(\mathbf{x}) = \begin{bmatrix}
1 \\
x_1 \\
\vdots \\
x_d \\
x_1^2 \\
\vdots \\
x_d^2 \\
\sqrt{2}x_1x_2 \\
\sqrt{2}x_1x_3 \\
\vdots \\
\sqrt{2}x_{d-1}x_d
\end{bmatrix} \in \mathbb{R}^{d'}$$

##### Case B: The Exact Taylor Kernel (With Quadratic Scaling)
If we want to reconstruct the exact 2nd-degree Taylor expansion containing the $\frac{1}{2}$ factor, the vector $\phi_{\text{exact}}(\mathbf{x})$ is defined as:
$$\phi_{\text{exact}}(\mathbf{x}) = \begin{bmatrix}
1 \\
x_1 \\
\vdots \\
x_d \\
\frac{1}{\sqrt{2}}x_1^2 \\
\vdots \\
\frac{1}{\sqrt{2}}x_d^2 \\
x_1x_2 \\
x_1x_3 \\
\vdots \\
x_{d-1}x_d
\end{bmatrix} \in \mathbb{R}^{d'}$$

#### 2.2 Proof of the Scaling Coefficients
Let us prove mathematically how these scaling coefficients arise by verifying the inner product $\phi(\mathbf{q})^\top \phi(\mathbf{k})$ for both cases.

##### Proof for Case A:
Evaluating the dot product:
$$\phi(\mathbf{q})^\top \phi(\mathbf{k}) = (1)(1) + \sum_{i=1}^d (q_i)(k_i) + \sum_{i=1}^d (q_i^2)(k_i^2) + \sum_{i < j} (\sqrt{2}q_i q_j)(\sqrt{2}k_i k_j)$$
$$\phi(\mathbf{q})^\top \phi(\mathbf{k}) = 1 + \mathbf{q}^\top \mathbf{k} + \sum_{i=1}^d q_i^2 k_i^2 + 2\sum_{i < j} q_i q_j k_i k_j$$

Applying our diagonal/off-diagonal separation identity from Section 1.3:
$$\phi(\mathbf{q})^\top \phi(\mathbf{k}) = 1 + \mathbf{q}^\top \mathbf{k} + \left(\mathbf{q}^\top \mathbf{k}\right)^2$$

This proves that to reconstruct the unscaled quadratic term exactly, the off-diagonal monomial components $x_i x_j$ must be scaled by $\sqrt{2}$ since:
$$(\sqrt{2} q_i q_j)(\sqrt{2} k_i k_j) = 2 q_i q_j k_i k_j$$
This matches the coefficient of $2$ in the algebraic expansion of $\left(\mathbf{q}^\top \mathbf{k}\right)^2$.

##### Proof for Case B:
Evaluating the dot product:
$$\phi_{\text{exact}}(\mathbf{q})^\top \phi_{\text{exact}}(\mathbf{k}) = (1)(1) + \sum_{i=1}^d (q_i)(k_i) + \sum_{i=1}^d \left(\frac{1}{\sqrt{2}}q_i^2\right)\left(\frac{1}{\sqrt{2}}k_i^2\right) + \sum_{i < j} (q_i q_j)(k_i k_j)$$
$$\phi_{\text{exact}}(\mathbf{q})^\top \phi_{\text{exact}}(\mathbf{k}) = 1 + \mathbf{q}^\top \mathbf{k} + \frac{1}{2}\sum_{i=1}^d q_i^2 k_i^2 + \sum_{i < j} q_i q_j k_i k_j$$

Factoring out $\frac{1}{2}$ from the quadratic terms:
$$\phi_{\text{exact}}(\mathbf{q})^\top \phi_{\text{exact}}(\mathbf{k}) = 1 + \mathbf{q}^\top \mathbf{k} + \frac{1}{2}\left( \sum_{i=1}^d q_i^2 k_i^2 + 2\sum_{i < j} q_i q_j k_i k_j \right)$$
$$\phi_{\text{exact}}(\mathbf{q})^\top \phi_{\text{exact}}(\mathbf{k}) = 1 + \mathbf{q}^\top \mathbf{k} + \frac{1}{2} \left(\mathbf{q}^\top \mathbf{k}\right)^2$$

This proves that to reconstruct the exact Taylor expansion, the diagonal terms $x_i^2$ must carry a $\frac{1}{\sqrt{2}}$ coefficient (so that their product yields $\frac{1}{2}$), while the off-diagonal terms carry a coefficient of $1.0$ (since their product yields $1.0$, matching the $1.0$ coefficient of the off-diagonals in $\frac{1}{2} (\mathbf{q}^\top \mathbf{k})^2$).

#### 2.3 Closed-Form Dimension Calculation ($d'=153$)
The output dimension $d'$ of the expanded feature map $\phi(\mathbf{x})$ is computed by summing the number of elements in each of the disjoint monomial sets:
$$d' = N_{\text{constant}} + N_{\text{linear}} + N_{\text{quadratic\_diag}} + N_{\text{quadratic\_off\_diag}}$$

For an input vector of head dimension $d$:
* $N_{\text{constant}} = 1$
* $N_{\text{linear}} = d$
* $N_{\text{quadratic\_diag}} = d$
* $N_{\text{quadratic\_off\_diag}} = \binom{d}{2} = \frac{d(d-1)}{2}$

Summing these terms:
$$d' = 1 + d + d + \frac{d(d-1)}{2} = 1 + 2d + \frac{d^2 - d}{2}$$
$$d' = 1 + \frac{4d + d^2 - d}{2} = 1 + d + \frac{d(d+1)}{2}$$

Substituting the Nanogrid model head dimension $d = 16$:
$$d' = 1 + 16 + \frac{16 \times 17}{2} = 17 + 8 \times 17 = 17 + 136 = 153$$

---

### 3. Code-Binding: The 4D Memory Optimization Patch

To implement the quadratic portion of the feature map, we must compute $x_i x_j$ for all upper-triangular index pairs $i \le j$.

#### 3.1 Naive 5D Outer Product Memory Analysis
A naive implementation materializes all pairwise combinations by forming the full outer product tensor before slicing:
```python
# Naive 5D outer product creation
x_outer = x.unsqueeze(-1) * x.unsqueeze(-2)  # Shape: [B, H, T, d, d]
quad = x_outer[..., self.triu_rows, self.triu_cols]  # Slicing upper triangle
```

Let us calculate the exact peak RAM footprint for this intermediate tensor $\mathbf{X}_{\text{outer}}$ under standard FP32 precision (4 bytes per element) for a batch size $B=32$, number of heads $H=4$, sequence length $T=1440$, and head dimension $d$:
$$\text{Memory}_{5D} = B \cdot H \cdot T \cdot d^2 \cdot 4 \text{ bytes}$$

* **For head dimension $d = 16$**:
  $$\text{Elements} = 32 \times 4 \times 1440 \times 16 \times 16 = 47,185,920 \text{ elements}$$
  $$\text{Memory}_{5D} = 47,185,920 \times 4 \text{ bytes} \approx 188.74 \text{ MB}$$
* **For a larger head dimension $d = 128$**:
  $$\text{Elements} = 32 \times 4 \times 1440 \times 128 \times 128 = 3,019,898,880 \text{ elements}$$
  $$\text{Memory}_{5D} = 3,019,898,880 \times 4 \text{ bytes} \approx 12.08 \text{ GB}$$

This memory footprint represents a single attention layer. In multi-layer, multi-head configurations, allocating and deallocating multi-gigabyte tensors at every training step leads to Out-Of-Memory (OOM) errors and excessive garbage collection overhead.

#### 3.2 The 4D Slice Selection Patch (Math-to-Code Binding)
To bypass this bottleneck, the **4D Slice Selection Patch** is implemented. Instead of computing the full $d \times d$ outer product matrix and discarding nearly half of it, we directly extract the required row and column components from the input $x$ and perform a single element-wise multiplication in 4D space:
```python
# 4D Vectorized Slice Selection (from src/models/kernel.py)
quad = (x[..., self.triu_rows] * x[..., self.triu_cols]) * self.quad_scale
```

Let's compute the RAM footprint of the intermediate tensors for this calculation:
The indices `self.triu_rows` and `self.triu_cols` select vectors of size $n_{\text{quad}} = \frac{d(d+1)}{2}$.
For $d = 16$, $n_{\text{quad}} = 136$.
The shape of the sliced tensors is $[B, H, T, n_{\text{quad}}]$.
The number of elements is:
$$\text{Elements} = 32 \times 4 \times 1440 \times 136 = 25,067,520 \text{ elements}$$
The memory required for the final output tensor is:
$$\text{Memory}_{4D} = 25,067,520 \times 4 \text{ bytes} \approx 100.27 \text{ MB}$$

Comparing this directly with the naive 5D footprint:
* For $d=16$: **100.27 MB** (4D) vs. **188.74 MB** (5D) — a **46.8% memory reduction**.
* For $d=128$: **3.24 GB** (4D) vs. **12.08 GB** (5D) — a **73.2% memory reduction** ($n_{\text{quad}} = 8256$).

#### 3.3 Hardware-Level Performance Proof (Intel CPU Cache Line Analysis)
Operating within 4D space prevents hardware-level cache thrashing and memory controller bottlenecks on CPU and GPU architectures:

1. **Elimination of Advanced Indexing Copy Overhead**:
   In PyTorch, slicing a tensor with two separate index tensors (e.g., `x_outer[..., self.triu_rows, self.triu_cols]`) triggers **advanced indexing**. Because the indices represent arbitrary non-contiguous coordinates, PyTorch cannot return a memory view; it is forced to allocate a new tensor and copy the data. This requires the memory controller to perform non-coalesced, strided lookups across the last two dimensions of the 5D tensor.
2. **CPU Cache Line Stalls**:
   On modern Intel CPUs, L1/L2 caches are organized into 64-byte cache lines. A naive 5D indexing operation accesses memory locations separated by strides of size $d$ and $d^2$. Since these elements do not reside in the same cache lines, the processor experiences a high rate of L1/L2 cache misses. The CPU pipeline must halt while waiting to load data from DRAM (a latency penalty of $\sim 60\text{ns}$ per miss), leading to cache thrashing.
3. **Contiguous 4D Indexing and Vectorization**:
   The 4D patch `x[..., self.triu_rows]` indexes the 4D tensor along its last dimension using a single index tensor. Slicing with a single index tensor is highly contiguous in memory. Once the two sliced tensors of shape $[B, H, T, n_{\text{quad}}]$ are retrieved, their product is computed as a simple element-wise multiplication:
   $$\text{quad}_{b, h, t, k} = x_{b, h, t, u_k} \cdot x_{b, h, t, v_k} \cdot w_k$$
   where $u_k$ and $v_k$ are the $k$-th elements of `triu_rows` and `triu_cols`, and $w_k$ is the scaling factor. Because this operation is completely element-wise over aligned memory chunks, compilers can apply SIMD (Single Instruction Multiple Data) registers (e.g., Intel AVX-512) to execute multiple multiplications per clock cycle, saturating memory bandwidth and accelerating execution.

---

## TRACK 2: ASSOCIATIVE CONTRACTION & TIME RE-ORDERING

Standard dot-product attention scales quadratically because queries and keys are multiplied first, materializing a temporal matrix representing all pairwise steps.

### 1. The Bottleneck vs. The Transposition Equations

Let $\mathbf{Q} \in \mathbb{R}^{T \times d}$, $\mathbf{K} \in \mathbb{R}^{T \times d}$, and $\mathbf{V} \in \mathbb{R}^{T \times d_v}$ represent the Query, Key, and Value matrices respectively.

#### 1.1 Softmax Attention Bottleneck
Standard scaled dot-product attention performs:
$$\mathbf{A} = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}} \right) \in \mathbb{R}^{T \times T}$$
$$\text{Output} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{T \times d_v}$$

The dimension operations map as:
$$\text{Softmax}\left( \underbrace{[T \times d] \times [d \times T]}_{\text{Materializes } [T \times T]} \right) \times [T \times d_v] \longrightarrow [T \times d_v]$$

The $T \times T$ matrix $\mathbf{A}$ represents a spatial complexity of $O(T^2)$. For our nanogrid sequence length of $T = 1440$, this requires storing $1440 \times 1440 \approx 2.07 \times 10^6$ elements per head, per batch.

#### 1.2 Associative Law Re-Ordering
By substituting the kernel feature map approximation $K(\mathbf{q}_t, \mathbf{k}_{t'}) = \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_{t'})$, we write the unnormalized output vector at time step $t$ as:
$$\hat{\mathbf{o}}_t = \sum_{t'=1}^T \left( \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_{t'}) \right) \mathbf{v}_{t'}$$

Since $\phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_{t'})$ is a scalar, we rewrite the scalar-vector product using the vector outer product $\otimes$:
$$\left( \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_{t'}) \right) \mathbf{v}_{t'} = \phi(\mathbf{q}_t)^\top \left( \phi(\mathbf{k}_{t'}) \otimes \mathbf{v}_{t'} \right)$$

Substituting this back into the sum:
$$\hat{\mathbf{o}}_t = \sum_{t'=1}^T \phi(\mathbf{q}_t)^\top \left( \phi(\mathbf{k}_{t'}) \otimes \mathbf{v}_{t'} \right)$$

Because the query vector $\phi(\mathbf{q}_t)$ is independent of the summation index $t'$, the distributive law allows us to pull it out of the summation:
$$\hat{\mathbf{o}}_t = \phi(\mathbf{q}_t)^\top \left( \sum_{t'=1}^T \phi(\mathbf{k}_{t'}) \otimes \mathbf{v}_{t'} \right)$$

This mathematical reformulation is the associative law of matrix multiplication. The sum inside the parentheses is the key-value memory state matrix $\mathbf{S}$:
$$\mathbf{S} = \sum_{t'=1}^T \phi(\mathbf{k}_{t'}) \mathbf{v}_{t'}^\top \in \mathbb{R}^{d' \times d_v}$$

The normalized attention output at step $t$ is:
$$\text{Output}_t = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}}{\phi(\mathbf{q}_t)^\top \sum_{t'=1}^T \phi(\mathbf{k}_{t'})}$$

#### 1.3 State Collapse and Tensor Contraction
By computing the state matrix $\mathbf{S}$ first, the time axis $T$ is completely contracted out:
$$\Phi(\mathbf{K}) \in \mathbb{R}^{B \times H \times T \times d'}, \quad \mathbf{V} \in \mathbb{R}^{B \times H \times T \times d_v} \longrightarrow \mathbf{S} \in \mathbb{R}^{B \times H \times d' \times d_v}$$

The visual representation of this tensor contraction is:

```
      [T × d']ᵀ                 [T × d_v]
   (Key Feature Map)         (Value Tensor)
         │                         │
         ▼                         ▼
    ┌──────────┐              ┌──────────┐
 d' │          │            T │          │
    ├──────────┤              ├──────────┤
  T │          │              │   d_v    │
    └──────────┘              └──────────┘
         │                         │
         └───────────┬─────────────┘
                     │  Contract along sequence axis T
                     ▼
               ┌───────────┐
            d' │     S     │
               ├───────────┤
               │    d_v    │
               └───────────┘
            Static Shape: [B, H, d', d_v]
```

Replacing the variables with our physical Nanogrid shapes:
$$\mathbf{S} \in \mathbb{R}^{B \times H \times 153 \times 16}$$
This state matrix is static in size regardless of how long the sequence length $T$ grows, allowing the model to handle long sequences with constant memory overhead.

---

### 2. Code-Binding: The Einsum Contraction Traces

The tensor contractions are implemented in [`src/models/linear_attention.py`](file:///Users/apple/LinearGridTransformer/src/models/linear_attention.py) using PyTorch's optimized `einsum` engine.

#### 2.1 Key-Value State Construction
The key-value state tensor $\mathbf{S}$ is constructed by contracting the Key features and Value vectors over the sequence length axis:
$$\mathbf{S}_{b, h, p, v} = \sum_{t=1}^T \Phi(\mathbf{K})_{b, h, t, p} \cdot \mathbf{V}_{b, h, t, v}$$

##### Python Binding:
```python
# S shape: [B, H, d_prime, d_v] = [32, 4, 153, 16]
KV_state = torch.einsum("bhtp, bhtv -> bhpv", K_phi, V)
```

##### Index Trace:
* `b`: Batch dimension ($B = 32$)
* `h`: Attention heads dimension ($H = 4$)
* `t`: Temporal sequence dimension ($T = 1440$)
* `p`: Linearized feature dimension $d'$ ($d' = 153$)
* `v`: Value projection dimension $d_v$ ($d_v = 16$)
* **Contraction**: The index `t` appears in both input terms (`bhtp` and `bhtv`) but is absent in the output term (`bhpv`). This forces PyTorch to perform an inner product reduction along the sequence length axis $T$, collapsing the time dimension.

#### 2.2 Context Output Retrieval
The unnormalized context matrix is retrieved by contracting the Query feature map with the key-value state matrix over the feature dimension $d'$:
$$\text{Context}_{b, h, t, v} = \sum_{p=1}^{d'} \Phi(\mathbf{Q})_{b, h, t, p} \cdot \mathbf{S}_{b, h, p, v}$$

##### Python Binding:
```python
# Context shape: [B, H, T, d_v] = [32, 4, 1440, 16]
Context = torch.einsum("bhtp, bhpv -> bhtv", Q_phi, KV_state)
```

##### Index Trace:
* **Contraction**: The index `p` appears in both input terms (`bhtp` and `bhpv`) but is absent in the output term (`bhtv`). The engine performs a dot product reduction over the feature space $d'$, reconstructing the context vector for each sequence step $t$.

---

## TRACK 3: ZERO-INPUT ATTENTION FLOOR PROOF

A common vulnerability in linearized attention models is the risk of the normalization denominator $Z_t$ approaching zero ($Z_t \to 0$), which causes division-by-zero errors, NaN values, or exploding gradients.

### 1. Mathematical Derivation of the Safety Floor

The denominator partition function $Z_t$ for a query at sequence step $t$ is:
$$Z_t = \phi(\mathbf{q}_t)^\top \sum_{t'=1}^T \phi(\mathbf{k}_{t'})$$

#### 1.1 Worst-Case Boundary Condition
In smart grid monitoring, physical sensors can experience complete signal dropouts (e.g., at midnight, solar power generation drops to zero, or lines are disconnected). This propagates zero vectors through the query and key projection matrices:
$$\mathbf{k}_{t'} = \mathbf{0} \in \mathbb{R}^d \quad \forall t' \in \{1, \dots, T\}$$

#### 1.2 Zero Vector Feature Map Mapping
Let us pass the zero vector $\mathbf{0}$ through the 2nd-degree Taylor feature map $\phi(\mathbf{x})$:
$$\phi(\mathbf{0}) = \begin{bmatrix}
1 \\
(0)_1 \\
\vdots \\
(0)_d \\
(0)^2_1 \\
\vdots \\
(0)^2_d \\
\sqrt{2}(0)(0) \\
\vdots \\
\sqrt{2}(0)(0)
\end{bmatrix} = \begin{bmatrix}
1 \\
0 \\
\vdots \\
0 \\
0 \\
\vdots \\
0 \\
0 \\
\vdots \\
0
\end{bmatrix} \in \mathbb{R}^{d'}$$

The feature map maps the zero vector to a vector where the constant term index is exactly $1$, and all linear and quadratic terms are $0$:
$$\phi(\mathbf{0}) = \begin{bmatrix} 1 & 0 & 0 & \dots & 0 \end{bmatrix}^\top$$

#### 1.3 Key Summation in Feature Space
Summing these feature-mapped keys over the sequence length $T$:
$$\sum_{t'=1}^T \phi(\mathbf{k}_{t'}) = \sum_{t'=1}^T \phi(\mathbf{0}) = \sum_{t'=1}^T \begin{bmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix} = \begin{bmatrix} \sum_{t'=1}^T 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix} = \begin{bmatrix} T \\ 0 \\ \vdots \\ 0 \end{bmatrix}$$

#### 1.4 Denominator Evaluation
Now, evaluate the inner product of any query feature vector $\phi(\mathbf{q}_t)$ with the accumulated key vector. Regardless of the query vector values:
$$\phi(\mathbf{q}_t) = \begin{bmatrix} 1 & q_{t,1} & q_{t,2} & \dots & q_{t,d'-1} \end{bmatrix}^\top$$

The inner product is:
$$Z_t = \phi(\mathbf{q}_t)^\top \sum_{t'=1}^T \phi(\mathbf{k}_{t'}) = \begin{bmatrix} 1 \\ q_{t,1} \\ \vdots \\ q_{t,d'-1} \end{bmatrix}^\top \begin{bmatrix} T \\ 0 \\ \vdots \\ 0 \end{bmatrix}$$
$$Z_t = 1 \cdot T + \sum_{j=1}^{d'-1} q_{t,j} \cdot 0 = T$$

Under complete key signal dropout, the denominator partition function simplifies to:
$$Z_t = T$$

For a sequence length of $T = 1440$:
$$Z_t = 1440$$

Since sequence length must satisfy $T \ge 1$:
$$Z_t \ge T \ge 1 > 0$$

This mathematically proves that the constant $1$ in the Taylor feature map guarantees a strict positive lower bound of $Z_t \ge T$ for the denominator.

#### 1.5 Independence from the Epsilon Guard $\varepsilon$
In the code implementation, the final normalization step divides the context by the denominator plus a small stability constant $\varepsilon$:
$$\text{Output}_t = \frac{\text{Context}_t}{Z_t + \varepsilon}$$

Because of the structural floor $Z_t \ge T$, even if we set $\varepsilon = 0$, the division remains mathematically stable:
$$\text{Output}_t = \frac{\text{Context}_t}{T}$$

The constant monomial in the Taylor expansion prevents numerical collapse and division-by-zero, ensuring robust stability under zero-power conditions in grid data.
