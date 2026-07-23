# ⚡ LinearGridTransformer: $O(T)$ Memory-Efficient Time-Series Forecasting

A production-grade PyTorch implementation of a Linear Attention Transformer for multivariate smart-grid time-series forecasting.

This model replaces standard $O(T^2)$ Softmax attention with an $O(T)$ linear attention mechanism powered by a 2nd-degree Taylor feature map, enabling long-sequence processing without memory bottlenecks.

---

## 🏛️ Key Architectural Features

- **2nd-Degree Taylor Feature Map (`src/models/kernel.py`)**: Approximates $\exp(\mathbf{q}^T\mathbf{k})$ using polynomial expansion ($\varphi(\mathbf{x}) \in \mathbb{R}^{153}$ for $d=16$) executed via `torch.einsum`.
- **Associative KV Memory Accumulation (`src/models/linear_attention.py`)**: Computes $\mathbf{S} = \sum \varphi(\mathbf{k}_{t'})^T \otimes \mathbf{v}_{t'}$ to maintain $O(1)$ memory growth relative to sequence length $T$.
- **Attention Diagnostics via PyTorch Hooks (`src/training/train.py`)**: Features a custom `DenominatorTracker` forward hook to monitor partition function variance ($\mathbf{Z}_t$) and prevent attention dilution.
- **Lazy Loaders & Data Pipeline (`src/data/dataset.py`)**: Handles $1,440 \rightarrow 60$ step lookback/forecast windows on the UCI Household Power Consumption dataset with Z-score scaling.

---

## 📊 Empirical Performance & Benchmarks

| Metric | Target Value |
| :--- | :--- |
| **Lookback Window ($T$)** | 1,440 steps (~24 hrs at 1-min resolution) |
| **Forecast Horizon** | 60 steps (~1 hr ahead) |
| **Global Test MAE** | **2.76** (Physical Units) |
| **Global Test RMSE** | **4.45** (Physical Units) |
| **Attention Memory Complexity** | $O(T)$ Linear Scaling |

---

## 📂 Repository Structure

```text
LinearGridTransformer/
├── src/
│   ├── models/
│   │   ├── kernel.py            # TaylorFeatureMap implementation
│   │   └── linear_attention.py  # LinearKernelAttention module
│   ├── data/
│   │   └── dataset.py           # GridSequenceDataset lazy loader
│   ├── training/
│   │   └── train.py             # Training loop & DenominatorTracker hook
│   └── benchmarks/
│       └── profile_memory.py    # Memory & FLOP profiling scripts
├── test_attention.py            # Encoder block unit tests
├── test_pipeline.py             # End-to-end integration tests
└── README.md
