# ⚡ LinearGridTransformer: $O(T)$ Memory-Efficient Time-Series Forecasting

A PyTorch implementation of a Linear Attention Transformer for multivariate smart-grid time-series forecasting, benchmarked head-to-head against a simple linear baseline (CDLinear) and a naive persistence baseline.

This project replaces standard $O(T^2)$ Softmax attention with an $O(T)$ linear attention mechanism powered by a 2nd-degree Taylor feature map, enabling long-sequence processing without the quadratic memory blowup of vanilla attention. The headline engineering result — $O(T)$ memory scaling — holds up. The headline *accuracy* result does not: on this dataset and training budget, the much simpler linear baseline wins, and the reason why turned out to be the more interesting finding.

---

## 🏛️ Key Architectural Features

- **2nd-Degree Taylor Feature Map (`src/models/kernel.py`)**: Approximates $\exp(\mathbf{q}^T\mathbf{k})$ using polynomial expansion ($\varphi(\mathbf{x}) \in \mathbb{R}^{153}$ for $d=16$) executed via `torch.einsum`.
- **Associative KV Memory Accumulation (`src/models/linear_attention.py`)**: Computes $\mathbf{S} = \sum \varphi(\mathbf{k}_{t'})^T \otimes \mathbf{v}_{t'}$ to maintain $O(1)$ memory growth relative to sequence length $T$.
- **Attention Diagnostics via PyTorch Hooks (`src/training/train.py`)**: Features a custom `DenominatorTracker` forward hook to monitor partition function variance ($\mathbf{Z}_t$) and prevent attention dilution.
- **Lazy Loaders & Data Pipeline (`src/data/dataset.py`)**: Handles $1{,}440 \rightarrow 60$ step lookback/forecast windows on the UCI Household Power Consumption dataset with Z-score scaling.
- **Comparative Training Harness (`src/training/train_comparative.py`)**: Trains LinearGridTransformer and a CDLinear baseline side by side under identical data, optimizer, and epoch budgets so the comparison is apples-to-apples.

---

## 🛠️ Setup

```bash
pip install -r requirements.txt
```

`data/processed/clean_grid.pkl` (the cleaned UCI Household Power Consumption data) and `src/data/global_mean.npy` / `global_std.npy` (normalisation stats) are included in this repo directly, so training/evaluation scripts work out of the box against them. Note: `src/data/dataset.py`'s own error messages point to `src/utils/clean_chunks.py` and `src/utils/compute_stats.py` as the scripts that generate those two artifacts from the raw dataset — neither script is actually present in this repo, so the pickle/`.npy` files here can't currently be regenerated from scratch. Treat them as fixed inputs rather than reproducible outputs until that gap is closed.

---

## 📊 Empirical Performance & Benchmarks

Full run on GPU, 10 epochs, 10,458 batches/epoch, identical data/optimizer/schedule for both models. Metrics are on denormalised (physical-unit) validation data.

| Model | val MAE | val RMSE | Train time / epoch |
| :--- | :---: | :---: | :---: |
| **LinearGridTransformer** | 2.0027 | 3.6370 | ~1,330 s (Tesla T4) |
| **CDLinear (baseline)** | **1.8282** | **3.2264** | **~43 s (Tesla T4)** |
| Naive (persistence) | 2.4923 | — | — |

Both models comfortably beat the naive persistence baseline. But **CDLinear wins outright** — about 9.5% lower MAE than LinearGridTransformer — while training roughly **30x faster per epoch**, since it has no attention mechanism at all.

### Why the transformer loses: it plateaus almost immediately, then overfits

The more useful result isn't the scoreboard, it's the full training trajectory. LinearGridTransformer's train loss falls steadily and monotonically for all 10 epochs, but its validation MAE never gets meaningfully better than where it already was after epoch 1 — it just oscillates:

| epoch | LGT train_loss | LGT val_MAE | LGT val_RMSE | CDLinear train_loss | CDLinear val_MAE | CDLinear val_RMSE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.800 | 2.0200 | 3.5699 | 0.728 | 1.9182 | 3.3048 |
| 2 | 0.774 | 1.9825 | 3.5663 (best) | 0.660 | 1.8797 | 3.2696 |
| 3 | 0.766 | 1.9839 | 3.5778 | 0.650 | 1.8817 | 3.2852 |
| 4 | 0.759 | 1.9881 | 3.5939 | 0.646 | 1.8196 (best) | 3.2478 |
| 5 | 0.753 | 2.0269 | 3.6191 | 0.644 | 1.8685 | 3.2809 |
| 6 | 0.746 | 1.9944 | 3.6150 | 0.643 | 1.8568 | 3.2399 (best) |
| 7 | 0.739 | 1.9920 | 3.6005 | 0.642 | 1.8504 | 3.2471 |
| 8 | 0.731 | 1.9714 (best) | 3.6035 | 0.641 | 1.8642 | 3.2950 |
| 9 | 0.724 | 2.0367 | 3.6308 | 0.641 | 1.8524 | 3.2433 |
| 10 | 0.717 | 2.0027 | 3.6370 | 0.641 | 1.8282 | 3.2264 |

(Full numbers in `docs/gpu_comparative_report.json`; per-epoch wall-clock timing cross-checked against the raw training console log in `docs/gpu_training_console_log.txt` — LGT logged `train=1328.6s`–`1332.4s` per epoch, CDLinear logged `train=42.3s`–`43.3s` per epoch, both on a Kaggle Tesla T4. Both models beat the naive baseline at every single epoch, not just at the end.)

LGT's train loss drops from 0.800 to 0.717 across the run — it's still finding ways to fit the training set the whole time — but val MAE just oscillates in a 1.97–2.04 band from epoch 2 onward, with no clear downward trend, and val RMSE is actually at its *best* at epoch 2 (3.5663) and drifts slightly worse from there (3.637 by epoch 10). That's a textbook overfitting signature: past epoch 1–2, additional training buys the transformer nothing on unseen data.

CDLinear shows the opposite pattern: train loss drops fast for the first ~4 epochs then flattens out around 0.64, and val MAE settles into a tight, stable 1.82–1.92 band for the whole run — no epoch is dramatically worse than any other, unlike LGT's epoch-9 spike back up to 2.0367.

The likely cause isn't a bug, and it isn't "needed more training" — it's architectural: a single-layer linear-attention encoder simply has a low capacity ceiling for this task, and it hits that ceiling within the first epoch or two of exposure to 10,458 batches/epoch of grid data. Everything after that is the optimizer polishing training-set fit with no generalization left to gain.

### Takeaway

For this task, at this training budget, the $O(T)$ linear-attention encoder is not worth its cost: it's slower, more complex, and less accurate than a linear baseline that fits directly. The linear attention *mechanism itself* still does what it claims — constant memory growth in $T$ (see benchmarks below) — but that efficiency win doesn't translate into a forecasting-quality win here, because the bottleneck for this dataset turned out to be model capacity, not attention's quadratic cost.

---

## 🧠 Memory Scaling: $O(T)$ vs $O(T^2)$ Attention

Independent of the accuracy comparison above, `src/benchmarks/profile_memory.py` isolates each attention mechanism's peak memory in a fresh subprocess per data point (see `src/benchmarks/_memory_worker.py`), avoiding the allocator-caching artifacts that come from measuring multiple sequence lengths in one long-lived process. Results in `docs/memory_vs_sequence.png` / `data/benchmarks/memory_scale.csv` show vanilla softmax attention's memory growing quadratically with sequence length $T$, while `LinearKernelAttention` grows linearly — the architectural claim the model is built on holds regardless of the downstream accuracy result.

---

## 🔭 Future Work

The overfitting diagnosis points to specific, cheap things to try before concluding linear attention can't work here at all:
- Increase dropout / add weight decay specifically on the attention/FFN blocks to slow the epoch-2 plateau.
- Add a second encoder layer or widen `d_model` — the current single-layer encoder may simply be under-parameterized relative to the 1,440-step lookback window.
- Try a shorter lookback or a curriculum that ramps sequence length, to see if the capacity ceiling is a function of $T$ specifically.

None of these have been run yet — the honest baseline result above is what's actually verified.

---

## 📂 Repository Structure

```text
LinearGridTransformer/
├── src/
│   ├── models/
│   │   ├── kernel.py               # TaylorFeatureMap implementation
│   │   ├── linear_attention.py     # LinearKernelAttention module
│   │   ├── cdlinear.py             # CDLinear baseline
│   │   └── dlinear.py              # DLinear reference implementation
│   ├── data/
│   │   └── dataset.py              # GridSequenceDataset lazy loader
│   ├── training/
│   │   ├── train.py                # LinearGridTransformer training loop & DenominatorTracker hook
│   │   ├── train_comparative.py    # Side-by-side LGT vs. CDLinear training harness
│   │   └── evaluate.py             # Denormalised-metric evaluation
│   ├── benchmarks/
│   │   ├── profile_memory.py       # Memory & FLOP profiling scripts
│   │   └── _memory_worker.py       # Isolated-process peak-RSS measurement
│   └── utils/
│       └── test_gradient.py        # Autograd correctness check: 5D vs. 4D feature-map paths
├── docs/
│   ├── gpu_comparative_report.json          # Full 10-epoch GPU comparative results (source of the tables above)
│   ├── gpu_training_console_log.txt         # Raw Kaggle console log (verifies per-epoch timing, Tesla T4)
│   ├── generalized_comparative_report.json  # Earlier CPU comparative run (superseded by the GPU report)
│   ├── memory_vs_sequence.png               # O(T) vs O(T^2) memory plot
│   └── nanogrid_math_compendium.md          # Math derivations
├── checkpoints/
│   ├── comparative_lgt.pt                   # LinearGridTransformer, final epoch (10)
│   └── comparative_cdlinear.pt              # CDLinear baseline, final epoch (10)
├── data/
│   ├── processed/clean_grid.pkl             # Cleaned UCI dataset (see Setup — regeneration script not included)
│   └── benchmarks/memory_scale.csv          # Raw data behind the memory plot
├── test_attention.py                # Encoder block validation script
├── test_pipeline.py                 # Integration smoke-test for dataset and dataloader
├── requirements.txt
├── LICENSE
└── README.md
```
