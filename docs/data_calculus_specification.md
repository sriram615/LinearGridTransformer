# Data Calculus & Matrix Geometry Ledger

This document details the physical properties, grid coupling physics, statistical normalization matrices, macro-temporal striding geometry, and boundary crossover protection mapping for the multi-channel power consumption data manifold.

---

## 1. THE REVENUE & PHYSICAL CORE: CHANNEL METADATA AUDIT

The input data manifold consists of $C=7$ continuous physical channels. Each channel measures a distinct electrical property of the household micro-grid.

### 1.1 Physical Property and Unit Breakdown

| Channel ID | Column Name | Physical Property | Measurement Unit |
| :--- | :--- | :--- | :--- |
| **$c=1$** | `Global_active_power` | Real Active Power ($P$) | Kilowatts ($\text{kW}$) |
| **$c=2$** | `Global_reactive_power` | Reactive Power ($Q$) | Kilowatts ($\text{kW}$) or Kilovolt-Amperes Reactive ($\text{kVAR}$) |
| **$c=3$** | `Voltage` | Root-Mean-Square Voltage ($V$) | Volts ($\text{V}$) |
| **$c=4$** | `Global_intensity` | Current Intensity ($I$) | Amperes ($\text{A}$) |
| **$c=5$** | `Sub_metering_1` | Active Energy (Kitchen) | Watt-hours ($\text{Wh}$) |
| **$c=6$** | `Sub_metering_2` | Active Energy (Laundry Room) | Watt-hours ($\text{Wh}$) |
| **$c=7$** | `Sub_metering_3` | Active Energy (HVAC & Water Heater) | Watt-hours ($\text{Wh}$) |

---

### 1.2 Grid Coupling Physics & Mathematical Coupling

The $C=7$ channels are coupled through physical constraints of electrical grid networks:

#### 1. Apparent Power and Complex Current Conservation
The Apparent Power magnitude $|S|$ (in $\text{kVA}$) is the vector sum of Active Power $P$ and Reactive Power $Q$:
$$|S| = \sqrt{P^2 + Q^2}$$

In a single-phase AC circuit, $|S|$ is the product of the RMS Voltage $V$ and the RMS Current $I$:
$$|S| = V \cdot I \implies I = \frac{\sqrt{P^2 + Q^2}}{V}$$

This couples $P$ (`Global_active_power`), $Q$ (`Global_reactive_power`), $V$ (`Voltage`), and $I$ (`Global_intensity`).

#### 2. Ohm's Law and Voltage Drops
The voltage drop $\Delta V$ across the feeder line impedance $Z = R + jX$ (where $R$ is resistance and $X$ is reactance) is approximated by:
$$\Delta V = V_{\text{source}} - V \approx \frac{R P + X Q}{V}$$

This indicates that increases in active and reactive power demand ($P$ and $Q$) lead to a non-linear drop in node voltage $V$.

#### 3. Energy Conservation Constraint (Sub-metering Decomposition)
The total active energy consumed is the sum of sub-metered active energies plus any unmetered load:
$$P_{\text{active\_Wh}} = P_{\text{sub1}} + P_{\text{sub2}} + P_{\text{sub3}} + P_{\text{unmetered}}$$
where $P_{\text{active\_Wh}} = \frac{1000}{60} \cdot P_{\text{active\_kW}}$ (integrating real power over 1-minute intervals).

---

### 1.3 Inductive Bias Mapping: CDLinear vs. LinearGridTransformer

The physical constraints detailed above give the $420 \times 10,080$ parameter matrix of `CDLinearBaseline` a significant inductive bias advantage over the `LinearGridTransformer` under full-throughput convergence:

1. **CDLinear Spatial-Temporal Correlation**:
   The single linear projection layer $\mathbf{W} \in \mathbb{R}^{420 \times 10,080}$ maps the flattened history directly to the forecast:
   $$\mathbf{y}_{\text{flat}, b} = \mathbf{W} \mathbf{x}_{\text{flat}, b} + \mathbf{b}$$
   This allows the model to learn and enforce linear approximations of physical constraints directly (e.g., current intensity scaling proportionally with active power, or energy conservation constraints) across all channels and time steps simultaneously.
2. **LinearGridTransformer Sequence Routing**:
   The attention mechanism routes information temporally through query-key matching. While this allows for dynamic temporal modeling, the transformer processes channels at each time step through projection layers, requiring it to reconstruct spatial physical correlations indirectly. This makes learning static physical constraints less direct compared to the flat projection layer of `CDLinear`.

---
---

## 2. VOLTAGE-STABILIZATION: STATISTICAL PRE-PROCESSING

To prevent features with different numerical ranges from biasing the optimization path, Z-score standardization is applied to each channel independently at slice-time.

### 2.1 The Standardization Equation

Let $x_{t, c}$ represent the raw value of channel $c$ at sequence step $t$. The standardized value $\tilde{x}_{t, c}$ is computed as:
$$\tilde{x}_{t, c} = \frac{x_{t, c} - \mu_c}{\sigma_c}$$
where $\mu_c$ is the training mean and $\sigma_c$ is the training standard deviation of channel $c$.

---

### 2.2 Order-of-Magnitude Alignment

Without standardization, high-magnitude features can dominate gradient updates:
* **Raw Voltage**: $\mu_{\text{voltage}} \approx 240\text{ V}$, with small fluctuations ($\sigma_{\text{voltage}} \approx 3.2\text{ V}$).
* **Raw Sub-metering**: $\mu_{\text{sub1}} \approx 1.1\text{ Wh}$, with high variability ($\sigma_{\text{sub1}} \approx 6.1\text{ Wh}$, ranging from $0$ to $80\text{ Wh}$).

The gradient of the loss $\mathcal{L}$ with respect to the linear weight $W_{j, c}$ is:
$$\frac{\partial \mathcal{L}}{\partial W_{j, c}} = \frac{\partial \mathcal{L}}{\partial \hat{y}_j} \cdot x_c$$

If raw values are used, the gradient for the Voltage channel would be more than two orders of magnitude larger than that of the Sub-metering channel:
$$\frac{\partial \mathcal{L}}{\partial W_{j, \text{voltage}}} \gg \frac{\partial \mathcal{L}}{\partial W_{j, \text{sub1}}}$$

This would cause the optimizer to prioritize minimizing voltage reconstruction errors, leading to gradient instability and poor convergence on the sub-metering channels. Z-score standardization scales all features to a common range, ensuring stable gradient updates across all channels:
$$\tilde{x}_{t, c} \sim \mathcal{N}(0, 1) \implies \mathbb{E}\left[ \frac{\partial \mathcal{L}}{\partial W_{j, c}} \right] \sim \text{aligned}$$

---

### 2.3 The Transformation Lifecycle

The tensor shapes and data types transition through the following pipeline:

```
┌─────────────────────────────────┐
│           Raw CSV Row           │ ──> dtype: object / float64 (loaded via pandas)
│           Shape: [7]            │
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│      Raw NumPy Array View       │ ──> shape: (T_window, 7), dtype: float32
│       Shape: [1500, 7]          │
└─────────────────────────────────┘
                 │
                 ▼  Apply Z-score: (x - mean) / std
┌─────────────────────────────────┐
│     Normalized NumPy Array      │ ──> shape: (T_window, 7), dtype: float32
│       Shape: [1500, 7]          │
└─────────────────────────────────┘
                 │
                 ▼  torch.from_numpy(x_norm.copy())
┌─────────────────────────────────┐
│    PyTorch FloatTensor Slices   │ ──> X: [1440, 7] float32, Y: [60, 7] float32
│     X: [1440, 7], Y: [60, 7]    │
└─────────────────────────────────┘
```

---
---

## 3. THE SEASONAL STRIDING CALCULUS (`src/data/dataset.py`)

When `sub_sample=True`, the dataset groups rows by year and month to preserve seasonal representation while keeping the memory footprint low.

### 3.1 Training Extraction Block

* **Months**: 39 training months.
* **Extraction size per month**: 1 week ($10,080$ consecutive minutes).
* **Derivation**:
  $$\text{Train Rows} = 39 \text{ months} \times 10,080 \text{ minutes/month} = 393,120 \text{ rows}$$

This array is stored in memory as a contiguous float32 block:
$$\text{Memory} = 393,120 \text{ rows} \times 7 \text{ channels} \times 4 \text{ bytes/element} \approx 11.00 \text{ MB}$$

---

### 3.2 Validation Extraction Block

* **Months**: 10 validation months.
* **Extraction size per month**: 2 days ($2,880$ consecutive minutes).
* **Derivation**:
  $$\text{Validation Rows} = 10 \text{ months} \times 2,880 \text{ minutes/month} = 28,800 \text{ rows}$$

Stored in memory as:
$$\text{Memory} = 28,800 \text{ rows} \times 7 \text{ channels} \times 4 \text{ bytes/element} \approx 806.4 \text{ KB}$$

---

### 3.3 Sliding Window Step Calculation

The number of valid sliding window starting positions within a block of length $L_i$ is determined by:
$$N_{\text{block\_windows}} = L_i - (L_{\text{lookback}} + L_{\text{forecast}}) + 1$$

For $L_{\text{lookback}} = 1440$ and $L_{\text{forecast}} = 60$, the window size is:
$$W = 1440 + 60 = 1500 \text{ steps}$$
$$N_{\text{block\_windows}} = L_i - 1500 + 1 = L_i - 1499$$

#### Verification of Train Set Windows:
For each of the $39$ training months, $L_i = 10,080$:
$$N_{\text{train\_block\_windows}} = 10,080 - 1500 + 1 = 8,581 \text{ windows}$$
$$\text{Total Train Windows} = 39 \text{ blocks} \times 8,581 \text{ windows/block} = 334,659 \text{ window pairs}$$

#### Verification of Validation Set Windows:
For each of the $10$ validation months, $L_i = 2,880$:
$$N_{\text{val\_block\_windows}} = 2,880 - 1500 + 1 = 1,381 \text{ windows}$$
$$\text{Total Val Windows} = 10 \text{ blocks} \times 1,381 \text{ windows/block} = 13,810 \text{ window pairs}$$

---
---

## 4. BOUNDARY LEAKAGE DEFENSE & INDEX MAPPING

To prevent sliding windows from crossing boundaries between discontinuous monthly blocks (e.g., from the end of January to the beginning of February), we restrict the starting indices.

### 4.1 Boundary Leakage Math

Let the concatenated array `self._data` contain $M$ blocks. The $k$-th block starts at flat index offset $O_k$ and has length $L_k$:
$$O_k = \sum_{j=1}^{k-1} L_j$$

A window starting at flat index $s$ spans $[s, s + W - 1]$. If the window crosses the boundary of block $k$, it spans into block $k+1$:
$$s + W - 1 \ge O_{k+1} \implies s + W - 1 \ge O_k + L_k$$

To prevent this crossover, the window must be contained entirely within block $k$:
$$O_k \le s \le O_k + L_k - W$$

Any start index $s$ in the boundary region:
$$s \in [O_k + L_k - W + 1, O_k + L_k - 1]$$
is excluded from the valid indices set.

---

### 4.2 Code-to-Math Trace in `__getitem__`

```python
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
        # Map sequential index pointer to discontinuous valid index array
        start = self._valid_indices[idx]
        mid   = start + self.t_lookback
        end   = mid   + self.t_forecast

        x_raw: np.ndarray = self._data[start:mid]
        y_raw: np.ndarray = self._data[mid:end]
```

1. **Index Resolution**:
   The index mapping translates the sequential index `idx` to the valid starting index:
   $$s = \text{valid\_indices}[\text{idx}]$$
2. **Slicing**:
   The lookback and forecast windows are sliced from the contiguous array:
   $$\mathbf{X}_{\text{raw}} = \mathbf{X}_{s : s + L_{\text{lookback}}, :}$$
   $$\mathbf{Y}_{\text{raw}} = \mathbf{Y}_{s + L_{\text{lookback}} : s + L_{\text{lookback}} + L_{\text{forecast}}, :}$$
3. **Data Contiguity**:
   Because $s$ satisfies the boundary constraint $O_k \le s \le O_k + L_k - W$, both slices lie entirely within the contiguous bounds of block $k$, avoiding data leakage across discontinuous blocks.
