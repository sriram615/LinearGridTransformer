# LinearGridTransformer

A production-grade, dual-dataset multi-zone electricity forecasting system comparing a Taylor-feature linear-attention Transformer with a compact direct-linear baseline across **NYISO bulk transmission grid** telemetry (2023, Megawatts) and the UCI **Tetouan City** distribution dataset (2017). The project trains both architectures across three independent seeds (42, 123, 2026), validates through a 7-point zero-leakage scientific protocol, confirms statistical superiority via the Diebold-Mariano test, and exposes inference and historical experiment governance through a FastAPI service, TimescaleDB telemetry backend, and interactive React dashboard.

The main engineering result: **CDLinear was selected as the operational forecaster** — it achieved 66.4% lower validation loss on NYISO Grid and statistically significant superiority over LinearGridTransformer (Harvey-Leybourne-Newbold corrected *DM* = 72.63, *p* < 10⁻¹⁵). The Transformer remains available as a comparative baseline. Model-family selection for NYISO Grid was decided strictly on validation loss prior to test evaluation, avoiding test-selection bias; on the legacy Tetouan benchmark, held-out test metrics informed the selection.

> **Live demo:** [Frontend dashboard](https://lineargridtransformer-frontend.onrender.com) · [Backend API](https://lineargridtransformer-api.onrender.com) · [API Docs (Swagger)](https://lineargridtransformer-api.onrender.com/docs)
>
> Render automatic deployment is disabled; repository changes require a manual service redeploy. Free-tier services may experience cold starts after idle periods.

## What is included

- **Dual-dataset forecasting** across NYISO Regional Grid (Megawatts) and Tetouan City (legacy benchmark).
- `LinearGridTransformer`, using a second-degree Taylor feature map and associative linear attention.
- `CDLinear`, a direct flatten-and-linear-projection baseline.
- Persistence and training-mean as non-trained evaluation baselines.
- Three-seed (`42`, `123`, `2026`) training histories, per-zone, per-horizon, and per-seed evaluation reports.
- **TimescaleDB hypertable** telemetry storage with 1-month chunk partitioning, columnar compression, and continuous aggregations.
- **1-click telemetry retrieval** in the dashboard: fetches the latest 144-step (24-hour) lookback window from TimescaleDB with automatic fallback to disk cache.
- **Diebold-Mariano econometric significance testing** (HLN-corrected) confirming CDLinear's superiority.
- A **7-Point Scientific Governance Framework** enforcing zero data leakage.
- A FastAPI backend for experiment data, telemetry retrieval, and CPU inference.
- A React dashboard with dataset switcher, 1-click telemetry loader, and five analytical sections.
- Docker image, Docker Compose, and Render Blueprint for deployment.
- Optional local MLflow experiment tracking and model registry.

## Datasets

### NYISO Regional Grid (Primary)

| Property | Value |
|---|---|
| **Source** | [NYISO Public Market Data](http://mis.nyiso.com/public/csv/pal/) (PAL zonal load archives) |
| **Period** | 2023-01-01 00:00:00 through 2023-12-31 23:00:00 (52,555 observations) |
| **Cadence** | Uniform 10-minute intervals |
| **Timezone** | UTC with timezone metadata (`America/New_York` source reference) |
| **Physical Units** | **Megawatts (MW)** — documented bulk regional transmission power |
| **Zones** | New York City (N.Y.C.), Long Island (LONGIL), Hudson Valley (HUD VL) |
| **Weather** | Open-Meteo ERA5 reanalysis (temperature, humidity, wind speed, solar radiation) |
| **Partitions** | **Train:** 36,788 rows (Jan 1 – Sep 13, 2023)<br/>**Validation:** 7,883 rows (Sep 13 – Nov 7, 2023)<br/>**Test:** 7,884 rows (Nov 7 – Dec 31, 2023) |
| **Storage** | PostgreSQL / TimescaleDB hypertable with disk-cached fallback (`series.npz`) |

The grid ETL pipeline downloads NYISO transmission load CSVs and matching Open-Meteo weather reanalysis, then formats them into the canonical 8-channel contract. Ingesting into PostgreSQL/TimescaleDB enables sliding-window telemetry retrieval via `GET /api/v1/grid/telemetry/latest?limit=144`.

### Tetouan City (Legacy Benchmark)

| Property | Value |
|---|---|
| **Source** | [UCI Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city) (Salam & El Hibaoui, 2018) |
| **Period** | 2017-01-01 00:00 through 2017-12-30 23:50 (52,416 observations; Dec 31 absent) |
| **Cadence** | 10-minute intervals |
| **Timezone** | Timezone-naive (undocumented) |
| **Physical Units** | Undocumented (original dataset numeric values) |
| **Zones** | Zone 1, Zone 2, Zone 3 (distribution feeders) |
| **Partitions** | **Train:** 36,691 rows (36,542 windows)<br/>**Validation:** 7,862 rows (7,857 windows)<br/>**Test:** 7,863 rows (7,858 windows) |

## ETL pipeline and data quality audit

The ingestion pipeline [`src/data/grid_etl.py`](src/data/grid_etl.py) downloads NYISO transmission load CSVs and Open-Meteo weather reanalysis, applies resampling and timestamp synchronization, and enforces an automated data-quality audit prior to persistence.

The official quality audit report is generated at [`results/etl/quality_report.json`](results/etl/quality_report.json):

| Audit Category | Metric / Check | Recorded Result | Status |
|---|---|---|---|
| **Total Observations** | Uniform 10-minute records | 52,555 rows | **Passed** |
| **Cadence Continuity** | Adjacent row delta == 10 min | 52,554 / 52,554 intervals exact (0 cadence violations) | **Passed** |
| **Internal Missing Intervals** | Date range gaps `[min, max]` | 0 missing steps between Jan 1 00:00 and Dec 31 23:00 | **Passed** |
| **Calendar Horizon Reconciliation** | Nominal 365 days (52,560) vs observed 52,555 | 5-step boundary deficit (50 min, Dec 31 23:10–23:50) due to Open-Meteo hourly weather reanalysis ceiling (23:00 UTC) | **Reconciled** |
| **Duplicates** | Duplicate timestamps or rows | 0 timestamp duplicates, 0 row duplicates | **Passed** |
| **Missing / Null Values** | NaNs across all 9 channels | 0 nulls detected | **Passed** |
| **Data Types** | Datetime parsing & float64 cast | 0 type coercion failures | **Passed** |
| **Invalid Values** | Non-finite values (`inf`, `-inf`) | 0 non-finite values | **Passed** |
| **Expected Zones** | NYC, Long Island, Hudson Valley | 3 / 3 expected transmission zones present | **Passed** |
| **Physical Bounds** | Zone 1 (NYC) Load | Min: 3,689.88 MW, Max: 10,413.68 MW (bounds: 100–25,000 MW; baseload ~3.5 GW, historical peak ~11.5 GW) | **Passed** |
| **Physical Bounds** | Zone 2 (Long Island) Load | Min: 1,238.33 MW, Max: 5,054.60 MW (bounds: 50–15,000 MW; baseload ~1.2 GW, summer peak ~5.5 GW) | **Passed** |
| **Physical Bounds** | Zone 3 (Hudson Valley) Load | Min: 410.36 MW, Max: 2,055.93 MW (bounds: 20–10,000 MW; baseload ~400 MW, summer peak ~2.1 GW) | **Passed** |
| **Physical Bounds** | Temperature | Min: -16.5°C, Max: 35.9°C (bounds: -60°C to +60°C; NY records: -47°C to +42°C) | **Passed** |
| **Physical Bounds** | Relative Humidity | Min: 14.0%, Max: 100.0% (bounds: 0% to 100%; physical definition) | **Passed** |
| **Physical Bounds** | Wind Speed | Min: 0.0 km/h, Max: 53.5 km/h (bounds: 0–150 km/h; Open-Meteo km/h, 150 km/h ~41.7 m/s Cat 1 hurricane) | **Passed** |
| **Physical Bounds** | General Diffuse Solar | Min: 0.0 W/m², Max: 421.0 W/m² (bounds: 0–2,000 W/m²; clear-sky diffuse ceiling) | **Passed** |
| **Physical Bounds** | Direct Normal Irradiance | Min: 0.0 W/m², Max: 962.1 W/m² (bounds: 0–2,000 W/m²; clear-sky direct surface ceiling) | **Passed** |

To run the data-quality audit or re-generate the report:
```bash
# Audit an existing dataset CSV
python -m src.data.grid_etl --validate-file data/raw/grid/grid_power_consumption.csv

# Generate JSON quality report
python -m src.data.grid_etl --generate-report --report-output results/etl/quality_report.json
```

## Prediction contract

Both datasets share the identical prediction contract:

- **Input:** 144 historical observations (24 hours), eight channels, shape `[B, 144, 8]`.
- **Output:** six forecast steps (one hour), three target channels, shape `[B, 6, 3]`.
- **Sampling:** ten minutes; forecast leads are +10, +20, +30, +40, +50 and +60 minutes.

Input channels (canonical order):

1. `zone_1_power_consumption` — Zone 1 load (NYC / Zone 1)
2. `zone_2_power_consumption` — Zone 2 load (Long Island / Zone 2)
3. `zone_3_power_consumption` — Zone 3 load (Hudson Valley / Zone 3)
4. `temperature` — Temperature
5. `humidity` — Humidity / Relative humidity
6. `wind_speed` — Wind speed
7. `general_diffuse_flows` — General diffuse solar radiation
8. `diffuse_flows` — Direct/diffuse solar radiation

Output channels are `zone_1_power_consumption`, `zone_2_power_consumption`, and `zone_3_power_consumption`. Feature and target z-score statistics use training rows only (`ddof=0`). Predictions and targets are inverse-transformed with training target means and standard deviations before computing evaluation metrics. NYISO Grid results are expressed in **real Megawatts (MW)**.

## System architecture

```mermaid
flowchart LR
    subgraph Data Sources
        NYISO[NYISO PAL Archives]
        METEO[Open-Meteo Reanalysis]
        UCI[UCI Tetouan CSV]
    end

    subgraph ETL & Storage
        GETL[Grid ETL Pipeline]
        TSDB[(TimescaleDB<br/>Hypertable)]
        DISK[Disk Fallback Cache<br/>series.npz]
        TETL[Tetouan Preprocessing]
    end

    subgraph Training & Governance
        WINDOWS[Chronological Windows<br/>Train / Val / Test]
        TRAIN[Tri-Seed Training<br/>Seeds 42, 123, 2026]
        EVAL[Zero-Leakage Validation<br/>& Test Evaluation]
        DM[Diebold-Mariano Test<br/>DM=72.63, p<1e-15]
    end

    subgraph Serving
        API[FastAPI Backend]
        STORE[Experiment Stores<br/>Grid + Tetouan]
        INF[CPU Inference<br/>CDLinear + LGT]
        REACT[React Dashboard]
    end

    NYISO --> GETL
    METEO --> GETL
    GETL --> TSDB
    GETL --> DISK
    UCI --> TETL --> WINDOWS
    GETL --> WINDOWS
    WINDOWS --> TRAIN --> EVAL --> DM
    EVAL --> STORE
    TSDB -.->|REST /telemetry/latest| API
    DISK -.->|fallback| API
    STORE --> API
    INF --> API
    REACT --> API
    USER[Browser / 1-Click Telemetry / CSV] --> REACT
```

## Models and experimental method

### LinearGridTransformer

The model projects the eight input channels to a 64-dimensional representation, applies a single encoder block with four linear-attention heads (16 dimensions per head), residual and feed-forward paths, then maps the encoded sequence to forecast outputs. The attention module uses a second-degree Taylor feature map and associative key/value contractions, avoiding quadratic sequence-length memory cost. A forecasting head projects the final six encoded positions from eight encoder output channels to three target channels.

### CDLinear

CDLinear flattens the `[144, 8]` input window into 1,152 values and applies one learned linear projection to 18 outputs, reshaped to `[6, 3]`. It has no attention or nonlinear decoder. This simpler mapping was substantially faster in CPU inference and, across both datasets and all random seeds, achieved lower errors than the Transformer.

### Training and checkpoint selection

Both models were trained for 50 epochs across seeds **42, 123, and 2026** on both datasets. Configurations: batch size 32, AdamW, learning rate 0.001, weight decay 0.01, MSE loss, and gradient clipping at 1.0. The Transformer uses `d_model=64`, four heads, `ffn_dim=256`, and dropout 0.1.

- **NYISO Grid:** Checkpoints were selected by lowest validation loss within each seed. CDLinear seed 42 was selected as the operational forecaster with a validation loss of **0.005920** (vs. 0.017643 for LinearGridTransformer — a **66.4% advantage**).
- **Zero-leakage guarantee:** Model-family selection for NYISO Grid was decided strictly on validation loss prior to test partition access.

### 7-Point Scientific Governance Framework

| # | Principle | Legacy Tetouan (2017) | NYISO Regional Grid (2023) | Status |
|---|---|---|---|---|
| 1 | **Zero-Leakage Governance** | Model family chosen by inspecting test metrics | Model family frozen strictly on validation partition prior to test access | **Eliminated** |
| 2 | **Chronological Holdout Protocol** | 2017 partition explored during retrospective research | Held-out test partition (Nov 7 – Dec 31, 2023; 7,884 windows) strictly untouched during training | **Resolved** |
| 3 | **Standardized Physical Units** | Undocumented numeric counts; physical units unknown | Real Megawatts (MW) bulk regional transmission (NYC ~5.7 GW, LI ~2.3 GW, HUD ~1.0 GW) | **Verified MW** |
| 4 | **Timezone & Cadence Parity** | Timezone-naive timestamps, missing final day | America/New_York (US Eastern Time), uniform UTC 10-minute cadence | **Synchronized** |
| 5 | **Multi-Seed Stability** | Single training run in baseline | 3-Seed execution (Seeds 42, 123, 2026) with mean ± population standard deviation | **Tri-Seed Verified** |
| 6 | **Econometric Significance** | Raw metric differences without significance test | Harvey-Leybourne-Newbold corrected Diebold-Mariano test (*DM* = 72.63, *p* < 10⁻¹⁵) | **p < 10⁻¹⁵** |
| 7 | **TimescaleDB Telemetry Backend** | Static offline CSV artifact only | PostgreSQL TimescaleDB hypertable with indexed lookback queries and disk fallback | **Production Ready** |

### Diebold-Mariano econometric significance

To verify that CDLinear's outperformance is not a sampling artifact, the Diebold-Mariano test with Harvey-Leybourne-Newbold (HLN) finite-sample correction was computed:

| Metric | Value |
|---|---|
| **Test statistic** | *DM* = 72.63 |
| **p-value** | *p* < 10⁻¹⁵ |
| **Conclusion** | CDLinear statistically outperforms LinearGridTransformer at *p* < 10⁻¹⁵ |

## Recorded results

### NYISO Regional Grid (Primary — Megawatts)

Values from [`results/grid/`](results/grid/), evaluated across 7,884 held-out chronological test windows (Nov 7 – Dec 31, 2023). Results report three-seed mean ± population standard deviation across seeds 42, 123, and 2026. Persistence is a deterministic baseline (no across-seed variation).

#### Macro-average across all 3 zones

| Model | Test MAE (MW), mean ± SD | Test RMSE (MW), mean ± SD |
|---|---:|---:|
| **CDLinear (selected)** | **32.19 ± 1.17** | **42.07 ± 1.05** |
| LinearGridTransformer (comparative baseline) | 68.08 ± 2.57 | 88.25 ± 2.94 |
| Persistence (evaluation benchmark) | 54.45 | 74.93 |

#### Per-zone performance (mean across 3 seeds)

| Model | Zone | MAE (MW) | RMSE (MW) |
|---|---|---:|---:|
| **CDLinear** | New York City (N.Y.C.) | **46.83** | **60.82** |
| **CDLinear** | Long Island (LONGIL) | **30.70** | **40.32** |
| **CDLinear** | Hudson Valley (HUD VL) | **19.04** | **25.06** |
| LinearGridTransformer | New York City (N.Y.C.) | 108.02 | 143.51 |
| LinearGridTransformer | Long Island (LONGIL) | 61.44 | 77.43 |
| LinearGridTransformer | Hudson Valley (HUD VL) | 34.77 | 43.82 |
| Persistence | New York City (N.Y.C.) | 84.25 | 118.61 |
| Persistence | Long Island (LONGIL) | 51.46 | 69.09 |
| Persistence | Hudson Valley (HUD VL) | 27.65 | 37.10 |

### Tetouan City (Legacy Benchmark — Original Numeric Values)

From [`evaluation_report.json`](runtime-artifacts/results/tetouan/extended_training_v2/evaluation_test_v1/evaluation_report.json), evaluated across 7,858 held-out chronological test windows across seeds 42, 123, and 2026:

| Model | Test MAE, mean ± SD | Test RMSE, mean ± SD |
|---|---:|---:|
| **CDLinear (selected)** | **525.10 ± 52.13** | **713.74 ± 44.17** |
| LinearGridTransformer (experimental alternative) | 1,904.08 ± 241.17 | 2,386.88 ± 269.79 |
| Persistence (evaluation baseline) | 881.21 | 1,424.05 |

## TimescaleDB telemetry backend

The NYISO grid dataset is stored in a PostgreSQL/TimescaleDB time-series database optimized for fast lookback retrieval.

### Schema design

```sql
-- Hypertable partitioned by 1-month chunks with columnar compression
CREATE TABLE grid_telemetry (
    time                  TIMESTAMPTZ NOT NULL,
    zone_id               SMALLINT    NOT NULL REFERENCES grid_zones(zone_id),
    load_mw               REAL        NOT NULL,
    temperature_c         REAL,
    relative_humidity_pct REAL,
    wind_speed_ms         REAL,
    diffuse_solar_wm2     REAL,
    direct_solar_wm2      REAL
);
SELECT create_hypertable('grid_telemetry', 'time', chunk_time_interval => INTERVAL '1 month');

-- Continuous aggregate for hourly rollups
CREATE MATERIALIZED VIEW grid_telemetry_hourly ...

-- Wide view aligned to the 8-channel input contract
CREATE MATERIALIZED VIEW grid_telemetry_wide ...
```

### Retrieval architecture

- **Sliding-window lookback:** The API queries `grid_telemetry_wide` for the most recent 144 records (`GET /api/v1/grid/telemetry/latest?limit=144`) using time-descending indexed queries (<35ms execution).
- **Graceful degradation:** If TimescaleDB is not configured or temporarily unreachable, the endpoint automatically falls back to reading the pre-processed telemetry series from `data/processed/grid/series.npz`.
- **Note on streaming:** Telemetry retrieval is request-driven (REST API sliding-window queries). It does not employ WebSocket or SSE push streaming.

## Dashboard and forecast workflow

The React dashboard features a topbar dataset toggle for switching between **NYISO Grid (MW)** and **Tetouan (Legacy)**.

### Sections

- **Overview:** Dataset summary, active experiment, model comparison metrics, **Diebold-Mariano significance card**, and **7-Point Scientific Framework matrix**.
- **Forecast Explorer:** Upload a 144-row CSV, or click **"⚡ Load Live Telemetry (TimescaleDB)"** to load the latest 24 hours of data into the lookback window. Supports single-model and comparative inference with dynamic MW/legacy unit formatting.
- **Model Laboratory:** Training/validation loss curves, model configurations, per-zone and per-horizon evaluation charts, and seed-level metric tables.
- **Dataset Explorer:** Provenance, temporal coverage, sampling cadence, channel descriptions, chronological split boundaries, normalization statistics, and data quality checks.
- **Experiment Details:** Training hyperparameter manifests, hardware environment, best-validation checkpoints, and artifact verification status.

![Overview dashboard](docs/images/frontend/dashboard-overview.png)

![Model Laboratory](docs/images/frontend/model-lab.png)

![Model Laboratory: per-zone and horizon analysis](docs/images/frontend/model-lab-per-zone-and-horizon.png)

![Model Laboratory: seed-level metrics](docs/images/frontend/model-lab-seed-metrics.png)

![Dataset Explorer](docs/images/frontend/dataset-explorer.png)

![Forecast Explorer input](docs/images/frontend/forecast-explorer.png)

## API reference

| Method and path | Purpose |
|---|---|
| `GET /` | Public project information and links |
| `GET /health` | Service health and artifact readiness status |
| `GET /api/v1/datasets` | List available datasets (`grid`, `tetouan`) with metadata |
| `GET /api/v1/grid/telemetry/latest?limit=144` | Fetch latest *N* observations from TimescaleDB (with disk fallback) |
| `GET /api/v1/models?dataset=grid` | Forecastable models, parameters, and selection status |
| `GET /api/v1/dataset?dataset=grid` | Dataset provenance, channels, split boundaries, and normalization |
| `GET /api/v1/experiments/{experiment_id}` | Experiment configuration, model roles, and econometric significance |
| `GET /api/v1/experiments/{experiment_id}/history` | Training and validation epoch loss curves |
| `GET /api/v1/experiments/{experiment_id}/metrics` | Macro, per-zone, per-horizon, and per-seed evaluation metrics |
| `POST /api/v1/forecasts/{model_name}?dataset=grid` | Run single-model forecast on 144-row input |
| `POST /api/v1/forecasts/compare?dataset=grid` | Run both models simultaneously on the same 144-row input |

Supported `experiment_id` values: `grid_colab_v1` (NYISO primary), `extended_training_v2` (Tetouan extended), `tetouan_t4_20261101_v1` (Tetouan original).

## Run locally

### Docker Compose (recommended)

Requirements: Docker Engine / Docker Desktop with Compose v2.

```bash
docker compose config
docker compose build
docker compose up -d
```

Open the dashboard at [http://localhost:8080](http://localhost:8080). Nginx proxies API calls to the FastAPI container.

```bash
docker compose logs -f backend frontend
docker compose down
```

### Native development

**Backend** (Python 3.11):

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements-api.txt

# Configure grid artifact paths (defaults match repository layout)
export GRID_ARTIFACT_DIR="$PWD/data/processed/grid"
export GRID_EXPERIMENT_DIR="$PWD/results/grid"
export GRID_CHECKPOINT_DIR="$PWD/checkpoints/grid"
export GRID_RUN_NAME="grid_colab_s42"

# Optional: TimescaleDB connection (falls back to series.npz if unset or offline)
export GRID_DB_URL="postgresql://postgres:postgres@localhost:5433/grid_forecasting"

# Configure Tetouan artifact paths
export TETOUAN_ARTIFACT_DIR="$PWD/runtime-artifacts/data/processed/tetouan"
export TETOUAN_EXPERIMENT_DIR="$PWD/runtime-artifacts/results/tetouan/extended_training_v2"
export TETOUAN_CHECKPOINT_DIR="$PWD/runtime-artifacts/checkpoints/tetouan"

export DEVICE=cpu
export API_CORS_ORIGINS=http://localhost:8443,http://127.0.0.1:8443,http://localhost:5173,http://127.0.0.1:5173
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

**Frontend** (Node ≥ 20):

```bash
cd frontend
pnpm install
VITE_API_BASE_URL=http://127.0.0.1:8000 pnpm dev -- --host 127.0.0.1
```

Vite serves the UI on port 8443 (or port 5173 depending on configuration).

### TimescaleDB ingestion CLI

To ingest NYISO load archives and weather data into PostgreSQL/TimescaleDB:

```bash
# Ingest 2023 NYISO data into the configured database
python -m src.data.grid_etl --year 2023 --load-db

# Or export to CSV without database loading
python -m src.data.grid_etl --year 2023 --output data/raw/grid/grid_power_consumption.csv
```

### Multi-seed training pipeline CLI

```bash
# Dry run verification
.venv311/bin/python3 -m src.pipelines.tetouan_pipeline \
  --run-name tetouan-research-2027-01 --seeds 42 123 2026

# Execute 50-epoch training across seeds 42, 123, 2026
.venv311/bin/python3 -m src.pipelines.tetouan_pipeline \
  --run-name tetouan-research-2027-01 --seeds 42 123 2026 \
  --epochs 50 --batch-size 32 --execute
```

### Drift monitoring CLI

```bash
mkdir -p results/monitoring
.venv311/bin/python3 -m src.monitoring.tetouan_drift \
  --current-features /path/to/current_features.npy \
  --output results/monitoring/tetouan-window-001.json \
  --log-mlflow
```

### Local MLflow tracking (optional)

```bash
mlflow ui \
  --backend-store-uri "sqlite:///$(pwd)/.mlflow/tracking.db" \
  --default-artifact-root "$(pwd)/.mlflow/artifacts" \
  --host 127.0.0.1 --port 5000
```

![MLflow experiment runs](docs/images/mlflow/experiment-tracking.png)

![Registered Tetouan models](docs/images/mlflow/model-registry.png)

![Registered model versions and source-run lineage](docs/images/mlflow/model-version-lineage.png)

## Deployment

`render.yaml` defines a free Docker Web Service for FastAPI and a free Static Site for the React dashboard. The backend image bundles the pre-computed artifacts and checkpoints, so the service does not require an active database connection or persistent disk to serve historical data or execute CPU inference.

| Service | Type | Plan |
|---|---|---|
| `lineargridtransformer-api` | Docker Web Service | Free |
| `lineargridtransformer-frontend` | Static Site (pnpm build) | Free |

Automatic deployment is disabled (`autoDeployTrigger: 'off'`). Redeploy manually from the Render dashboard after pushing to `main`.

## Repository map

```text
LinearGridTransformer/
├── src/
│   ├── api/                    # FastAPI routes, schemas, and configuration
│   ├── data/
│   │   ├── db/                 # TimescaleDB schema (DDL), PostgresManager
│   │   ├── grid_etl.py         # NYISO + Open-Meteo ETL and ingest CLI
│   │   ├── grid_experiment_store.py  # NYISO 3-seed experiment metadata & metrics
│   │   └── ...                 # Tetouan preprocessing, windowing, and store
│   ├── inference/
│   │   ├── grid_service.py     # Grid checkpoint loading and MW inference
│   │   └── ...                 # Tetouan inference service
│   ├── models/                 # Linear attention, Taylor feature map, CDLinear
│   ├── mlops/                  # MLflow model registry and historical importers
│   ├── monitoring/             # Population Stability Index (PSI) drift monitoring
│   ├── pipelines/              # Multi-seed training pipeline CLI
│   └── training/               # Training harness and checkpoint evaluation
├── frontend/
│   ├── src/
│   │   ├── api/                # TypeScript types, API client, and AppData context
│   │   ├── components/         # Status badges, charts, and layout components
│   │   ├── App.tsx             # Main dashboard (dataset switcher, sections)
│   │   └── index.css           # Styling, animations, and responsive rules
│   ├── Dockerfile              # Production Nginx container
│   └── nginx.conf              # Reverse proxy configuration
├── results/grid/               # 3-seed NYISO manifests, histories, and eval reports
├── results/etl/                # Automated ETL data-quality audit report (quality_report.json)
├── checkpoints/grid/           # PyTorch checkpoints for CDLinear and LGT (seeds 42, 123, 2026)
├── data/processed/grid/        # NYISO metadata.json and cached series.npz
├── runtime-artifacts/          # Tetouan checkpoints, histories, and eval reports
├── docs/                       # Architecture specifications, diagrams, and images
├── Dockerfile                  # CPU FastAPI backend container
├── compose.yaml                # Local multi-container Docker Compose
├── render.yaml                 # Render Blueprint specification
├── requirements.txt            # Complete Python dependencies
├── requirements-api.txt        # Pinned production API dependencies
└── .env.example                # Environment variable configuration template
```

## Limitations and next research steps

- **Test-selection bias in Tetouan:** On the legacy Tetouan benchmark, the held-out test comparison informed model-family selection. On NYISO Grid, model selection was pre-specified strictly on validation MSE.
- **Three seeds:** While the Diebold-Mariano test establishes statistical significance beyond seed variability, evaluation is conducted over a single fixed chronological test partition.
- **Sliding-window lookback vs. streaming:** Telemetry retrieval operates as request-driven REST queries for 144-step lookback windows; continuous WebSocket or SSE push streaming is not implemented.
- **Database dependency:** TimescaleDB provides fast indexing and aggregation, but the system relies on disk-cached series fallback when a live database is unavailable.
- **Inference scope:** CPU-only inference is configured for free-tier compatibility; GPU acceleration is supported via environment variable (`DEVICE=cuda`).
- **Baseline dominance:** Across both datasets, the direct linear projection (`CDLinear`) outperforms the Taylor-feature linear-attention Transformer (`LinearGridTransformer`), demonstrating that increased model complexity does not inherently improve short-horizon zonal energy forecasting.

## Tech stack

| Layer | Technology |
|---|---|
| **ML Framework** | PyTorch ≥ 2.0, NumPy, Pandas |
| **Backend** | FastAPI, Uvicorn, Pydantic v2 |
| **Database** | PostgreSQL + TimescaleDB (optional, with disk fallback) |
| **Frontend** | React 19, TypeScript, Vite |
| **Experiment Tracking** | MLflow 3.x (local research registry) |
| **Deployment** | Docker, Nginx, Render (Blueprint) |
| **Data Sources** | NYISO Public Market Data (PAL), Open-Meteo ERA5, UCI ML Repository |

## Acknowledgements and references

- **NYISO** — [New York Independent System Operator](http://mis.nyiso.com/public/csv/pal/) public market data for regional transmission load.
- **Open-Meteo** — ERA5 weather reanalysis data for atmospheric features.
- Salam, A. & El Hibaoui, A. (2018). [Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city). UCI Machine Learning Repository. DOI: [10.24432/C5B034](https://doi.org/10.24432/C5B034).
- Diebold, F.X. & Mariano, R.S. (1995). "Comparing predictive accuracy." *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). "Testing the equality of prediction mean squared errors." *International Journal of Forecasting*, 13(2), 281–291.
- Source repository: [sriram615/LinearGridTransformer](https://github.com/sriram615/LinearGridTransformer).
