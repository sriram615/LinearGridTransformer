-- ==============================================================================
-- LinearGridTransformer: Industry-Grade PostgreSQL / TimescaleDB Preprocessing
-- Deterministic Chronological Splits & Population Normalization Pipeline
-- ==============================================================================

-- 1. Create Split Bounds Table (Records exact split cutoffs)
CREATE TABLE IF NOT EXISTS grid_split_bounds (
    split_name VARCHAR(32) PRIMARY KEY,
    start_index INT NOT NULL,
    end_index_exclusive INT NOT NULL,
    row_count INT NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time_exclusive TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Seed Chronological Split Boundaries: 70% Train, 15% Validation, 15% Test
-- For 52,555 ten-minute timestamps (2023-01-01 00:00 UTC to 2023-12-31 23:00 UTC):
-- - Train:      indices [0, 36788)     (36,788 rows: 2023-01-01 00:00 to 2023-09-13 11:20 UTC)
-- - Validation: indices [36788, 44671)  (7,883 rows: 2023-09-13 11:20 to 2023-11-07 05:10 UTC)
-- - Test:       indices [44671, 52555)  (7,884 rows: 2023-11-07 05:10 to 2023-12-31 23:10 UTC)
INSERT INTO grid_split_bounds (split_name, start_index, end_index_exclusive, row_count, start_time, end_time_exclusive)
VALUES
    ('train', 0, 36788, 36788, '2023-01-01 00:00:00+00', '2023-09-13 11:20:00+00'),
    ('validation', 36788, 44671, 7883, '2023-09-13 11:20:00+00', '2023-11-07 05:10:00+00'),
    ('test', 44671, 52555, 7884, '2023-11-07 05:10:00+00', '2023-12-31 23:10:00+00')
ON CONFLICT (split_name) DO UPDATE SET
    start_index = EXCLUDED.start_index,
    end_index_exclusive = EXCLUDED.end_index_exclusive,
    row_count = EXCLUDED.row_count,
    start_time = EXCLUDED.start_time,
    end_time_exclusive = EXCLUDED.end_time_exclusive;

-- 2. Materialized Wide View: Pivots Multi-Zone Telemetry into 8 Canonical Channels
CREATE MATERIALIZED VIEW IF NOT EXISTS grid_telemetry_wide AS
SELECT
    time,
    MAX(CASE WHEN zone_id = 'ZONE_1' THEN load_mw END) AS zone_1_power_consumption,
    MAX(CASE WHEN zone_id = 'ZONE_2' THEN load_mw END) AS zone_2_power_consumption,
    MAX(CASE WHEN zone_id = 'ZONE_3' THEN load_mw END) AS zone_3_power_consumption,
    MAX(temperature_c) AS temperature,
    MAX(relative_humidity_pct) AS humidity,
    MAX(wind_speed_ms) AS wind_speed,
    MAX(diffuse_solar_wm2) AS general_diffuse_flows,
    MAX(direct_solar_wm2) AS diffuse_flows
FROM grid_telemetry
GROUP BY time
ORDER BY time ASC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_grid_telemetry_wide_time ON grid_telemetry_wide (time ASC);

-- 3. Train-Only Normalization Statistics Table
-- STRICT CONTRACT: Only compute statistics on rows strictly prior to train_end_time.
-- Prevents look-ahead leakage into validation and test sets.
CREATE TABLE IF NOT EXISTS grid_train_stats (
    stat_id INT PRIMARY KEY DEFAULT 1,
    train_end_time TIMESTAMPTZ NOT NULL,
    train_row_count INT NOT NULL,
    zone_1_mean DOUBLE PRECISION NOT NULL,
    zone_1_std DOUBLE PRECISION NOT NULL,
    zone_2_mean DOUBLE PRECISION NOT NULL,
    zone_2_std DOUBLE PRECISION NOT NULL,
    zone_3_mean DOUBLE PRECISION NOT NULL,
    zone_3_std DOUBLE PRECISION NOT NULL,
    temp_mean DOUBLE PRECISION NOT NULL,
    temp_std DOUBLE PRECISION NOT NULL,
    humidity_mean DOUBLE PRECISION NOT NULL,
    humidity_std DOUBLE PRECISION NOT NULL,
    wind_mean DOUBLE PRECISION NOT NULL,
    wind_std DOUBLE PRECISION NOT NULL,
    diffuse_mean DOUBLE PRECISION NOT NULL,
    diffuse_std DOUBLE PRECISION NOT NULL,
    direct_mean DOUBLE PRECISION NOT NULL,
    direct_std DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO grid_train_stats (
    stat_id, train_end_time, train_row_count,
    zone_1_mean, zone_1_std, zone_2_mean, zone_2_std, zone_3_mean, zone_3_std,
    temp_mean, temp_std, humidity_mean, humidity_std, wind_mean, wind_std,
    diffuse_mean, diffuse_std, direct_mean, direct_std
)
SELECT
    1,
    b.end_time_exclusive,
    COUNT(*)::INT,
    AVG(w.zone_1_power_consumption),
    STDDEV_POP(w.zone_1_power_consumption),
    AVG(w.zone_2_power_consumption),
    STDDEV_POP(w.zone_2_power_consumption),
    AVG(w.zone_3_power_consumption),
    STDDEV_POP(w.zone_3_power_consumption),
    AVG(w.temperature),
    STDDEV_POP(w.temperature),
    AVG(w.humidity),
    STDDEV_POP(w.humidity),
    AVG(w.wind_speed),
    STDDEV_POP(w.wind_speed),
    AVG(w.general_diffuse_flows),
    STDDEV_POP(w.general_diffuse_flows),
    AVG(w.diffuse_flows),
    STDDEV_POP(w.diffuse_flows)
FROM grid_telemetry_wide w
CROSS JOIN (SELECT end_time_exclusive FROM grid_split_bounds WHERE split_name = 'train') b
WHERE w.time < b.end_time_exclusive
GROUP BY b.end_time_exclusive
ON CONFLICT (stat_id) DO UPDATE SET
    train_end_time = EXCLUDED.train_end_time,
    train_row_count = EXCLUDED.train_row_count,
    zone_1_mean = EXCLUDED.zone_1_mean,
    zone_1_std = EXCLUDED.zone_1_std,
    zone_2_mean = EXCLUDED.zone_2_mean,
    zone_2_std = EXCLUDED.zone_2_std,
    zone_3_mean = EXCLUDED.zone_3_mean,
    zone_3_std = EXCLUDED.zone_3_std,
    temp_mean = EXCLUDED.temp_mean,
    temp_std = EXCLUDED.temp_std,
    humidity_mean = EXCLUDED.humidity_mean,
    humidity_std = EXCLUDED.humidity_std,
    wind_mean = EXCLUDED.wind_mean,
    wind_std = EXCLUDED.wind_std,
    diffuse_mean = EXCLUDED.diffuse_mean,
    diffuse_std = EXCLUDED.diffuse_std,
    direct_mean = EXCLUDED.direct_mean,
    direct_std = EXCLUDED.direct_std;

-- 4. Normalized View: Standardized Telemetry using Train-Only Mean and Std
CREATE OR REPLACE VIEW grid_telemetry_normalized AS
SELECT
    w.time,
    (w.zone_1_power_consumption - s.zone_1_mean) / s.zone_1_std AS norm_zone_1,
    (w.zone_2_power_consumption - s.zone_2_mean) / s.zone_2_std AS norm_zone_2,
    (w.zone_3_power_consumption - s.zone_3_mean) / s.zone_3_std AS norm_zone_3,
    (w.temperature - s.temp_mean) / s.temp_std AS norm_temperature,
    (w.humidity - s.humidity_mean) / s.humidity_std AS norm_humidity,
    (w.wind_speed - s.wind_mean) / s.wind_std AS norm_wind_speed,
    (w.general_diffuse_flows - s.diffuse_mean) / s.diffuse_std AS norm_diffuse_flows,
    (w.diffuse_flows - s.direct_mean) / s.direct_std AS norm_direct_flows
FROM grid_telemetry_wide w
CROSS JOIN grid_train_stats s
ORDER BY w.time ASC;

-- 5. Split-Specific Normalized Views for Fast Training & Evaluation Queries
CREATE OR REPLACE VIEW grid_train_view AS
SELECT n.*
FROM grid_telemetry_normalized n
CROSS JOIN (SELECT start_time, end_time_exclusive FROM grid_split_bounds WHERE split_name = 'train') b
WHERE n.time >= b.start_time AND n.time < b.end_time_exclusive;

CREATE OR REPLACE VIEW grid_validation_view AS
SELECT n.*
FROM grid_telemetry_normalized n
CROSS JOIN (SELECT start_time, end_time_exclusive FROM grid_split_bounds WHERE split_name = 'validation') b
WHERE n.time >= b.start_time AND n.time < b.end_time_exclusive;

CREATE OR REPLACE VIEW grid_test_view AS
SELECT n.*
FROM grid_telemetry_normalized n
CROSS JOIN (SELECT start_time, end_time_exclusive FROM grid_split_bounds WHERE split_name = 'test') b
WHERE n.time >= b.start_time AND n.time < b.end_time_exclusive;
