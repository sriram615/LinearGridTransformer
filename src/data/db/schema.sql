-- ==============================================================================
-- LinearGridTransformer: Industry-Grade PostgreSQL / TimescaleDB Schema
-- Multi-Zone Grid Telemetry and Weather Storage
-- ==============================================================================

-- 1. Metadata: Transmission / Distribution Grid Zones
CREATE TABLE IF NOT EXISTS grid_zones (
    zone_id VARCHAR(32) PRIMARY KEY,
    name TEXT NOT NULL,
    iso_code VARCHAR(16) NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2. Telemetry: High-Resolution Load & Weather Telemetry (Hypertable)
CREATE TABLE IF NOT EXISTS grid_telemetry (
    time TIMESTAMPTZ NOT NULL,
    zone_id VARCHAR(32) NOT NULL REFERENCES grid_zones(zone_id),
    load_mw DOUBLE PRECISION NOT NULL,
    temperature_c DOUBLE PRECISION,
    relative_humidity_pct DOUBLE PRECISION,
    wind_speed_ms DOUBLE PRECISION,
    diffuse_solar_wm2 DOUBLE PRECISION,
    direct_solar_wm2 DOUBLE PRECISION,
    CONSTRAINT pk_grid_telemetry PRIMARY KEY (time, zone_id)
);

-- Index for fast sliding-window temporal range scans
CREATE INDEX IF NOT EXISTS idx_grid_telemetry_time ON grid_telemetry (time DESC);
CREATE INDEX IF NOT EXISTS idx_grid_telemetry_zone_time ON grid_telemetry (zone_id, time DESC);

-- 3. TimescaleDB Extension & Hypertable Setup (if TimescaleDB is installed)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb') THEN
        CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
        
        -- Partition by 1-month chunks
        PERFORM create_hypertable('grid_telemetry', 'time', 
            chunk_time_interval => INTERVAL '1 month', 
            if_not_exists => TRUE
        );

        -- Columnar compression: 90%+ disk reduction
        ALTER TABLE grid_telemetry SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'zone_id',
            timescaledb.compress_orderby = 'time DESC'
        );
        
        -- Automatically compress chunks older than 3 months
        PERFORM add_compression_policy('grid_telemetry', INTERVAL '3 months', if_not_exists => TRUE);
    END IF;
END $$;

-- 4. Initial Seed for 3 Major Transmission Zones
INSERT INTO grid_zones (zone_id, name, iso_code, latitude, longitude, timezone)
VALUES
    ('ZONE_1', 'New York City (N.Y.C.)', 'NYISO', 40.7128, -74.0060, 'America/New_York'),
    ('ZONE_2', 'Long Island (LONGIL)', 'NYISO', 40.7891, -73.1350, 'America/New_York'),
    ('ZONE_3', 'Hudson Valley (HUD VL)', 'NYISO', 41.7004, -73.9210, 'America/New_York')
ON CONFLICT (zone_id) DO NOTHING;

-- 5. Continuous Aggregate: 1-Hour Downsampled Operational View
CREATE OR REPLACE VIEW grid_telemetry_hourly AS
SELECT
    date_trunc('hour', time) AS hour,
    zone_id,
    AVG(load_mw) AS avg_load_mw,
    MAX(load_mw) AS peak_load_mw,
    MIN(load_mw) AS min_load_mw,
    AVG(temperature_c) AS avg_temp_c,
    AVG(relative_humidity_pct) AS avg_humidity_pct,
    AVG(wind_speed_ms) AS avg_wind_speed_ms,
    AVG(diffuse_solar_wm2) AS avg_diffuse_solar_wm2,
    AVG(direct_solar_wm2) AS avg_direct_solar_wm2
FROM grid_telemetry
GROUP BY date_trunc('hour', time), zone_id;
