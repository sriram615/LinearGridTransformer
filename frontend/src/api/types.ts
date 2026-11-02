export const FEATURE_NAMES = [
  'zone_1_power_consumption',
  'zone_2_power_consumption',
  'zone_3_power_consumption',
  'temperature',
  'humidity',
  'wind_speed',
  'general_diffuse_flows',
  'diffuse_flows',
] as const

export const TARGET_NAMES = [
  'zone_1_power_consumption',
  'zone_2_power_consumption',
  'zone_3_power_consumption',
] as const

export type FeatureName = (typeof FEATURE_NAMES)[number]
export type TargetName = (typeof TARGET_NAMES)[number]
export type ModelName = 'LinearGridTransformer' | 'CDLinear'
export type BaselineName = 'training_mean' | 'persistence'
export type DatasetId = 'grid' | 'tetouan'

export interface DatasetSummary {
  id: DatasetId
  name: string
  description: string
  physical_units: string
  zones: string[]
  active: boolean
}

export interface DatasetListResponse {
  active_dataset: string
  datasets: DatasetSummary[]
}

export interface GridTelemetryResponse {
  timestamp_range: { start: string; end: string }
  row_count: number
  feature_names: string[]
  observations: number[][]
  timestamps: string[]
  units: string
}

export interface DieboldMarianoResult {
  test: string
  dm_statistic: number
  p_value: number
  conclusion?: string
  significance?: string
}

export interface ApiHealth {
  status: 'ready' | 'degraded'
  inference_service_available: boolean
  experiment_store_available: boolean
  selected_device: string | null
  experiment_id: string
}

export interface ModelInfo {
  name: ModelName
  input_shape: number[]
  forecast_output_shape: number[]
  configuration: Record<string, unknown>
  selected_inference_device: string | null
  loaded: boolean
  available: boolean
  role: string
  selected: boolean
}

export interface SafeArtifactInfo {
  name: string
  available: boolean
  size_bytes: number | null
}

export interface DatasetInfo {
  dataset_name: string
  source: Record<string, unknown>
  source_file_available?: boolean
  source_path?: string
  row_count: number
  date_range: { first: string | null; last: string | null }
  cadence_minutes: number | null
  timezone: string | null
  timezone_status: string | null
  physical_units?: string
  units?: string
  zones?: Record<string, string>
  features: { names: FeatureName[]; source_columns: string[] }
  targets: { names: TargetName[]; source_columns: string[] }
  splits: {
    method: string | null
    partitions: Record<string, {
      start_index: number
      end_index_exclusive: number
      row_count: number
      first_timestamp: string | null
      last_timestamp: string | null
    }>
  }
  windows: {
    lookback: number
    horizon: number
    stride: number
    input_contract: number[]
    target_contract: number[]
    validation_test_context_may_precede_partition: boolean
    targets_must_remain_in_partition: boolean
    future_weather_features: boolean
    samples?: Record<string, number>
  }
  normalization: {
    method: string
    fit_partition: string
    fit_row_range_half_open: number[]
    std_ddof: number
    feature_names: FeatureName[]
    feature_mean: number[]
    feature_std: number[]
    target_names: TargetName[]
    target_mean: number[]
    target_std: number[]
    inverse: string
  }
  data_quality: {
    available: boolean
    missing_values_per_column: number | null
    exact_duplicate_rows: number | null
    duplicate_timestamps: number | null
    timestamp_parse_failures: number | null
    irregular_intervals: number | null
    chronological_ordering: string | null
  } | null
  coverage_uncertainties: {
    missing_final_calendar_day: string | null
    handling: string | null
    units: string | null
    units_status: string | null
  }
}

export interface ExperimentModelInfo {
  configuration: Record<string, unknown>
  checkpoint: SafeArtifactInfo
  best_validation_epoch: number | null
  best_validation_loss: number | null
  role?: string
  selected?: boolean
  checkpoint_seed?: number | null
  validation_loss_mean?: number
  validation_loss_std?: number
  held_out_test_mae_mean?: number
  held_out_test_mae_std?: number
  held_out_test_rmse_mean?: number
  held_out_test_rmse_std?: number
  seed_runs?: {
    seed: number
    best_validation_epoch: number
    best_validation_loss: number
    test_mae?: number
    test_rmse?: number
    checkpoint?: SafeArtifactInfo
  }[] | null
}

export interface ExperimentInfo {
  run_identifier: string
  run_type: string | null
  dataset_name?: string
  physical_units?: string
  created_utc: string | null
  provenance: {
    git_commit?: string | null
    training_environment?: {
      python_version: string | null
      pytorch_version: string | null
      device: string | null
      cuda_available: boolean | null
      cuda_version: string | null
      gpu_name: string | null
    }
    source?: string
    database?: string
    verified_clean?: boolean
    [key: string]: unknown
  }
  training_configuration: Record<string, unknown> | null
  optimizer: string | null
  loss: string | null
  seed_policy: Record<string, unknown> | null
  test_partition_used: boolean | null
  models: Record<ModelName, ExperimentModelInfo>
  artifact_references?: Record<string, SafeArtifactInfo>
  model_selection?: {
    selected_model?: ModelName
    selection_basis?: string
    inference_checkpoint_seed?: number
    inference_checkpoint_selection_metric?: string
    validation_loss_advantage?: string
    econometric_significance?: DieboldMarianoResult
    [key: string]: unknown
  }
  baselines?: Record<string, { role?: string; trained?: boolean; test_mae?: number; test_rmse?: number; [key: string]: unknown }>
}

export interface TrainingEpoch {
  epoch: number
  train_loss: number
  validation_loss: number
  train_loss_std?: number
  validation_loss_std?: number
  seeds?: number[]
  train_batches?: number
  validation_batches?: number
  train_samples?: number
  validation_samples?: number
  optimizer_updates?: number
  is_best?: boolean
  best_epoch?: number
  best_validation_loss?: number
}

export interface TrainingHistory {
  experiment_id: string
  models: Record<ModelName, TrainingEpoch[]>
  per_seed?: Record<ModelName, Record<string, TrainingEpoch[]>> | null
}

export interface MetricPair {
  mae: number
  rmse: number
}

export interface ModelMetrics {
  per_zone: Record<string, MetricPair | ExtendedPerZoneMetrics>
  macro_average_across_zones?: MetricPair
  pooled_across_all_samples_horizons_and_zones?: MetricPair
  three_seed_summary?: ThreeSeedSummary
  per_seed?: SeedOverallMetric[]
}

export interface ThreeSeedSummary {
  model: ModelName
  MAE_mean: number
  MAE_std: number
  RMSE_mean: number
  RMSE_std: number
  macro_zone_MAE_mean?: number
  macro_zone_MAE_std?: number
}

export interface SeedOverallMetric {
  model: ModelName
  seed: number
  MAE: number
  RMSE: number
  macro_zone_MAE?: number
}

export interface SeedZoneMetric {
  model: ModelName | 'Persistence'
  seed: number | 'baseline'
  zone: string
  MAE: number
  RMSE: number
}

export interface ExtendedPerZoneMetrics {
  per_seed: SeedZoneMetric[]
  mae_mean: number
  rmse_mean: number
}

export interface HorizonMetric {
  horizon_step: number
  horizon_minutes: number
  per_zone: Record<TargetName, MetricPair>
  macro_average_across_zones: MetricPair
  pooled_across_test_windows_and_zones: { rmse: number }
}

export interface SeedHorizonMetric {
  model: ModelName | 'Persistence'
  seed: number | 'baseline'
  horizon_step: number
  minutes_ahead: number
  MAE: number
  RMSE: number
}

export interface EvaluationMetrics {
  experiment_id: string
  test_split: Record<string, unknown>
  test_usage: Record<string, boolean | string>
  physical_units?: string
  overall: {
    aggregation_note?: string | null
    models: Record<ModelName, ModelMetrics>
    baselines: Record<string, ModelMetrics | ExtendedBaselineMetrics>
  }
  horizon_wise: {
    forecast_horizon_count: number
    target_zone_names: (TargetName | string)[]
    metric_definitions: Record<string, string>
    models: Record<ModelName, (HorizonMetric | SeedHorizonMetric)[] | Record<string, MetricPair>>
    baselines?: Record<string, (HorizonMetric | SeedHorizonMetric)[] | ExtendedBaselineMetrics | Record<string, MetricPair>>
    consistency_with_overall?: Record<string, unknown>
  }
  overall_macro_averages?: Record<string, {
    mae_mean?: number
    mae_std?: number
    rmse_mean?: number
    rmse_std?: number
    mae?: number
    rmse?: number
    [key: string]: unknown
  }>
  per_zone_summary?: Record<string, Record<string, MetricPair>>
  econometric_significance?: DieboldMarianoResult
}

export interface ExtendedBaselineMetrics {
  name: string
  role: string
  trained: false
  seed: 'baseline'
  MAE: number
  RMSE: number
  macro_zone_MAE?: number | null
  per_zone: Record<string, SeedZoneMetric | { per_seed: SeedZoneMetric[]; mae_mean: number; rmse_mean: number }>
  per_horizon: SeedHorizonMetric[]
  macro_average_across_zones?: MetricPair
  pooled_across_all_samples_horizons_and_zones?: MetricPair
}

export interface ForecastRequest {
  feature_names: FeatureName[]
  observations: number[][]
  timestamps?: string[]
}

export interface ForecastStep {
  horizon_step: number
  minutes_ahead: number
  values: number[]
}

export interface ForecastResponse {
  experiment_id: string
  model_name: ModelName
  input_dimensions: number[]
  output_dimensions: number[]
  target_zone_names: (TargetName | string)[]
  forecast_steps: ForecastStep[]
  units_status?: string
  physical_units?: string
}

export interface ComparedModelForecast {
  output_dimensions: number[]
  forecast_steps: ForecastStep[]
}

export interface ForecastComparisonResponse {
  experiment_id: string
  input_dimensions: number[]
  target_zone_names: (TargetName | string)[]
  models: Record<ModelName, ComparedModelForecast>
  units_status?: string
  physical_units?: string
}

export interface ApiErrorBody {
  error: { code: string; message: string }
}
