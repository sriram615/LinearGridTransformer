"""Public response schemas for historical Tetouan API data."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator

from src.data.tetouan_preprocess import FEATURE_NAMES


class ApiResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class ModelInfo(ApiResponse):
    name: str
    input_shape: list[int]
    forecast_output_shape: list[int]
    configuration: dict[str, Any]
    selected_inference_device: str | None
    loaded: bool
    available: bool
    role: str = "forecasting_model"
    selected: bool = False


class SafeArtifactInfo(ApiResponse):
    name: str
    available: bool
    size_bytes: int | None = None


class DatasetInfo(ApiResponse):
    dataset_name: str
    source: dict[str, Any]
    row_count: int
    date_range: dict[str, Any]
    cadence_minutes: int | None
    timezone: str | None
    timezone_status: str | None
    features: dict[str, Any]
    targets: dict[str, Any]
    splits: dict[str, Any]
    windows: dict[str, Any]
    normalization: dict[str, Any]
    data_quality: dict[str, Any] | None
    coverage_uncertainties: dict[str, Any]


class ExperimentModelInfo(ApiResponse):
    configuration: dict[str, Any]
    checkpoint: SafeArtifactInfo
    best_validation_epoch: int | None
    best_validation_loss: float | None
    role: str = "forecasting_model"
    selected: bool = False
    checkpoint_seed: int | None = None
    seed_runs: list[dict[str, Any]] | None = None


class ExperimentInfo(ApiResponse):
    run_identifier: str
    run_type: str | None
    created_utc: str | None
    provenance: dict[str, Any]
    training_configuration: dict[str, Any] | None
    optimizer: str | None
    loss: str | None
    seed_policy: dict[str, Any] | None
    test_partition_used: bool | None
    models: dict[str, ExperimentModelInfo]
    artifact_references: dict[str, SafeArtifactInfo]


class TrainingHistory(ApiResponse):
    experiment_id: str
    models: dict[str, list[dict[str, Any]]]
    per_seed: dict[str, dict[str, list[dict[str, Any]]]] | None = None


class EvaluationMetrics(ApiResponse):
    experiment_id: str
    test_split: dict[str, Any]
    test_usage: dict[str, Any]
    overall: dict[str, Any]
    horizon_wise: dict[str, Any]


class ApiError(BaseModel):
    error: dict[str, str]


class HealthResponse(BaseModel):
    status: Literal["ready", "degraded"]
    inference_service_available: bool
    experiment_store_available: bool
    selected_device: str | None
    experiment_id: str


ObservationRow = Annotated[list[FiniteFloat], Field(min_length=8, max_length=8)]


class ForecastRequest(BaseModel):
    feature_names: Annotated[list[str], Field(min_length=8, max_length=8)]
    observations: Annotated[list[ObservationRow], Field(min_length=144, max_length=144)]
    timestamps: Annotated[list[str], Field(min_length=144, max_length=144)] | None = None

    @field_validator("feature_names")
    @classmethod
    def canonical_feature_order(cls, value: list[str]) -> list[str]:
        if tuple(value) != tuple(FEATURE_NAMES):
            raise ValueError("feature_names must match the canonical feature order.")
        return value


class ForecastStep(BaseModel):
    horizon_step: int
    minutes_ahead: int
    values: list[float]


class ForecastResponse(ApiResponse):
    experiment_id: str
    model_name: str
    input_dimensions: list[int]
    output_dimensions: list[int]
    target_zone_names: list[str]
    forecast_steps: list[ForecastStep]
    units_status: str


class ComparedModelForecast(ApiResponse):
    output_dimensions: list[int]
    forecast_steps: list[ForecastStep]


class ForecastComparisonResponse(ApiResponse):
    experiment_id: str
    input_dimensions: list[int]
    target_zone_names: list[str]
    models: dict[str, ComparedModelForecast]
    units_status: str
