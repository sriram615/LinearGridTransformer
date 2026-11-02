"""FastAPI lifecycle and historical Tetouan endpoints."""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any, Callable, Literal

import numpy as np
from fastapi import FastAPI, Request
import json
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.config import ApiConfig
from src.api.schemas import (
    DatasetInfo,
    DatasetListResponse,
    DatasetSummary,
    EvaluationMetrics,
    ExperimentInfo,
    ForecastComparisonResponse,
    ForecastRequest,
    ForecastResponse,
    GridTelemetryResponse,
    HealthResponse,
    ModelInfo,
    SafeArtifactInfo,
    TrainingHistory,
)
from src.data.db.postgres_manager import PostgresManager
from src.data.grid_experiment_store import GridExperimentStore
from src.data.tetouan_experiment_store import (
    MODEL_ORDER,
    InvalidArtifactError,
    MissingArtifactError,
    TetouanExperimentStore,
)
from src.data.tetouan_extended_experiment_store import (
    InvalidExtendedArtifactError,
    MissingExtendedArtifactError,
    TetouanExtendedExperimentStore,
)
from src.data.tetouan_preprocess import FEATURE_NAMES
from src.inference.grid_service import GridInferenceService
from src.inference.tetouan_service import MODEL_NAMES, TetouanInferenceService

GRID_EXPERIMENT_IDS: tuple[str, ...] = (
    "grid_colab_v1",
    "grid_colab_s42",
    "grid_colab_s123",
    "grid_colab_s2026",
)

logger = logging.getLogger(__name__)
StoreFactory = Callable[..., Any]
InferenceFactory = Callable[..., Any]


class PublicApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _json_safe(value: Any) -> Any:
    """Convert saved values to JSON types and remove filesystem path fields."""
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            lower_key = normalized_key.lower()
            if lower_key == "path" or lower_key.endswith("_path") or lower_key.endswith("_artifact"):
                continue
            cleaned[normalized_key] = _json_safe(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        if value.startswith(("https://", "http://")):
            return value
        if value.startswith("file://"):
            return Path(value[7:]).name
        if value.startswith("\\\\") or (len(value) > 2 and value[1:3] == ":\\"):
            return PureWindowsPath(value).name
        if os.path.isabs(value) or value.startswith("~/"):
            return Path(value).name
    return value


def _artifact_info(raw_path: str | Path) -> SafeArtifactInfo:
    path = Path(raw_path)
    available = path.is_file()
    return SafeArtifactInfo(
        name=path.name,
        available=available,
        size_bytes=path.stat().st_size if available else None,
    )


def _read_store(request: Request, method_name: str, store: Any | None = None) -> Any:
    store = store or request.app.state.experiment_store
    if store is None:
        raise PublicApiError(503, "service_unavailable", "Historical experiment data is unavailable.")
    try:
        return getattr(store, method_name)()
    except (MissingArtifactError, MissingExtendedArtifactError) as exc:
        logger.error("Historical artifact unavailable (%s)", type(exc).__name__)
        raise PublicApiError(503, "artifact_unavailable", "A required experiment artifact is unavailable.") from exc
    except (InvalidArtifactError, InvalidExtendedArtifactError) as exc:
        logger.error("Historical artifact validation failed (%s)", type(exc).__name__)
        raise PublicApiError(500, "invalid_artifact", "An experiment artifact is invalid.") from exc
    except Exception as exc:
        logger.error("Historical data request failed (%s)", type(exc).__name__)
        raise PublicApiError(500, "internal_error", "Internal server error.") from exc


def _safe_artifacts(raw_references: Mapping[str, str]) -> dict[str, SafeArtifactInfo]:
    return {
        key: _artifact_info(path)
        for key, path in raw_references.items()
    }


def _forecast_steps(values: Any) -> tuple[list[int], list[dict[str, Any]]]:
    prediction = np.asarray(values, dtype=np.float64)
    if prediction.shape != (6, 3) or not np.isfinite(prediction).all():
        raise PublicApiError(500, "inference_failed", "The model could not produce a valid forecast.")
    steps = [
        {
            "horizon_step": step + 1,
            "minutes_ahead": (step + 1) * 10,
            "values": prediction[step].tolist(),
        }
        for step in range(6)
    ]
    return list(prediction.shape), steps


def _inference_service(request: Request) -> Any:
    service = request.app.state.inference_service
    if service is None:
        raise PublicApiError(503, "service_unavailable", "Forecast inference is unavailable.")
    return service


def create_app(
    config: ApiConfig | None = None,
    *,
    store_factory: StoreFactory | None = None,
    extended_store_factory: StoreFactory | None = None,
    inference_factory: InferenceFactory | None = None,
    grid_store_factory: StoreFactory | None = None,
    grid_inference_factory: InferenceFactory | None = None,
) -> FastAPI:
    """Create an app; services are initialized once during each app lifespan."""
    settings = config or ApiConfig.from_env()
    build_store = store_factory or TetouanExperimentStore
    build_extended_store = extended_store_factory or (
        TetouanExtendedExperimentStore if store_factory is None else None
    )
    build_inference = inference_factory or TetouanInferenceService
    build_grid_store = grid_store_factory or GridExperimentStore
    build_grid_inference = grid_inference_factory or GridInferenceService

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.experiment_store = None
        app.state.latest_experiment_store = None
        app.state.experiment_store_available = False
        app.state.inference_service = None
        app.state.grid_store = None
        app.state.grid_inference_service = None
        app.state.grid_db_manager = None
        app.state.startup_error_type = None
        app.state.ready = False
        try:
            store = build_store(
                run_name=settings.run_name,
                artifact_dir=settings.artifact_dir,
                experiment_dir=settings.experiment_dir,
                checkpoint_dir=settings.checkpoint_dir,
                raw_dir=settings.raw_dir,
            )
            app.state.experiment_store = store
            verification = store.verify_artifacts()
            if not isinstance(verification, dict) or verification.get("status") != "ok":
                raise RuntimeError("Experiment artifact verification failed.")

            if build_extended_store is None:
                latest_store = store
            else:
                latest_store = build_extended_store(
                    experiment_dir=settings.latest_experiment_dir,
                    checkpoint_dir=settings.latest_checkpoint_dir,
                    evaluation_dir=settings.latest_evaluation_dir,
                    processed_dir=settings.artifact_dir,
                )
                latest_verification = latest_store.verify_artifacts()
                if not isinstance(latest_verification, dict) or latest_verification.get("status") != "ok":
                    raise RuntimeError("Latest experiment artifact verification failed.")
            app.state.latest_experiment_store = latest_store
            app.state.experiment_store_available = True
            latest_inference_run_name = settings.latest_inference_run_name
            if latest_inference_run_name is None:
                latest_inference_run_name = (
                    latest_store.get_inference_run_name()
                    if hasattr(latest_store, "get_inference_run_name")
                    else settings.run_name
                )

            inference = build_inference(
                artifact_dir=settings.artifact_dir,
                experiment_dir=settings.latest_experiment_dir,
                checkpoint_dir=settings.latest_checkpoint_dir,
                run_name=latest_inference_run_name,
                device=settings.device,
            )
            app.state.inference_service = inference

            # Grid store & inference initialization (graceful in isolated test environments)
            try:
                grid_store = build_grid_store(
                    results_dir=settings.grid_experiment_dir,
                    checkpoint_dir=settings.grid_checkpoint_dir,
                    processed_dir=settings.grid_artifact_dir,
                )
                grid_verification = grid_store.verify_artifacts()
                if isinstance(grid_verification, dict) and grid_verification.get("status") == "ok":
                    app.state.grid_store = grid_store
            except Exception as exc:
                logger.debug("Grid store initialization skipped: %s", exc)

            try:
                grid_inference = build_grid_inference(
                    processed_dir=settings.grid_artifact_dir,
                    checkpoint_dir=settings.grid_checkpoint_dir,
                    results_dir=settings.grid_experiment_dir,
                    run_name=settings.grid_run_name,
                    device=settings.device,
                )
                app.state.grid_inference_service = grid_inference
            except Exception as exc:
                logger.debug("Grid inference service initialization skipped: %s", exc)

            try:
                app.state.grid_db_manager = PostgresManager(connection_url=settings.grid_db_url)
            except Exception as exc:
                logger.debug("PostgresManager initialization skipped: %s", exc)

            app.state.ready = True
        except Exception as exc:  # startup stays available for sanitized health reporting
            app.state.startup_error_type = type(exc).__name__
            app.state.ready = False
            logger.error("API service initialization failed (%s)", type(exc).__name__)
        try:
            yield
        finally:
            app.state.ready = False
            app.state.experiment_store_available = False
            app.state.inference_service = None
            app.state.experiment_store = None
            app.state.latest_experiment_store = None
            app.state.grid_store = None
            app.state.grid_inference_service = None
            app.state.grid_db_manager = None

    app = FastAPI(title=settings.title, version=settings.version, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.state.config = settings
    app.state.experiment_store = None
    app.state.latest_experiment_store = None
    app.state.experiment_store_available = False
    app.state.inference_service = None
    app.state.grid_store = None
    app.state.grid_inference_service = None
    app.state.grid_db_manager = None
    app.state.startup_error_type = None
    app.state.ready = False

    @app.get("/")
    def project_info() -> dict[str, Any]:
        """Provide a small public index for the API without runtime details."""
        return {
            "name": "LinearGridTransformer",
            "description": "Multi-zone electricity consumption forecasting API.",
            "links": {
                "health": "/health",
                "docs": "/docs",
                "openapi": "/openapi.json",
                "api": "/api/v1",
            },
        }

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> JSONResponse:
        store_available = bool(request.app.state.experiment_store_available)
        inference = request.app.state.inference_service
        inference_available = inference is not None
        ready = bool(request.app.state.ready and store_available and inference_available)
        device = getattr(inference, "device", None) if inference_available else None
        payload = HealthResponse(
            status="ready" if ready else "degraded",
            inference_service_available=inference_available,
            experiment_store_available=store_available,
            selected_device=str(device) if device is not None else settings.device,
            experiment_id=settings.latest_experiment_id,
        ).model_dump()
        return JSONResponse(status_code=200 if ready else 503, content=payload)

    @app.get("/api/v1/datasets", response_model=DatasetListResponse)
    def list_datasets() -> dict[str, Any]:
        return {
            "active_dataset": settings.active_dataset,
            "datasets": [
                {
                    "id": "grid",
                    "name": "NYISO Regional Multi-Zone Transmission Grid",
                    "description": "New York Independent System Operator 2023 power consumption across 3 key zones with real Megawatts (MW) units.",
                    "physical_units": "Megawatts (MW)",
                    "zones": [
                        "New York City (N.Y.C.)",
                        "Long Island (LONGIL)",
                        "Hudson Valley (HUD VL)",
                    ],
                    "active": settings.active_dataset == "grid",
                },
                {
                    "id": "tetouan",
                    "name": "Power Consumption of Tetouan City",
                    "description": "Legacy 2017 Moroccan power distribution dataset across 3 distribution zones.",
                    "physical_units": "undocumented",
                    "zones": ["Zone 1", "Zone 2", "Zone 3"],
                    "active": settings.active_dataset == "tetouan",
                },
            ],
        }

    @app.get("/api/v1/grid/telemetry/latest", response_model=GridTelemetryResponse)
    def latest_telemetry(request: Request, limit: int = 144) -> dict[str, Any]:
        """Fetch recent sliding-window observations in real Megawatts (MW)."""
        limit = max(1, min(limit, 1000))
        # 1. Try TimescaleDB if manager is configured
        db_manager = request.app.state.grid_db_manager
        if db_manager is not None:
            try:
                df = db_manager.fetch_latest_telemetry(limit=limit)
                if not df.empty and len(df) == limit:
                    timestamps = [str(t) for t in df["time"]]
                    feat_cols = [c for c in df.columns if c != "time"]
                    observations = df[feat_cols].to_numpy().astype(np.float64).tolist()
                    return {
                        "timestamp_range": {
                            "start": timestamps[0],
                            "end": timestamps[-1],
                        },
                        "row_count": len(observations),
                        "feature_names": feat_cols,
                        "observations": observations,
                        "timestamps": timestamps,
                        "units": "Megawatts (MW)",
                    }
            except Exception as exc:
                logger.warning("TimescaleDB telemetry query failed, falling back to disk cache: %s", exc)

        # 2. Disk fallback from data/processed/grid/series.npz
        processed_file = settings.grid_artifact_dir / "series.npz"
        if not processed_file.is_file():
            raise PublicApiError(503, "telemetry_unavailable", "Live grid telemetry is currently unavailable.")
        try:
            data = np.load(processed_file, allow_pickle=True)
            series = data["series"]
            ts = data["timestamps"]
            obs = series[-limit:].astype(np.float64).tolist()
            timestamps = [str(t) for t in ts[-limit:]]
            metadata_file = settings.grid_artifact_dir / "metadata.json"
            meta = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.is_file() else {}
            feature_names = meta.get("features", {}).get("names", list(FEATURE_NAMES))
            return {
                "timestamp_range": {
                    "start": timestamps[0],
                    "end": timestamps[-1],
                },
                "row_count": len(obs),
                "feature_names": feature_names,
                "observations": obs,
                "timestamps": timestamps,
                "units": "Megawatts (MW)",
            }
        except Exception as exc:
            logger.error("Failed to read processed grid telemetry fallback: %s", exc)
            raise PublicApiError(500, "internal_error", "Failed to retrieve grid telemetry.") from exc

    @app.post("/api/v1/forecasts/compare", response_model=ForecastComparisonResponse)
    def compare_forecasts(
        body: ForecastRequest, request: Request, dataset: str | None = None
    ) -> dict[str, Any]:
        selected = dataset or ("tetouan" if store_factory is not None else settings.active_dataset)
        use_grid = selected == "grid" and request.app.state.grid_inference_service is not None
        service = request.app.state.grid_inference_service if use_grid else _inference_service(request)
        experiment_id = settings.grid_experiment_id if use_grid else settings.latest_experiment_id

        observations = np.asarray(body.observations, dtype=np.float64)
        try:
            predictions = service.predict_both(
                observations,
                feature_names=body.feature_names,
                timestamps=body.timestamps,
            )
        except (TypeError, ValueError) as exc:
            logger.info("Forecast request rejected (%s)", type(exc).__name__)
            raise PublicApiError(422, "invalid_forecast_input", "Forecast input is invalid.") from exc
        except Exception as exc:
            logger.error("Forecast comparison failed (%s)", type(exc).__name__)
            raise PublicApiError(500, "inference_failed", "Forecast inference failed.") from exc
        if not isinstance(predictions, Mapping) or tuple(predictions) != tuple(MODEL_NAMES):
            raise PublicApiError(500, "inference_failed", "The model comparison result is invalid.")
        models_result: dict[str, Any] = {}
        for name in MODEL_NAMES:
            output_dimensions, steps = _forecast_steps(predictions[name])
            models_result[name] = {
                "output_dimensions": output_dimensions,
                "forecast_steps": steps,
            }
        units_default = (
            "Physical units: Megawatts (MW) from NYISO telemetry."
            if use_grid
            else "Physical units are undocumented; values are original dataset numeric values."
        )
        return {
            "experiment_id": experiment_id,
            "input_dimensions": list(observations.shape),
            "target_zone_names": list(service.target_channel_names),
            "models": models_result,
            "units_status": service.metadata.get("units_status", units_default),
        }

    @app.post("/api/v1/forecasts/{model_name}", response_model=ForecastResponse)
    def forecast(
        model_name: str, body: ForecastRequest, request: Request, dataset: str | None = None
    ) -> dict[str, Any]:
        if model_name not in MODEL_NAMES:
            raise PublicApiError(404, "model_not_found", "The requested model was not found.")
        selected = dataset or ("tetouan" if store_factory is not None else settings.active_dataset)
        use_grid = selected == "grid" and request.app.state.grid_inference_service is not None
        service = request.app.state.grid_inference_service if use_grid else _inference_service(request)
        experiment_id = settings.grid_experiment_id if use_grid else settings.latest_experiment_id

        observations = np.asarray(body.observations, dtype=np.float64)
        try:
            prediction = service.predict(
                observations,
                model_name,
                feature_names=body.feature_names,
                timestamps=body.timestamps,
            )
        except (TypeError, ValueError) as exc:
            logger.info("Forecast request rejected (%s)", type(exc).__name__)
            raise PublicApiError(422, "invalid_forecast_input", "Forecast input is invalid.") from exc
        except Exception as exc:
            logger.error("Forecast inference failed (%s)", type(exc).__name__)
            raise PublicApiError(500, "inference_failed", "Forecast inference failed.") from exc
        output_dimensions, steps = _forecast_steps(prediction)
        units_default = (
            "Physical units: Megawatts (MW) from NYISO telemetry."
            if use_grid
            else "Physical units are undocumented; values are original dataset numeric values."
        )
        return {
            "experiment_id": experiment_id,
            "model_name": model_name,
            "input_dimensions": list(observations.shape),
            "output_dimensions": output_dimensions,
            "target_zone_names": list(service.target_channel_names),
            "forecast_steps": steps,
            "units_status": service.metadata.get("units_status", units_default),
        }

    @app.get("/api/v1/models", response_model=list[ModelInfo])
    def models(request: Request, dataset: str | None = None) -> list[dict[str, Any]]:
        selected = dataset or ("tetouan" if store_factory is not None else settings.active_dataset)
        if selected == "grid" and request.app.state.grid_inference_service is not None:
            inference = request.app.state.grid_inference_service
            grid_store = request.app.state.grid_store
            roles = grid_store.get_model_roles() if grid_store is not None else {}
            checkpoints = getattr(inference, "checkpoints", {})
            loaded_models = getattr(inference, "models", {})
            result = []
            for name in MODEL_ORDER:
                payload = checkpoints.get(name, {})
                configuration = payload.get("model_config", {})
                lookback = configuration.get("lookback", configuration.get("seq_len", 144))
                horizon = configuration.get("horizon", configuration.get("pred_len", 6))
                input_channels = configuration.get("in_channels", 8)
                output_channels = configuration.get("out_channels", 3)
                loaded = name in loaded_models and loaded_models[name] is not None
                role_info = roles.get(name, {})
                result.append({
                    "name": name,
                    "input_shape": [lookback, input_channels],
                    "forecast_output_shape": [horizon, output_channels],
                    "configuration": _json_safe(configuration),
                    "selected_inference_device": str(inference.device),
                    "loaded": loaded,
                    "available": loaded,
                    "role": role_info.get("role", "forecasting_model"),
                    "selected": role_info.get("selected", False),
                })
            return result

        inference = request.app.state.inference_service
        if inference is None:
            raise PublicApiError(503, "service_unavailable", "Inference models are unavailable.")
        result = []
        checkpoints = getattr(inference, "checkpoints", {})
        loaded_models = getattr(inference, "models", {})
        latest_store = request.app.state.latest_experiment_store
        roles = (
            latest_store.get_model_roles()
            if latest_store is not None and hasattr(latest_store, "get_model_roles")
            else {}
        )
        for name in MODEL_ORDER:
            payload = checkpoints.get(name, {})
            configuration = payload.get("model_config")
            if not isinstance(configuration, Mapping):
                raise PublicApiError(503, "model_unavailable", "A configured model is unavailable.")
            lookback = configuration.get("lookback", configuration.get("seq_len"))
            horizon = configuration.get("horizon", configuration.get("pred_len"))
            input_channels = configuration.get("in_channels")
            output_channels = configuration.get("out_channels")
            if not all(isinstance(value, int) and value > 0 for value in
                       (lookback, horizon, input_channels, output_channels)):
                raise PublicApiError(503, "model_unavailable", "A configured model has invalid dimensions.")
            loaded = name in loaded_models and loaded_models[name] is not None
            result.append({
                "name": name,
                "input_shape": [lookback, input_channels],
                "forecast_output_shape": [horizon, output_channels],
                "configuration": _json_safe(configuration),
                "selected_inference_device": str(inference.device),
                "loaded": loaded,
                "available": loaded,
                "role": roles.get(name, {}).get("role", "forecasting_model"),
                "selected": roles.get(name, {}).get("selected", False),
            })
        return result

    @app.get("/api/v1/dataset", response_model=DatasetInfo)
    def dataset(request: Request, dataset: str | None = None) -> dict[str, Any]:
        selected = dataset or ("tetouan" if store_factory is not None else settings.active_dataset)
        if selected == "grid" and request.app.state.grid_store is not None:
            return _json_safe(request.app.state.grid_store.get_dataset_info())
        return _json_safe(_read_store(request, "get_dataset_info"))

    @app.get("/api/v1/experiments/{experiment_id}", response_model=ExperimentInfo)
    def experiment(request: Request, experiment_id: str) -> dict[str, Any]:
        is_grid = (
            experiment_id == settings.grid_experiment_id
            or experiment_id == settings.grid_run_name
            or experiment_id in GRID_EXPERIMENT_IDS
        )
        if is_grid:
            store = request.app.state.grid_store
        elif experiment_id == settings.latest_experiment_id:
            store = request.app.state.latest_experiment_store
        elif experiment_id == settings.run_name:
            store = request.app.state.experiment_store
        else:
            raise PublicApiError(404, "experiment_not_found", "The requested experiment was not found.")
        if store is None:
            raise PublicApiError(503, "service_unavailable", "Historical experiment data is unavailable.")
        raw = _read_store(request, "get_experiment_info", store)
        model_records: dict[str, Any] = {}
        for name in MODEL_ORDER:
            record = raw["models"][name]
            checkpoint = record.get("checkpoint", {})
            checkpoint_path = checkpoint.get("path") or checkpoint.get("recorded_manifest_reference")
            safe_checkpoint = (
                checkpoint if "name" in checkpoint and "available" in checkpoint
                else _artifact_info(checkpoint_path or name).model_dump()
            )
            model_records[name] = {
                "configuration": _json_safe(record.get("configuration", {})),
                "checkpoint": safe_checkpoint,
                "best_validation_epoch": record.get("best_validation_epoch"),
                "best_validation_loss": record.get("best_validation_loss"),
                "role": record.get("role", "forecasting_model"),
                "selected": record.get("selected", False),
                "checkpoint_seed": record.get("checkpoint_seed"),
                "seed_runs": _json_safe(record.get("seed_runs")),
            }
        references = _safe_artifacts(raw.get("artifact_references", {}))
        return {
            "run_identifier": raw.get("run_identifier"),
            "run_type": raw.get("run_type"),
            "physical_units": raw.get("physical_units"),
            "created_utc": raw.get("created_utc"),
            "provenance": _json_safe(raw.get("provenance", {})),
            "training_configuration": _json_safe(raw.get("training_configuration")),
            "optimizer": raw.get("optimizer"),
            "loss": raw.get("loss"),
            "seed_policy": _json_safe(raw.get("seed_policy")),
            "test_partition_used": raw.get("test_partition_used"),
            "model_selection": _json_safe(raw.get("model_selection")),
            "baselines": _json_safe(raw.get("baselines")),
            "models": model_records,
            "artifact_references": {key: value.model_dump() for key, value in references.items()},
        }

    @app.get("/api/v1/experiments/{experiment_id}/history", response_model=TrainingHistory)
    def history(request: Request, experiment_id: str) -> dict[str, Any]:
        is_grid = (
            experiment_id == settings.grid_experiment_id
            or experiment_id == settings.grid_run_name
            or experiment_id in GRID_EXPERIMENT_IDS
        )
        if is_grid:
            store = request.app.state.grid_store
        elif experiment_id == settings.latest_experiment_id:
            store = request.app.state.latest_experiment_store
        elif experiment_id == settings.run_name:
            store = request.app.state.experiment_store
        else:
            raise PublicApiError(404, "experiment_not_found", "The requested experiment was not found.")
        if store is None:
            raise PublicApiError(503, "service_unavailable", "Historical experiment data is unavailable.")
        per_seed = store.get_seed_training_history() if hasattr(store, "get_seed_training_history") else None
        return {
            "experiment_id": experiment_id,
            "models": _json_safe(_read_store(request, "get_training_history", store)),
            "per_seed": _json_safe(per_seed),
        }

    @app.get("/api/v1/experiments/{experiment_id}/metrics", response_model=EvaluationMetrics)
    def metrics(request: Request, experiment_id: str) -> dict[str, Any]:
        is_grid = (
            experiment_id == settings.grid_experiment_id
            or experiment_id == settings.grid_run_name
            or experiment_id in GRID_EXPERIMENT_IDS
        )
        if is_grid:
            store = request.app.state.grid_store
        elif experiment_id == settings.latest_experiment_id:
            store = request.app.state.latest_experiment_store
        elif experiment_id == settings.run_name:
            store = request.app.state.experiment_store
        else:
            raise PublicApiError(404, "experiment_not_found", "The requested experiment was not found.")
        if store is None:
            raise PublicApiError(503, "service_unavailable", "Historical experiment data is unavailable.")
        return _json_safe(_read_store(request, "get_evaluation_info", store))

    @app.exception_handler(PublicApiError)
    async def public_api_error_handler(request: Request, exc: PublicApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": "Request validation failed."}},
        )

    @app.exception_handler(Exception)
    async def sanitized_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled API exception (%s)", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal server error."}},
        )

    return app


app = create_app()
