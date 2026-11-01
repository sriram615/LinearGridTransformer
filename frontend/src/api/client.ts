import type {
  ApiErrorBody,
  ApiHealth,
  DatasetInfo,
  EvaluationMetrics,
  ExperimentInfo,
  ForecastComparisonResponse,
  ForecastRequest,
  ForecastResponse,
  ModelInfo,
  ModelName,
  TrainingHistory,
} from './types'

export const DEFAULT_EXPERIMENT_ID = 'extended_training_v2'
export const ORIGINAL_EXPERIMENT_ID = 'tetouan_t4_20261101_v1'

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '')

export class ApiClientError extends Error {
  readonly status: number
  readonly code: string

  constructor(status: number, code: string, message: string) {
    super(message)
    this.name = 'ApiClientError'
    this.status = status
    this.code = code
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...init?.headers },
    })
  } catch {
    throw new ApiClientError(0, 'network_unavailable', 'The backend could not be reached.')
  }

  const payload: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const error = payload as ApiErrorBody | null
    throw new ApiClientError(
      response.status,
      error?.error?.code ?? 'request_failed',
      error?.error?.message ?? 'The backend request failed.',
    )
  }
  return payload as T
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export const apiClient = {
  health: () => request<ApiHealth>('/health'),
  models: () => request<ModelInfo[]>('/api/v1/models'),
  dataset: () => request<DatasetInfo>('/api/v1/dataset'),
  experiment: (experimentId: string) =>
    request<ExperimentInfo>(`/api/v1/experiments/${encodeURIComponent(experimentId)}`),
  history: (experimentId: string) =>
    request<TrainingHistory>(`/api/v1/experiments/${encodeURIComponent(experimentId)}/history`),
  metrics: (experimentId: string) =>
    request<EvaluationMetrics>(`/api/v1/experiments/${encodeURIComponent(experimentId)}/metrics`),
  forecast: (modelName: ModelName, input: ForecastRequest) =>
    postJson<ForecastResponse>(`/api/v1/forecasts/${encodeURIComponent(modelName)}`, input),
  compareForecasts: (input: ForecastRequest) =>
    postJson<ForecastComparisonResponse>('/api/v1/forecasts/compare', input),
}

export type ApiClient = typeof apiClient
