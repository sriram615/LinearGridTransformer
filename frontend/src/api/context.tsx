import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { apiClient, ApiClientError, DEFAULT_EXPERIMENT_ID } from './client'
import type { ApiHealth, DatasetInfo, EvaluationMetrics, ExperimentInfo, ModelInfo, TrainingHistory } from './types'

export type ConnectionStatus = 'checking' | 'connected' | 'unavailable'

interface AppDataContextValue {
  api: typeof apiClient
  experimentId: string
  setExperimentId: (experimentId: string) => void
  connectionStatus: ConnectionStatus
  health: ApiHealth | null
  connectionError: string | null
  refreshHealth: () => Promise<void>
  historical: HistoricalData
  historicalLoading: Record<HistoricalKey, boolean>
  historicalErrors: Partial<Record<HistoricalKey, string>>
  refreshHistorical: () => Promise<void>
}

export type HistoricalKey = 'dataset' | 'models' | 'experiment' | 'history' | 'metrics'
export interface HistoricalData {
  dataset: DatasetInfo | null
  models: ModelInfo[] | null
  experiment: ExperimentInfo | null
  history: TrainingHistory | null
  metrics: EvaluationMetrics | null
}

const emptyHistorical: HistoricalData = { dataset: null, models: null, experiment: null, history: null, metrics: null }
const historicalKeys: HistoricalKey[] = ['dataset', 'models', 'experiment', 'history', 'metrics']

const AppDataContext = createContext<AppDataContextValue | null>(null)

export function AppDataProvider({ children }: { children: ReactNode }) {
  const historicalRequestId = useRef(0)
  const [experimentId, setExperimentId] = useState(DEFAULT_EXPERIMENT_ID)
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('checking')
  const [health, setHealth] = useState<ApiHealth | null>(null)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [historical, setHistorical] = useState<HistoricalData>(emptyHistorical)
  const [historicalLoading, setHistoricalLoading] = useState<Record<HistoricalKey, boolean>>({
    dataset: true, models: true, experiment: true, history: true, metrics: true,
  })
  const [historicalErrors, setHistoricalErrors] = useState<Partial<Record<HistoricalKey, string>>>({})

  const refreshHealth = useCallback(async () => {
    setConnectionStatus('checking')
    try {
      const current = await apiClient.health()
      setHealth(current)
      setConnectionError(null)
      setConnectionStatus(
        current.status === 'ready' && current.inference_service_available && current.experiment_store_available
          ? 'connected'
          : 'unavailable',
      )
    } catch (error) {
      setHealth(null)
      setConnectionStatus('unavailable')
      setConnectionError(
        error instanceof ApiClientError ? error.message : 'The backend could not be reached.',
      )
    }
  }, [])

  const refreshHistorical = useCallback(async () => {
    const requestId = ++historicalRequestId.current
    setHistoricalLoading({ dataset: true, models: true, experiment: true, history: true, metrics: true })
    setHistoricalErrors({})
    setHistorical((current) => ({ ...current, experiment: null, history: null, metrics: null }))
    const results = await Promise.allSettled([
      apiClient.dataset(),
      apiClient.models(),
      apiClient.experiment(experimentId),
      apiClient.history(experimentId),
      apiClient.metrics(experimentId),
    ])
    if (requestId !== historicalRequestId.current) return
    const nextData: HistoricalData = { ...emptyHistorical }
    const nextErrors: Partial<Record<HistoricalKey, string>> = {}
    results.forEach((result, index) => {
      const key = historicalKeys[index]
      if (result.status === 'fulfilled') {
        Object.assign(nextData, { [key]: result.value })
      } else {
        nextErrors[key] = result.reason instanceof ApiClientError
          ? result.reason.message
          : 'This historical API resource could not be loaded.'
      }
    })
    setHistorical(nextData)
    setHistoricalErrors(nextErrors)
    setHistoricalLoading({ dataset: false, models: false, experiment: false, history: false, metrics: false })
  }, [experimentId])

  useEffect(() => {
    void refreshHealth()
    const timer = window.setInterval(() => void refreshHealth(), 30_000)
    return () => window.clearInterval(timer)
  }, [refreshHealth])

  useEffect(() => {
    void refreshHistorical()
  }, [refreshHistorical])

  const value = useMemo<AppDataContextValue>(() => ({
    api: apiClient,
    experimentId,
    setExperimentId,
    connectionStatus,
    health,
    connectionError,
    refreshHealth,
    historical,
    historicalLoading,
    historicalErrors,
    refreshHistorical,
  }), [connectionError, connectionStatus, experimentId, health, historical, historicalErrors, historicalLoading, refreshHealth, refreshHistorical])

  return <AppDataContext.Provider value={value}>{children}</AppDataContext.Provider>
}

export function useAppData(): AppDataContextValue {
  const value = useContext(AppDataContext)
  if (!value) throw new Error('useAppData must be used inside AppDataProvider.')
  return value
}
