import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { apiClient, ApiClientError, DEFAULT_EXPERIMENT_ID, GRID_EXPERIMENT_ID } from './client'
import type {
  ApiHealth,
  DatasetId,
  DatasetInfo,
  DatasetSummary,
  EvaluationMetrics,
  ExperimentInfo,
  ModelInfo,
  TrainingHistory,
} from './types'

export type ConnectionStatus = 'checking' | 'connected' | 'unavailable'

interface AppDataContextValue {
  api: typeof apiClient
  datasetId: DatasetId
  setDatasetId: (datasetId: DatasetId) => void
  availableDatasets: DatasetSummary[]
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

const defaultDatasets: DatasetSummary[] = [
  {
    id: 'grid',
    name: 'NYISO Regional Multi-Zone Transmission Grid',
    description: 'New York Independent System Operator 2023 power consumption across 3 key zones with real Megawatts (MW) units.',
    physical_units: 'Megawatts (MW)',
    zones: ['New York City (N.Y.C.)', 'Long Island (LONGIL)', 'Hudson Valley (HUD VL)'],
    active: true,
  },
  {
    id: 'tetouan',
    name: 'Power Consumption of Tetouan City',
    description: 'Legacy 2017 Moroccan power distribution dataset across 3 distribution zones.',
    physical_units: 'undocumented',
    zones: ['Zone 1', 'Zone 2', 'Zone 3'],
    active: false,
  },
]

const emptyHistorical: HistoricalData = { dataset: null, models: null, experiment: null, history: null, metrics: null }
const historicalKeys: HistoricalKey[] = ['dataset', 'models', 'experiment', 'history', 'metrics']

const AppDataContext = createContext<AppDataContextValue | null>(null)

export function AppDataProvider({ children }: { children: ReactNode }) {
  const historicalRequestId = useRef(0)
  const [datasetId, setDatasetIdState] = useState<DatasetId>('grid')
  const [availableDatasets, setAvailableDatasets] = useState<DatasetSummary[]>(defaultDatasets)
  const [experimentId, setExperimentId] = useState(GRID_EXPERIMENT_ID)
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('checking')
  const [health, setHealth] = useState<ApiHealth | null>(null)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [historical, setHistorical] = useState<HistoricalData>(emptyHistorical)
  const [historicalLoading, setHistoricalLoading] = useState<Record<HistoricalKey, boolean>>({
    dataset: true, models: true, experiment: true, history: true, metrics: true,
  })
  const [historicalErrors, setHistoricalErrors] = useState<Partial<Record<HistoricalKey, string>>>({})

  const setDatasetId = useCallback((newId: DatasetId) => {
    setDatasetIdState(newId)
    setExperimentId(newId === 'grid' ? GRID_EXPERIMENT_ID : DEFAULT_EXPERIMENT_ID)
  }, [])

  const refreshHealth = useCallback(async () => {
    setConnectionStatus('checking')
    try {
      const [current, datasetList] = await Promise.allSettled([
        apiClient.health(),
        apiClient.datasets(),
      ])

      if (datasetList.status === 'fulfilled' && datasetList.value?.datasets?.length) {
        setAvailableDatasets(datasetList.value.datasets)
      }

      if (current.status === 'fulfilled') {
        const h = current.value
        setHealth(h)
        setConnectionError(null)
        setConnectionStatus(
          h.status === 'ready' && h.inference_service_available && h.experiment_store_available
            ? 'connected'
            : 'unavailable',
        )
      } else {
        setHealth(null)
        setConnectionStatus('unavailable')
        setConnectionError(
          current.reason instanceof ApiClientError ? current.reason.message : 'The backend could not be reached.',
        )
      }
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
      apiClient.dataset(datasetId),
      apiClient.models(datasetId),
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
  }, [datasetId, experimentId])

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
    datasetId,
    setDatasetId,
    availableDatasets,
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
  }), [availableDatasets, connectionError, connectionStatus, datasetId, experimentId, health, historical, historicalErrors, historicalLoading, refreshHealth, refreshHistorical, setDatasetId])

  return <AppDataContext.Provider value={value}>{children}</AppDataContext.Provider>
}

export function useAppData(): AppDataContextValue {
  const value = useContext(AppDataContext)
  if (!value) throw new Error('useAppData must be used inside AppDataProvider.')
  return value
}
