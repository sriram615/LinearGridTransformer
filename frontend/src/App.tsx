import { useEffect, useMemo, useRef, useState } from 'react'
import { AppDataProvider, useAppData } from './api/context'
import { ApiStatus } from './components/ApiStatus'
import type { HistoricalKey } from './api/context'
import type { DatasetInfo, EvaluationMetrics, ForecastRequest, ForecastResponse, ForecastStep, ForecastComparisonResponse, HorizonMetric, ModelInfo, ModelMetrics, ModelName, SeedHorizonMetric, TrainingEpoch, TargetName } from './api/types'
import { ApiClientError, DEFAULT_EXPERIMENT_ID, ORIGINAL_EXPERIMENT_ID } from './api/client'
import { CsvValidationError, parseForecastCsv, toForecastRequest, type ParsedForecastCsv } from './api/forecastCsv'

type Page = 'overview' | 'forecast' | 'models' | 'dataset' | 'experiment'
type IconName =
  | 'grid'
  | 'spark'
  | 'cube'
  | 'database'
  | 'file'
  | 'chevron'
  | 'activity'
  | 'settings'
  | 'upload'
  | 'check'
  | 'alert'
  | 'arrow'
  | 'menu'
  | 'close'
  | 'info'

const navItems: { id: Page; label: string; icon: IconName }[] = [
  { id: 'overview', label: 'Overview', icon: 'grid' },
  { id: 'forecast', label: 'Forecast Explorer', icon: 'spark' },
  { id: 'models', label: 'Model Laboratory', icon: 'cube' },
  { id: 'dataset', label: 'Dataset Explorer', icon: 'database' },
  { id: 'experiment', label: 'Experiment Details', icon: 'file' },
]

const pageTitles: Record<Page, string> = {
  overview: 'Overview',
  forecast: 'Forecast Explorer',
  models: 'Model Laboratory',
  dataset: 'Dataset Explorer',
  experiment: 'Experiment Details',
}

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, React.ReactNode> = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    spark: <><path d="M4 15l4-5 4 3 6-8" /><path d="M14 5h4v4" /></>,
    cube: <><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z" /><path d="M4 7.5l8 4.5 8-4.5M12 21v-9" /></>,
    database: <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" /></>,
    file: <><path d="M6 3h8l4 4v14H6z" /><path d="M14 3v5h5M9 13h6M9 17h6" /></>,
    chevron: <path d="M9 18l6-6-6-6" />,
    activity: <path d="M3 12h4l2-6 4 12 2-6h6" />,
    settings: <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 00.3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 00-1.9-.3 1.7 1.7 0 00-1 1.6v.2h-4v-.2a1.7 1.7 0 00-1-1.6 1.7 1.7 0 00-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 00.3-1.9A1.7 1.7 0 003 14H3v-4h.1a1.7 1.7 0 001.5-1 1.7 1.7 0 00-.3-1.9L4.2 7 7 4.2l.1.1A1.7 1.7 0 009 4.6a1.7 1.7 0 001-1.5V3h4v.1a1.7 1.7 0 001 1.5 1.7 1.7 0 001.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 00-.3 1.9 1.7 1.7 0 001.5 1H21v4h-.1a1.7 1.7 0 00-1.5 1z" /></>,
    upload: <><path d="M12 16V4M7 9l5-5 5 5M5 20h14" /></>,
    check: <path d="M5 12l4 4L19 6" />,
    alert: <><path d="M12 3L2.8 20h18.4L12 3z" /><path d="M12 9v4M12 17h.01" /></>,
    arrow: <><path d="M5 12h14M14 7l5 5-5 5" /></>,
    menu: <path d="M4 7h16M4 12h16M4 17h16" />,
    close: <path d="M6 6l12 12M18 6L6 18" />,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7h.01" /></>,
  }
  return <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

function Button({ children, variant = 'primary', icon, onClick, disabled, className = '' }: { children: React.ReactNode; variant?: 'primary' | 'secondary' | 'ghost'; icon?: IconName; onClick?: () => void; disabled?: boolean; className?: string }) {
  return <button className={`button button-${variant} ${className}`} onClick={onClick} disabled={disabled}>{icon && <Icon name={icon} size={16} />}<span>{children}</span></button>
}

function Badge({ children, tone = 'neutral' }: { children: React.ReactNode; tone?: 'neutral' | 'blue' | 'green' | 'amber' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

function Card({ children, className = '', role }: { children: React.ReactNode; className?: string; role?: string }) {
  return <section className={`card ${className}`} role={role}>{children}</section>
}

function SectionHeader({ eyebrow, title, action }: { eyebrow?: string; title: string; action?: React.ReactNode }) {
  return <div className="section-header"><div>{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2>{title}</h2></div>{action}</div>
}

const MODEL_ORDER: ModelName[] = ['LinearGridTransformer', 'CDLinear']
const ZONE_COLORS = ['#4c8dff', '#d5dfed', '#43cf91']
const MODEL_COLORS = ['#4c8dff', '#80b4ff', '#d5dfed', '#aab5c4']
const TARGET_ZONE_NAMES: TargetName[] = ['zone_1_power_consumption', 'zone_2_power_consumption', 'zone_3_power_consumption']
const EXPERIMENT_OPTIONS = [
  { id: DEFAULT_EXPERIMENT_ID, label: 'Latest · extended_training_v2' },
  { id: ORIGINAL_EXPERIMENT_ID, label: 'Original · tetouan_t4_20261101_v1' },
] as const

function formatNumber(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? 'Not available'
    : value.toLocaleString(undefined, { maximumFractionDigits: digits })
}

function formatDate(value: string | null | undefined): string {
  return value ?? 'Not available'
}

function humanize(value: string): string {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (letter: string) => letter.toUpperCase())
}

function latestSummary(metrics: ModelMetrics | undefined) {
  return metrics?.three_seed_summary
}

function metricMean(metrics: EvaluationMetrics | null, model: ModelName, metric: 'mae' | 'rmse'): number | undefined {
  const record = metrics?.overall.models[model]
  const summary = latestSummary(record)
  if (summary) return metric === 'mae' ? summary.MAE_mean : summary.RMSE_mean
  return record?.macro_average_across_zones?.[metric]
}

function metricStd(metrics: EvaluationMetrics | null, model: ModelName, metric: 'mae' | 'rmse'): number | undefined {
  const summary = latestSummary(metrics?.overall.models[model])
  return summary ? (metric === 'mae' ? summary.MAE_std : summary.RMSE_std) : undefined
}

function perZoneMetric(metrics: EvaluationMetrics | null, model: ModelName, zone: string, metric: 'mae' | 'rmse'): number | undefined {
  const modelMetrics = metrics?.overall.models[model]
  if (!modelMetrics) return undefined
  const match = /^zone_([1-3])(?:_power_consumption)?$/.exec(zone)
  const latestZone = match ? `Zone ${match[1]}` : zone
  const value = modelMetrics.per_zone[latestZone] ?? modelMetrics.per_zone[zone]
  if (!value) return undefined
  if ('mae_mean' in value) return metric === 'mae' ? value.mae_mean : value.rmse_mean
  return value[metric]
}

function latestHorizonRows(metrics: EvaluationMetrics | null, model: ModelName): SeedHorizonMetric[] | null {
  const rows = metrics?.horizon_wise.models[model] ?? []
  return rows.length > 0 && 'minutes_ahead' in rows[0] ? rows as SeedHorizonMetric[] : null
}

function groupedHorizonValues(rows: SeedHorizonMetric[], metric: 'MAE' | 'RMSE') {
  const steps = [...new Set(rows.map((row) => row.horizon_step))].sort((a, b) => a - b)
  return steps.map((step) => {
    const stepRows = rows.filter((row) => row.horizon_step === step)
    const values = stepRows.map((row) => row[metric])
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length
    const std = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length)
    return { step, minutes: stepRows[0]?.minutes_ahead ?? step * 10, mean, std, perSeed: stepRows }
  })
}

function horizonDisplayRows(metrics: EvaluationMetrics | null, model: ModelName, metric: 'mae' | 'rmse', zones: TargetName[], zoneLabels: string[]) {
  const latestRows = latestHorizonRows(metrics, model)
  if (latestRows) {
    return groupedHorizonValues(latestRows, metric === 'mae' ? 'MAE' : 'RMSE').map((row) => ({
      key: `${model}-${row.step}`,
      lead: `${row.minutes} min · saved zone aggregation`,
      mean: row.mean,
      std: row.std,
      seeds: row.perSeed,
    }))
  }
  const oldRows = (metrics?.horizon_wise.models[model] ?? []).filter((row): row is HorizonMetric => 'horizon_minutes' in row)
  return oldRows.flatMap((row) => zones.map((zone, index) => ({
    key: `${model}-${row.horizon_step}-${zone}`,
    lead: `${row.horizon_minutes} min · ${zoneLabels[index] ?? humanize(zone)}`,
    mean: row.per_zone[zone]?.[metric],
    std: undefined,
    seeds: [],
  })))
}

function isRoleRecorded(role: string | undefined): boolean {
  return Boolean(role && role !== 'forecasting_model')
}

function modelInfoRole(name: ModelName, experiment: NonNullable<ReturnType<typeof useAppData>['historical']['experiment']>, models: ModelInfo[] | null, experimentId: string): string {
  if (experimentId === DEFAULT_EXPERIMENT_ID) return models?.find((model) => model.name === name)?.role ?? 'Role unavailable'
  const role = experiment?.models[name]?.role
  return isRoleRecorded(role) ? role! : 'Role not recorded'
}

function DataLineChart({ series, labels, height = 'medium', ariaLabel }: {
  series: { name: string; values: number[]; color: string }[]
  labels: string[]
  height?: 'small' | 'medium' | 'large'
  ariaLabel: string
}) {
  const allValues = series.flatMap((item) => item.values).filter(Number.isFinite)
  const min = Math.min(...allValues, 0)
  const max = Math.max(...allValues, 0)
  const range = max - min || 1
  const paths = series.map((item) => {
    const points = item.values.map((value, index) => {
      const x = item.values.length < 2 ? 240 : (index / (item.values.length - 1)) * 480
      const y = 112 - ((value - min) / range) * 96
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`
    })
    return { ...item, path: points.join(' ') }
  })
  return <>
    <div className={`chart chart-${height}`}>
      {allValues.length === 0
        ? <div className="chart-empty">No recorded values are available.</div>
        : <svg viewBox="0 0 480 130" preserveAspectRatio="none" role="img" aria-label={ariaLabel}>
          {[16, 48, 80, 112].map((y) => <line key={y} x1="0" y1={y} x2="480" y2={y} className="gridline" />)}
          {paths.map((item) => <path key={item.name} d={item.path} className="line" style={{ stroke: item.color }} />)}
        </svg>}
      <div className="axis-labels">{labels.map((label, index) => <span key={`${index}-${label}`}>{label}</span>)}</div>
    </div>
    {series.length > 0 && <div className="legend">{series.map((item) => <span key={item.name}><i className="legend-line" style={{ borderColor: item.color }} />{item.name}</span>)}</div>}
  </>
}

function MetricValue({ value }: { value: number | null | undefined }) {
  return <>{formatNumber(value)}</>
}

function MeanAndStd({ mean, std }: { mean?: number; std?: number }) {
  return <>{mean == null ? 'Not available' : `${formatNumber(mean)}${std == null ? '' : ` ± ${formatNumber(std)}`}`}</>
}

function baselineMetric(metrics: EvaluationMetrics | null, name: string, metric: 'mae' | 'rmse'): number | undefined {
  const baseline = metrics?.overall.baselines[name]
  if (!baseline) return undefined
  if ('MAE' in baseline) return metric === 'mae' ? baseline.MAE : baseline.RMSE
  return baseline.macro_average_across_zones?.[metric]
}

function HistoricalStatus({ resource }: { resource: HistoricalKey }) {
  const { historical, historicalLoading, historicalErrors, refreshHistorical } = useAppData()
  if (historical[resource] && !historicalErrors[resource]) return null
  const error = historicalErrors[resource]
  return <div className={`data-binding-state ${error ? 'data-binding-error' : ''}`} role="status" aria-live="polite">
    <span className="data-binding-indicator" />
    <span>{error ? `Could not load ${humanize(resource)}: ${error}` : historicalLoading[resource] ? `Loading ${humanize(resource)}…` : `${humanize(resource)} is not available.`}</span>
    {error && <button className="text-action" onClick={() => void refreshHistorical()}>Retry</button>}
  </div>
}

function Kpi({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <Card className="kpi"><span className="kpi-label">{label}</span><strong>{value}</strong><span className="kpi-detail">{detail}</span></Card>
}

function Overview({ goTo }: { goTo: (page: Page) => void }) {
  const { experimentId, historical } = useAppData()
  const dataset = historical.dataset
  const metrics = historical.metrics
  const experiment = historical.experiment
  const history = historical.history
  const cadence = dataset?.cadence_minutes
  const horizon = dataset?.windows.horizon
  const lookback = dataset?.windows.lookback
  const selectedModel = experimentId === DEFAULT_EXPERIMENT_ID
    ? (experiment?.model_selection?.selected_model ?? historical.models?.find((model) => model.selected)?.name ?? null)
    : experiment?.model_selection?.selected_model ?? null
  const bestModel = (name: ModelName) => experiment?.models[name]
  const zoneNames = dataset?.targets.source_columns ?? dataset?.targets.names ?? []
  const zoneMax = Math.max(1, ...MODEL_ORDER.flatMap((name) => zoneNames.map((_, index) => {
    const target = dataset?.targets.names[index]
    return target ? perZoneMetric(metrics, name, target, 'mae') ?? 0 : 0
  })))
  const trainingLosses = MODEL_ORDER.flatMap((name, modelIndex) => {
    const epochs = history?.models[name] ?? []
    return [
      { name: `${name} train`, values: epochs.map((row) => row.train_loss), color: MODEL_COLORS[modelIndex * 2] },
      { name: `${name} validation`, values: epochs.map((row) => row.validation_loss), color: MODEL_COLORS[modelIndex * 2 + 1] },
    ]
  })
  const latestHorizons = latestHorizonRows(metrics, 'CDLinear')
  const legacyHorizons = (metrics?.horizon_wise.models.CDLinear ?? []).filter((row): row is HorizonMetric => 'horizon_minutes' in row)
  const horizonSeries = latestHorizons
    ? [{ name: 'Three-seed mean MAE', values: groupedHorizonValues(latestHorizons, 'MAE').map((row) => row.mean), color: ZONE_COLORS[0] }]
    : zoneNames.map((zone, index) => {
      const target = dataset?.targets.names[index]
      return {
        name: zone,
        values: legacyHorizons.map((row) => target ? row.per_zone[target]?.mae ?? Number.NaN : Number.NaN),
        color: ZONE_COLORS[index % ZONE_COLORS.length],
      }
    })
  const horizonLabels = latestHorizons
    ? groupedHorizonValues(latestHorizons, 'MAE').map((row) => `${row.minutes}m`)
    : legacyHorizons.map((row) => `${row.horizon_minutes}m`)
  return <div className="page">
    <HistoricalStatus resource="dataset" /><HistoricalStatus resource="metrics" /><HistoricalStatus resource="experiment" /><HistoricalStatus resource="history" />
    <div className="hero">
      <div className="hero-copy">
        <Badge tone="blue">Research workspace</Badge>
        <h1>LinearGrid<span>Transformer</span></h1>
        <p>Multi-Zone Energy Consumption Forecasting</p>
        <div className="hero-meta"><span><Icon name="database" size={15} /> {dataset?.dataset_name ?? 'Dataset metadata unavailable'}</span><span><Icon name="activity" size={15} /> {experiment?.run_identifier ?? experimentId}</span></div>
      </div>
      <div className="hero-mark" aria-hidden="true"><div className="mark-grid">{Array.from({ length: 16 }).map((_, i) => <i key={i} className={i === 5 || i === 6 || i === 10 ? 'active' : ''} />)}</div><span>{dataset?.targets.names.length ?? '—'} zones · {horizon ?? '—'} steps</span></div>
    </div>
    <div className="kpi-grid">
      <Kpi label="Dataset observations" value={dataset ? formatNumber(dataset.row_count, 0) : 'Not available'} detail={dataset ? `${formatDate(dataset.date_range.first)} – ${formatDate(dataset.date_range.last)}` : 'Dataset API'} />
      <Kpi label="Input features" value={dataset ? formatNumber(dataset.features.names.length, 0) : 'Not available'} detail="Historical channels" />
      <Kpi label="Forecast zones" value={dataset ? formatNumber(dataset.targets.names.length, 0) : 'Not available'} detail="Target channels" />
      <Kpi label="Forecast horizon" value={horizon != null && cadence != null ? `${formatNumber(horizon * cadence, 0)} min` : 'Not available'} detail={`${horizon ?? '—'} steps · ${cadence ?? '—'} min cadence · ${lookback ?? '—'} input steps`} />
    </div>
    <div className="grid-two">
      <Card className="chart-card">
        <SectionHeader eyebrow="Held-out test set · saved evaluation" title="Model performance" action={<Badge tone={selectedModel ? 'green' : 'neutral'}>{selectedModel ? `${selectedModel} selected by backend` : 'Selection not recorded'}</Badge>} />
        <div className="metric-table">
          <div className="metric-row metric-head"><span>Metric</span><span>LinearGridTransformer</span><span>CDLinear</span></div>
          {(['mae', 'rmse'] as const).map((metric) => <div className="metric-row" key={metric}><span>{latestSummary(metrics?.overall.models.CDLinear) ? `Three-seed mean ${metric.toUpperCase()} ± std` : `Macro ${metric.toUpperCase()}`}</span><strong><MeanAndStd mean={metricMean(metrics, 'LinearGridTransformer', metric)} std={metricStd(metrics, 'LinearGridTransformer', metric)} /></strong><strong className={selectedModel === 'CDLinear' ? 'best' : ''}><MeanAndStd mean={metricMean(metrics, 'CDLinear', metric)} std={metricStd(metrics, 'CDLinear', metric)} /></strong></div>)}
          {metrics?.overall.models.CDLinear.pooled_across_all_samples_horizons_and_zones && <div className="metric-row"><span>Pooled RMSE</span><strong><MetricValue value={metrics.overall.models.LinearGridTransformer.pooled_across_all_samples_horizons_and_zones?.rmse} /></strong><strong><MetricValue value={metrics.overall.models.CDLinear.pooled_across_all_samples_horizons_and_zones.rmse} /></strong></div>}
        </div>
        <p className="muted-note">{latestSummary(metrics?.overall.models.CDLinear) ? 'Model values are means ± population standard deviations across three independent training seeds. Persistence is a single, untrained evaluation baseline.' : 'Macro metrics average the three zone metrics. Pooled metrics combine test windows, horizons, and zones.'} Values are original dataset numeric units; physical units are undocumented.</p>
        {metrics?.overall.baselines.persistence && <p className="muted-note">Persistence evaluation baseline · MAE <MetricValue value={baselineMetric(metrics, 'persistence', 'mae')} /> · RMSE <MetricValue value={baselineMetric(metrics, 'persistence', 'rmse')} />. It is not a trained model.</p>}
      </Card>
      <Card className="insight-card">
        <span className="insight-icon"><Icon name="spark" /></span>
        <span className="eyebrow">Saved validation records</span>
        <h2>Best checkpoint selection</h2>
        {MODEL_ORDER.map((name) => <p key={name}>{name}: epoch {bestModel(name)?.best_validation_epoch ?? 'not available'}, validation loss {formatNumber(bestModel(name)?.best_validation_loss, 6)}.</p>)}
        <p>{selectedModel ? `${selectedModel} is marked selected by backend metadata based on the saved held-out evaluation. The decision is not recalculated in the frontend.` : 'The selected model is not recorded for this experiment.'}</p>
        <Button variant="secondary" onClick={() => goTo('models')}>Open model laboratory <Icon name="arrow" size={15} /></Button>
      </Card>
    </div>
    <div className="grid-two">
      <Card className="chart-card"><SectionHeader title="CDLinear horizon-wise test MAE" action={<span className="chart-note">{latestHorizons ? 'Mean across three seed runs' : 'Saved horizon evaluation'}</span>} /><DataLineChart series={horizonSeries} labels={horizonLabels} ariaLabel="Saved CDLinear test MAE by forecast horizon" /></Card>
      <Card className="chart-card"><SectionHeader title="Training & validation loss" action={<span className="chart-note">{history?.models.CDLinear?.[0]?.seeds ? 'Three-seed mean · saved epoch history' : 'Saved epoch history'}</span>} /><DataLineChart series={trainingLosses} labels={(history?.models.LinearGridTransformer ?? history?.models.CDLinear ?? []).map((row) => String(row.epoch))} ariaLabel="Saved training and validation loss histories for both models" /></Card>
    </div>
    <Card className="zone-card">
      <SectionHeader title="Per-zone MAE comparison" action={<Button variant="ghost" onClick={() => goTo('models')}>View details <Icon name="arrow" size={14} /></Button>} />
      <div className="zone-bars">
        {zoneNames.map((label, index) => {
          const target = dataset?.targets.names[index]
          const a = target ? perZoneMetric(metrics, 'LinearGridTransformer', target, 'mae') : undefined
          const b = target ? perZoneMetric(metrics, 'CDLinear', target, 'mae') : undefined
          return <div className="zone-group" key={label}><span>{label}</span><div><i className="zone-bar blue-bar" title={`LinearGridTransformer MAE ${formatNumber(a)}`} style={{ height: `${(a ?? 0) / zoneMax * 85}%` }} /><i className="zone-bar white-bar" title={`CDLinear MAE ${formatNumber(b)}`} style={{ height: `${(b ?? 0) / zoneMax * 85}%` }} /></div><small>{formatNumber(a)} / {formatNumber(b)}</small></div>
        })}
      </div>
      <div className="legend"><span><i className="legend-line blue" />LinearGridTransformer</span><span><i className="legend-line white" />CDLinear</span></div>
    </Card>
  </div>
}

const features = ['Zone 1 consumption', 'Zone 2 consumption', 'Zone 3 consumption', 'Temperature', 'Humidity', 'Wind speed', 'General diffuse flows', 'Diffuse flows']

function Forecast() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [model, setModel] = useState<'lgt' | 'cd' | 'both' | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const [validationState, setValidationState] = useState<'empty' | 'selected' | 'validating' | 'invalid' | 'ready'>('empty')
  const [validationError, setValidationError] = useState<string | null>(null)
  const [parsedInput, setParsedInput] = useState<ParsedForecastCsv | null>(null)
  const [forecastState, setForecastState] = useState<'idle' | 'loading' | 'completed' | 'failed' | 'unavailable'>('idle')
  const [forecastError, setForecastError] = useState<string | null>(null)
  const [forecastResults, setForecastResults] = useState<ForecastDisplay[]>([])
  const submitLock = useRef(false)
  const { api, connectionStatus, health, historical, refreshHealth } = useAppData()
  const availableModels = historical.models?.filter((item) => item.available) ?? []
  const canCompare = MODEL_ORDER.every((name) => availableModels.some((item) => item.name === name))
  const inferenceExperimentId = health?.experiment_id ?? DEFAULT_EXPERIMENT_ID

  useEffect(() => {
    if (model !== null || !historical.models?.length) return
    const recommended = historical.models.find((item) => item.selected && item.available)
      ?? historical.models.find((item) => item.available)
    if (recommended) setModel(recommended.name === 'CDLinear' ? 'cd' : 'lgt')
  }, [historical.models, model])

  const clearFile = () => {
    if (inputRef.current) inputRef.current.value = ''
    setFileName(null)
    setParsedInput(null)
    setValidationError(null)
    setValidationState('empty')
    setForecastResults([])
    setForecastError(null)
    setForecastState('idle')
  }

  const handleFile = async (file?: File) => {
    if (!file) return
    setFileName(file.name)
    setParsedInput(null)
    setForecastResults([])
    setForecastError(null)
    setForecastState('idle')
    setValidationError(null)
    setValidationState('selected')
    if (file.size > 5 * 1024 * 1024) {
      setValidationState('invalid')
      setValidationError('The selected file is larger than the 5 MB limit.')
      return
    }
    setValidationState('validating')
    try {
      const parsed = parseForecastCsv(await file.text())
      setParsedInput(parsed)
      setValidationState('ready')
    } catch (error) {
      setParsedInput(null)
      setValidationState('invalid')
      setValidationError(error instanceof CsvValidationError ? error.message : 'The selected file could not be read as CSV.')
    }
  }

  const submitForecast = async () => {
    if (submitLock.current || !parsedInput || validationState !== 'ready' || connectionStatus !== 'connected' || model === null) return
    submitLock.current = true
    setForecastState('loading')
    setForecastError(null)
    setForecastResults([])
    try {
      const payload: ForecastRequest = toForecastRequest(parsedInput)
      let nextResults: ForecastDisplay[]
      if (model === 'both') {
        const response = await api.compareForecasts(payload)
        validateComparisonResponse(response, inferenceExperimentId)
        nextResults = MODEL_ORDER.map((name) => ({
          modelName: name,
          experimentId: response.experiment_id,
          targetZoneNames: response.target_zone_names,
          forecastSteps: response.models[name].forecast_steps,
          unitsStatus: response.units_status,
        }))
      } else {
        const modelName: ModelName = model === 'lgt' ? 'LinearGridTransformer' : 'CDLinear'
        const response = await api.forecast(modelName, payload)
        validateSingleResponse(response, modelName, inferenceExperimentId)
        nextResults = [{
          modelName: response.model_name,
          experimentId: response.experiment_id,
          targetZoneNames: response.target_zone_names,
          forecastSteps: response.forecast_steps,
          unitsStatus: response.units_status,
        }]
      }
      setForecastResults(nextResults)
      setForecastState('completed')
    } catch (error) {
      const unavailable = error instanceof ApiClientError && (error.status === 0 || error.status === 503)
      setForecastState(unavailable ? 'unavailable' : 'failed')
      setForecastError(forecastErrorMessage(error))
      if (unavailable) void refreshHealth()
    } finally {
      submitLock.current = false
    }
  }

  const fileStatus = validationState === 'ready' ? 'valid' : validationState === 'invalid' ? 'invalid' : validationState === 'selected' || validationState === 'validating' ? 'selected' : ''
  const forecastDisabled = !parsedInput || validationState !== 'ready' || forecastState === 'loading' || connectionStatus !== 'connected' || model === null
  const featureLabels = parsedInput?.headers.filter((header) => header !== 'DateTime' && header !== 'timestamp') ?? features
  const zoneLabels = historicalZoneNames
  const timestampLabel = parsedInput?.timestamps ? `${parsedInput.timestamps[0]} – ${parsedInput.timestamps.at(-1)}` : null

  return <div className="page">
    <HistoricalStatus resource="models" />
    <div className="page-intro"><div><Badge tone="blue">Forecast workspace</Badge><h1>Generate a six-step forecast</h1><p>Provide 24 hours of ordered observations, select a model, and inspect the next 60 minutes across all three zones.</p></div><div className="api-state"><ApiStatus /></div></div>
    {connectionStatus === 'checking' && <div className="data-binding-state" role="status"><span className="data-binding-indicator" /><span>Checking backend availability. Forecast submission will be enabled when inference is ready.</span></div>}
    {connectionStatus === 'unavailable' && <div className="data-binding-state data-binding-error" role="status"><span className="data-binding-indicator" /><span>Forecast service unavailable. Recheck the backend connection before submitting.</span><button className="text-action" onClick={() => void refreshHealth()}>Retry connection</button></div>}
    <div className="forecast-layout">
      <div className="input-column">
        <Card>
          <SectionHeader eyebrow="Step 01" title="Historical input" action={<Badge>CSV</Badge>} />
          <div className={`upload-area ${fileStatus}`} role="button" tabIndex={0} aria-label="Choose a historical CSV file" onClick={() => inputRef.current?.click()} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click() }} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); void handleFile(event.dataTransfer.files[0]) }}>
            <input ref={inputRef} type="file" accept=".csv,text/csv" aria-label="Upload forecast CSV" onChange={(e) => void handleFile(e.target.files?.[0])} />
            <span className="upload-icon"><Icon name={validationState === 'ready' ? 'check' : validationState === 'invalid' ? 'alert' : validationState === 'validating' ? 'activity' : 'upload'} /></span>
            {validationState === 'empty' && <><strong>Drop historical observations here</strong><span>or click to browse a CSV file</span></>}
            {validationState === 'selected' && <><strong>{fileName}</strong><span>File selected · validation has not completed</span></>}
            {validationState === 'validating' && <><strong>{fileName}</strong><span>Reading and validating CSV…</span></>}
            {validationState === 'ready' && <><strong>{fileName}</strong><span>144 observations validated · input ready</span></>}
            {validationState === 'invalid' && <><strong>{fileName ?? 'CSV validation failed'}</strong><span>{validationError}</span></>}
          </div>
          <div className="upload-actions"><Button variant="ghost" disabled={!fileName} onClick={clearFile}>Remove selected file</Button><span>Maximum 5 MB</span></div>
          {validationState === 'ready' && parsedInput && <div className="csv-preview"><div className="preview-title"><strong>Parsed preview</strong><span>First 5 of 144 rows · {parsedInput.timestamps ? 'DateTime included' : 'No timestamps supplied'}</span></div>{timestampLabel && <p className="muted-note">Provided timestamp range: {timestampLabel} · the backend validates format, ordering, timezone, and cadence.</p>}<div className={`data-table preview-table ${parsedInput.timestamps ? 'has-timestamps' : ''}`}><div className="data-row data-head"><span>#</span>{parsedInput.timestamps && <span>DateTime</span>}{featureLabels.map((name) => <span key={name}>{name}</span>)}</div>{parsedInput.observations.slice(0, 5).map((row, index) => <div className="data-row" key={index}><span>{index + 1}</span>{parsedInput.timestamps && <span>{parsedInput.timestamps[index]}</span>}{row.map((value, column) => <span key={column}>{value}</span>)}</div>)}</div></div>}
        </Card>
        <Card>
          <SectionHeader eyebrow="Input contract" title="Expected data shape" />
          <div className="contract-grid">{[['144', 'observations'], ['8', 'ordered features'], ['10 min', 'sampling interval'], ['24 hr', 'lookback']].map(([v, l]) => <div key={l}><strong>{v}</strong><span>{l}</span></div>)}</div>
          <div className="feature-list">{features.map((f, i) => <div key={f}><span>{String(i + 1).padStart(2, '0')}</span>{f}<Icon name="check" size={14} /></div>)}</div>
          <div className="info-callout"><Icon name="info" size={16} /><p>Column order is required. Numeric values are shown without energy units because physical units are undocumented in the source dataset.</p></div>
        </Card>
      </div>
      <div className="result-column">
        <Card>
          <SectionHeader eyebrow="Step 02" title="Forecast model" />
          <div className="segmented">
            {availableModels.map((info) => {
              const id = info.name === 'CDLinear' ? 'cd' : 'lgt'
              return <button key={info.name} className={model === id ? 'active' : ''} onClick={() => setModel(id)}>{info.name}{info.selected ? ' · recommended' : ' · experimental'}</button>
            })}
            {canCompare && <button className={model === 'both' ? 'active' : ''} onClick={() => setModel('both')}>Compare both</button>}
          </div>
          <div className="model-summary"><div><span>Selected output</span><strong>{model === 'both' ? '2 models × 6 steps × 3 zones' : model === null ? 'Available model information is loading' : '6 steps × 3 zones'}</strong><span>Inference experiment · {inferenceExperimentId}</span></div><Button disabled={forecastDisabled} onClick={() => void submitForecast()}>{forecastState === 'loading' ? 'Generating forecast…' : 'Generate forecast'}</Button></div>
        </Card>
        {forecastState === 'loading' && <Card className="empty-state forecast-progress" role="status"><span className="empty-visual"><span className="spinner" /></span><h2>Generating forecast</h2><p>The selected model request is running against the backend.</p></Card>}
        {forecastState === 'unavailable' && <Card className="empty-state error-state"><span className="empty-visual"><Icon name="alert" size={26} /></span><Badge tone="amber">Backend unavailable</Badge><h2>Forecast could not be generated</h2><p>{forecastError}</p><Button variant="secondary" onClick={() => void refreshHealth()}>Retry connection</Button></Card>}
        {forecastState === 'failed' && <Card className="empty-state error-state"><span className="empty-visual"><Icon name="alert" size={26} /></span><Badge tone="amber">Request failed</Badge><h2>Forecast was not generated</h2><p>{forecastError}</p></Card>}
        {forecastState === 'completed' && forecastResults.length > 0 && <div className={`forecast-results ${forecastResults.length > 1 ? 'comparison-results' : ''}`}>{forecastResults.map((result) => <ForecastResultPanel key={result.modelName} result={result} zones={zoneLabels} />)}</div>}
        {forecastState === 'idle' && <Card className="empty-state"><span className="empty-visual"><Icon name="activity" size={26} /></span><h2>{validationState === 'invalid' ? 'CSV needs correction' : 'Ready when your input is ready'}</h2><p>{validationState === 'invalid' ? validationError : validationState === 'ready' ? 'Review the parsed rows, select a model, then generate a genuine backend forecast.' : 'Upload a CSV with 144 observations and the eight canonical feature columns in order.'}</p><div className="step-preview">{['+10', '+20', '+30', '+40', '+50', '+60'].map((s) => <span key={s}>{s}<small>min</small></span>)}</div></Card>}
      </div>
    </div>
  </div>
}

interface ForecastDisplay {
  modelName: ModelName
  experimentId: string
  targetZoneNames: TargetName[]
  forecastSteps: ForecastStep[]
  unitsStatus: string
}

const historicalZoneNames = ['Zone 1 Power Consumption', 'Zone 2 Power Consumption', 'Zone 3 Power Consumption']

function validateSteps(steps: ForecastStep[]): void {
  if (steps.length !== 6 || steps.some((step, index) => step.horizon_step !== index + 1 || step.minutes_ahead !== (index + 1) * 10 || step.values.length !== 3 || step.values.some((value) => !Number.isFinite(value)))) {
    throw new Error('The backend returned a forecast with an invalid shape or non-finite values.')
  }
}

function validateSingleResponse(response: ForecastResponse, expectedModel: ModelName, expectedExperiment: string): void {
  if (response.model_name !== expectedModel || response.experiment_id !== expectedExperiment || response.input_dimensions.join(',') !== '144,8' || response.output_dimensions.join(',') !== '6,3' || response.target_zone_names.join(',') !== TARGET_ZONE_NAMES.join(',')) {
    throw new Error('The backend returned a forecast that did not match the expected model contract.')
  }
  validateSteps(response.forecast_steps)
}

function validateComparisonResponse(response: ForecastComparisonResponse, expectedExperiment: string): void {
  if (response.experiment_id !== expectedExperiment || response.input_dimensions.join(',') !== '144,8' || response.target_zone_names.join(',') !== TARGET_ZONE_NAMES.join(',')) {
    throw new Error('The backend returned a comparison that did not match the expected input contract.')
  }
  for (const name of MODEL_ORDER) {
    const modelResponse = response.models[name]
    if (!modelResponse || modelResponse.output_dimensions.join(',') !== '6,3') {
      throw new Error('The backend returned an incomplete model comparison.')
    }
    validateSteps(modelResponse.forecast_steps)
  }
}

function forecastErrorMessage(error: unknown): string {
  if (error instanceof CsvValidationError) return error.message
  if (error instanceof ApiClientError) {
    if (error.status === 422) return 'The backend rejected this input. Check that timestamps are timezone-naive, strictly chronological, and exactly ten minutes apart.'
    if (error.status === 404) return 'The requested model or experiment was not found.'
    if (error.status === 503) return 'The inference service is currently unavailable. Check the backend and retry.'
    return 'The forecast request failed. You can retry without re-uploading the validated CSV.'
  }
  return 'The backend returned an invalid forecast response. No result was accepted.'
}

function ForecastResultPanel({ result, zones }: { result: ForecastDisplay; zones: string[] }) {
  const series = zones.map((zone, index) => ({
    name: zone,
    values: result.forecastSteps.map((step) => step.values[index]),
    color: ZONE_COLORS[index % ZONE_COLORS.length],
  }))
  return <Card className="forecast-result-card">
    <SectionHeader eyebrow="Forecast generated from uploaded historical input" title={result.modelName} action={<Badge tone="green">Backend response</Badge>} />
    <div className="forecast-result-meta"><span>Experiment <strong>{result.experimentId || DEFAULT_EXPERIMENT_ID}</strong></span><span>Input <strong>144 × 8</strong></span><span>Output <strong>6 × 3</strong></span></div>
    <DataLineChart series={series} labels={result.forecastSteps.map((step) => `${step.minutes_ahead}m`)} height="medium" ariaLabel={`${result.modelName} forecast values for the three target zones`} />
    <div className="forecast-table"><div className="table-row table-head"><span>Horizon</span>{zones.map((zone) => <span key={zone}>{zone}</span>)}</div>{result.forecastSteps.map((step) => <div className="table-row" key={step.horizon_step}><span>Step {step.horizon_step} · {step.minutes_ahead} min</span>{step.values.map((value, index) => <span key={index}>{formatNumber(value, 4)}</span>)}</div>)}</div>
    <p className="muted-note">{result.unitsStatus}</p>
  </Card>
}

function Models() {
  const [metric, setMetric] = useState<'mae' | 'rmse'>('mae')
  const { experimentId, historical } = useAppData()
  const models = historical.models
  const metrics = historical.metrics
  const history = historical.history
  const experiment = historical.experiment
  const zones = metrics?.horizon_wise.target_zone_names ?? []
  const zoneLabels = historical.dataset?.targets.source_columns ?? zones.map(humanize)
  const modelInfo = (name: ModelName) => models?.find((item) => item.name === name)
  const historyFor = (name: ModelName): TrainingEpoch[] => history?.models[name] ?? []
  const isLatest = experimentId === DEFAULT_EXPERIMENT_ID
  const modelRole = (name: ModelName) => {
    if (isLatest) return modelInfo(name)?.role ?? 'Role unavailable'
    const role = experiment?.models[name]?.role
    return role && isRoleRecorded(role) ? role : 'Role not recorded for this experiment'
  }
  const isSelected = (name: ModelName) => isLatest
    ? Boolean(modelInfo(name)?.selected)
    : Boolean(experiment?.models[name]?.selected)
  const hasPooled = MODEL_ORDER.some((name) => metrics?.overall.models[name].pooled_across_all_samples_horizons_and_zones)
  return <div className="page">
    <HistoricalStatus resource="models" /><HistoricalStatus resource="history" /><HistoricalStatus resource="metrics" /><HistoricalStatus resource="experiment" />
    <div className="page-intro"><div><Badge tone="blue">Technical analysis</Badge><h1>Model Laboratory</h1><p>Configurations, saved training behavior, and held-out metrics from the experiment artifacts.</p></div><Badge>{experiment?.run_identifier ?? experimentId} · {metrics ? metric.toUpperCase() : 'metrics unavailable'}</Badge></div>
    <div className="model-card-grid">{MODEL_ORDER.map((name) => {
      const info = modelInfo(name)
      const config = info?.configuration ?? {}
      const best = experiment?.models[name]
      const parameterCount = config.parameter_count ?? config.num_parameters
      return <ModelCard key={name} name={name} role={modelRole(name)} subtitle={name === 'CDLinear' ? 'Channel-independent linear forecasting model' : 'Linear-attention Transformer architecture'} params={typeof parameterCount === 'number' ? formatNumber(parameterCount, 0) : 'Not reported by API'} input={info?.input_shape} output={info?.forecast_output_shape} mae={metricMean(metrics, name, 'mae')} maeStd={metricStd(metrics, name, 'mae')} rmse={metricMean(metrics, name, 'rmse')} rmseStd={metricStd(metrics, name, 'rmse')} bestEpoch={best?.best_validation_epoch} bestLoss={best?.best_validation_loss} selected={isSelected(name)} />
    })}</div>
    <div className="grid-two">
      <Card className="chart-card"><SectionHeader title="Training behavior" action={<span className="chart-note">{history?.models.CDLinear?.[0]?.seeds ? 'Mean loss across seeds · saved epoch history' : 'Saved epoch history'}</span>} />
        {MODEL_ORDER.map((name, modelIndex) => {
          const rows = historyFor(name)
          return <div className="history-model" key={name}><h3>{name}</h3><DataLineChart series={[
            { name: 'Training loss', values: rows.map((row) => row.train_loss), color: MODEL_COLORS[modelIndex * 2] },
            { name: 'Validation loss', values: rows.map((row) => row.validation_loss), color: MODEL_COLORS[modelIndex * 2 + 1] },
          ]} labels={rows.map((row) => String(row.epoch))} height="small" ariaLabel={`${name} training and validation loss by epoch`} /></div>
        })}
      </Card>
      <Card><SectionHeader title="Best validation checkpoints" /><div className="metric-table"><div className="metric-row metric-head"><span>Model · seed</span><span>Best epoch</span><span>Validation loss</span></div>{MODEL_ORDER.flatMap((name) => {
        const record = experiment?.models[name]
        const rows = record?.seed_runs?.length ? record.seed_runs : [{ seed: record?.checkpoint_seed ?? null, best_validation_epoch: record?.best_validation_epoch, best_validation_loss: record?.best_validation_loss, checkpoint: record?.checkpoint }]
        return rows.map((run) => <div className="metric-row" key={`${name}-${run.seed}`}><span>{name} · seed {run.seed ?? 'not reported'}{run.seed === experiment?.model_selection?.inference_checkpoint_seed ? ' · inference checkpoint' : ''}</span><strong>{run.best_validation_epoch ?? 'Not available'}</strong><strong><MetricValue value={run.best_validation_loss} /></strong></div>)
      })}</div><p className="muted-note">Checkpoint selection uses validation loss independently by model and seed. Test metrics below summarize the separate held-out report.</p></Card>
    </div>
    <Card className="analysis-card">
      <SectionHeader title="Per-zone evaluation" action={<div className="metric-toggle"><button className={metric === 'mae' ? 'active' : ''} onClick={() => setMetric('mae')}>MAE</button><button className={metric === 'rmse' ? 'active' : ''} onClick={() => setMetric('rmse')}>RMSE</button></div>} />
      <div className="metric-table"><div className="metric-row metric-head"><span>Target zone · {metric.toUpperCase()}</span><span>LinearGridTransformer</span><span>CDLinear</span></div>{zones.map((zone, index) => <div className="metric-row" key={zone}><span>{zoneLabels[index] ?? humanize(zone)}{latestSummary(metrics?.overall.models.CDLinear) ? ' · three-seed mean' : ''}</span><strong><MetricValue value={perZoneMetric(metrics, 'LinearGridTransformer', zone, metric)} /></strong><strong><MetricValue value={perZoneMetric(metrics, 'CDLinear', zone, metric)} /></strong></div>)}</div>
    </Card>
    <Card className="analysis-card">
      <SectionHeader title="Horizon-wise error" action={<span className="chart-note">Saved test report · one point per 10-minute lead</span>} />
      <div className="analysis-grid">
        {MODEL_ORDER.map((name, modelIndex) => {
          const seedRows = latestHorizonRows(metrics, name)
          if (seedRows) {
            const grouped = groupedHorizonValues(seedRows, metric === 'mae' ? 'MAE' : 'RMSE')
            return <div key={name}><span className="eyebrow">{name} · saved report aggregation · seed mean</span><DataLineChart series={[{ name: `${metric.toUpperCase()} mean`, values: grouped.map((row) => row.mean), color: MODEL_COLORS[modelIndex * 2] }]} labels={grouped.map((row) => `${row.minutes}m`)} ariaLabel={`${name} mean ${metric.toUpperCase()} by forecast horizon`} /></div>
          }
          const legacyRows = (metrics?.horizon_wise.models[name] ?? []).filter((row): row is HorizonMetric => 'horizon_minutes' in row)
          return <div key={name}><span className="eyebrow">{name} · per-zone {metric.toUpperCase()}</span><DataLineChart series={zones.map((zone, index) => ({ name: zoneLabels[index] ?? humanize(zone), values: legacyRows.map((row) => row.per_zone[zone]?.[metric] ?? Number.NaN), color: ZONE_COLORS[index % ZONE_COLORS.length] }))} labels={legacyRows.map((row) => `${row.horizon_minutes}m`)} ariaLabel={`${name} per-zone ${metric.toUpperCase()} by forecast horizon`} /></div>
        })}
      </div>
      <div className="metric-table horizon-metric-table"><div className="metric-row metric-head"><span>Lead · reported {metric.toUpperCase()}</span><span>{isLatest ? 'LinearGridTransformer seed mean ± std' : 'LinearGridTransformer'}</span><span>{isLatest ? 'CDLinear seed mean ± std' : 'CDLinear'}</span></div>{[...horizonDisplayRows(metrics, 'LinearGridTransformer', metric, zones, zoneLabels), ...horizonDisplayRows(metrics, 'CDLinear', metric, zones, zoneLabels)].filter((row, index, rows) => rows.findIndex((candidate) => candidate.lead === row.lead) === index).map((row) => {
        const left = horizonDisplayRows(metrics, 'LinearGridTransformer', metric, zones, zoneLabels).find((candidate) => candidate.lead === row.lead)
        const right = horizonDisplayRows(metrics, 'CDLinear', metric, zones, zoneLabels).find((candidate) => candidate.lead === row.lead)
        return <div className="metric-row" key={row.key}><span>{row.lead}</span><strong><MeanAndStd mean={left?.mean} std={left?.std} /></strong><strong><MeanAndStd mean={right?.mean} std={right?.std} /></strong></div>
      })}</div>
      {MODEL_ORDER.some((name) => (metrics?.overall.models[name]?.per_seed?.length ?? 0) > 0) && <div className="metric-table"><div className="metric-row metric-head"><span>Seed-level held-out results</span><span>MAE</span><span>RMSE</span></div>{MODEL_ORDER.flatMap((name) => (metrics?.overall.models[name]?.per_seed ?? []).map((row) => <div className="metric-row" key={`${name}-${row.seed}`}><span>{name} · seed {row.seed}</span><strong><MetricValue value={row.MAE} /></strong><strong><MetricValue value={row.RMSE} /></strong></div>))}</div>}
    </Card>
    <div className="grid-two">
      <Card><SectionHeader title="Evaluation baseline" /><div className="metric-table"><div className="metric-row metric-head"><span>Baseline</span><span>Reported MAE</span><span>Reported RMSE</span></div>{Object.entries(metrics?.overall.baselines ?? {}).map(([name, baseline]) => <div className="metric-row" key={name}><span>{humanize(name)} · {('trained' in baseline && baseline.trained === false) || name === 'persistence' ? 'not trained' : 'baseline'}</span><strong><MetricValue value={baselineMetric(metrics, name, 'mae')} /></strong><strong><MetricValue value={baselineMetric(metrics, name, 'rmse')} /></strong></div>)}</div><p className="muted-note">Baselines are evaluation references, not trained models or selectable forecast models.</p></Card>
      {hasPooled && <Card><SectionHeader title="Pooled comparison" /><div className="metric-table"><div className="metric-row metric-head"><span>Model / baseline</span><span>Pooled MAE</span><span>Pooled RMSE</span></div>{[...MODEL_ORDER.map((name) => [name, metrics?.overall.models[name].pooled_across_all_samples_horizons_and_zones] as const), ...Object.entries(metrics?.overall.baselines ?? {}).map(([name, value]) => [humanize(name), 'pooled_across_all_samples_horizons_and_zones' in value ? value.pooled_across_all_samples_horizons_and_zones : undefined] as const)].map(([name, values]) => <div className="metric-row" key={name}><span>{name}</span><strong><MetricValue value={values?.mae} /></strong><strong><MetricValue value={values?.rmse} /></strong></div>)}</div><p className="muted-note">Pooled values combine test windows, forecast horizons, and zones.</p></Card>}
    </div>
  </div>
}

function ModelCard({ name, role, subtitle, params, input, output, mae, maeStd, rmse, rmseStd, bestEpoch, bestLoss, selected }: { name: string; role: string; subtitle: string; params: string; input?: number[]; output?: number[]; mae?: number; maeStd?: number; rmse?: number; rmseStd?: number; bestEpoch?: number | null; bestLoss?: number | null; selected?: boolean }) {
  return <Card className={`model-card ${selected ? 'winner' : ''}`}><div className="model-top"><span className="model-icon"><Icon name={selected ? 'activity' : 'cube'} /></span><Badge tone={selected ? 'green' : 'neutral'}>{humanize(role)}</Badge></div><h2>{name}</h2><p>{subtitle}</p><div className="io-row"><span>Input <strong>{input ? input.join(' × ') : 'Not available'}</strong></span><Icon name="arrow" size={14} /><span>Output <strong>{output ? output.join(' × ') : 'Not available'}</strong></span></div><div className="model-stats"><div><span>Parameters</span><strong>{params}</strong></div><div><span>{maeStd == null ? 'Macro MAE' : 'Mean MAE ± std'}</span><strong><MeanAndStd mean={mae} std={maeStd} /></strong></div><div><span>{rmseStd == null ? 'Macro RMSE' : 'Mean RMSE ± std'}</span><strong><MeanAndStd mean={rmse} std={rmseStd} /></strong></div></div><p className="muted-note">Best validation epoch: {bestEpoch ?? 'Not available'} · loss: {formatNumber(bestLoss, 6)}</p></Card>
}

function splitWindowCount(dataset: DatasetInfo, key: string): number | null {
  const saved = dataset.windows.samples?.[key]
  if (typeof saved === 'number') return saved
  const split = dataset.splits.partitions[key]
  if (!split) return null
  const contextCarries = key !== 'train' && dataset.windows.validation_test_context_may_precede_partition
  const count = split.row_count - dataset.windows.horizon + 1 - (contextCarries ? 0 : dataset.windows.lookback)
  return Math.max(0, count)
}

function Dataset() {
  const { historical } = useAppData()
  const dataset = historical.dataset
  if (!dataset) return <div className="page"><HistoricalStatus resource="dataset" /></div>
  const targetSet = new Set<string>(dataset.targets.names)
  const sourceColumns = dataset.features.source_columns ?? dataset.features.names
  const partitionEntries = Object.entries(dataset.splits.partitions)
  return <div className="page">
    <HistoricalStatus resource="dataset" />
    <div className="page-intro"><div><Badge tone="blue">Dataset provenance</Badge><h1>{dataset.dataset_name}</h1><p>{dataset.source.attribution ?? 'Attribution is not available in the dataset API.'}</p></div><a className="button button-secondary citation-link" href={dataset.source.dataset_url ?? '#'} target="_blank" rel="noreferrer">UCI dataset · DOI {dataset.source.doi ?? 'not supplied'}</a></div>
    <div className="kpi-grid">
      <Kpi label="Observations" value={formatNumber(dataset.row_count, 0)} detail={dataset.source_file_available ? 'Raw source artifact available' : 'Raw source artifact unavailable'} />
      <Kpi label="Sampling cadence" value={dataset.cadence_minutes == null ? 'Not available' : `${dataset.cadence_minutes} min`} detail="From dataset metadata" />
      <Kpi label="Input features" value={formatNumber(dataset.features.names.length, 0)} detail={`${dataset.targets.names.length} target channels`} />
      <Kpi label="Coverage" value={formatDate(dataset.date_range.first)} detail={`through ${formatDate(dataset.date_range.last)}`} />
    </div>
    <Card className="timeline-card"><SectionHeader title="Chronological partitions" /><div className="split-list">{partitionEntries.map(([name, part]) => <div className="split-row" key={name}><strong>{humanize(name)}</strong><span>{formatNumber(part.row_count, 0)} rows · indices [{part.start_index}, {part.end_index_exclusive})</span><span>{formatDate(part.first_timestamp)} – {formatDate(part.last_timestamp)}</span><span>{formatNumber(splitWindowCount(dataset, name), 0)} windows{dataset.windows.samples?.[name] == null ? ' · derived from split/window contract' : ''}</span></div>)}</div></Card>
    <Card className="feature-card"><SectionHeader eyebrow="Source schema" title="Input feature order and target roles" /><div className="data-table"><div className="data-row data-head"><span>#</span><span>Source column</span><span>Feature</span><span>Role</span></div>{sourceColumns.map((source, index) => { const name = dataset.features.names[index]; return <div className="data-row" key={`${index}-${source}`}><span>{index + 1}</span><span>{source}</span><span>{name ? humanize(name) : 'Not available'}</span><span className="role-cell">{name && targetSet.has(name) ? 'Input · target' : 'Input'}</span></div> })}</div><p className="muted-note">Source header whitespace is preserved in the displayed column names. No feature descriptions are supplied by this metadata endpoint.</p></Card>
    <div className="grid-two">
      <Card><SectionHeader title="Windowing & normalization" /><div className="detail-list"><div><span>Split strategy</span><strong>{dataset.splits.method ?? 'Not available'}</strong></div><div><span>Input window</span><strong>{dataset.windows.lookback} observations</strong></div><div><span>Forecast horizon</span><strong>{dataset.windows.horizon} observations</strong></div><div><span>Stride</span><strong>{dataset.windows.stride}</strong></div><div><span>Normalization</span><strong>{dataset.normalization.method}</strong></div><div><span>Statistics fit on</span><strong>{dataset.normalization.fit_partition} · rows [{dataset.normalization.fit_row_range_half_open.join(', ')})</strong></div><div><span>Population std ddof</span><strong>{dataset.normalization.std_ddof}</strong></div></div><p className="muted-note">Features are ordered as listed above. Targets are {dataset.targets.source_columns?.join(', ') ?? dataset.targets.names.map(humanize).join(', ')}.</p><div className="data-table compact-normalization"><div className="data-row data-head"><span>Channel</span><span>Training mean</span><span>Training std</span></div>{dataset.normalization.feature_names.map((name, index) => <div className="data-row" key={name}><span>{name}</span><span>{formatNumber(dataset.normalization.feature_mean[index], 5)}</span><span>{formatNumber(dataset.normalization.feature_std[index], 5)}</span></div>)}</div></Card>
      <Card className="limitations"><SectionHeader title="Data quality & uncertainties" action={<Icon name="alert" size={18} />} />{dataset.data_quality ? <ul>{Object.entries(dataset.data_quality).filter(([key]) => key !== 'available').map(([key, value]) => <li key={key}><span>{humanize(key)}</span><strong>{value == null ? 'Not available' : String(value)}</strong></li>)}</ul> : <p className="muted-note">Data quality report is not available from the API.</p>}<ul><li><span>Physical units</span><strong>{dataset.coverage_uncertainties.units_status ?? dataset.coverage_uncertainties.units ?? 'Unknown'}</strong></li><li><span>Timezone</span><strong>{dataset.timezone_status ?? 'Unknown'}</strong></li><li><span>Missing final day</span><strong>{dataset.coverage_uncertainties.missing_final_calendar_day ?? 'Not reported'}</strong></li><li><span>Handling</span><strong>{dataset.coverage_uncertainties.handling ?? 'Not reported'}</strong></li></ul></Card>
    </div>
    <Card><SectionHeader title="Dataset citation" /><p>{dataset.source.attribution ?? 'Attribution not supplied.'} DOI: {dataset.source.doi ?? 'not supplied'} · License: {dataset.source.license ?? 'not supplied'} · <a href={dataset.source.source_url} target="_blank" rel="noreferrer">Original data source</a></p></Card>
  </div>
}

function Experiment() {
  const { experimentId, historical } = useAppData()
  const experiment = historical.experiment
  const metrics = historical.metrics
  const isLatest = experimentId === DEFAULT_EXPERIMENT_ID
  const [open, setOpen] = useState<string[]>(['configuration'])
  const toggle = (id: string) => setOpen((s) => s.includes(id) ? s.filter(x => x !== id) : [...s, id])
  if (!experiment) return <div className="page"><HistoricalStatus resource="experiment" /></div>
  const training = experiment.training_configuration ?? {}
  const seedPolicy = experiment.seed_policy ?? {}
  const environment = experiment.provenance?.training_environment ?? {}
  const zones = metrics?.horizon_wise.target_zone_names ?? []
  const zoneLabels = historical.dataset?.targets.source_columns ?? zones.map(humanize)
  const cdHorizons = latestHorizonRows(metrics, 'CDLinear')
  const lgtHorizons = latestHorizonRows(metrics, 'LinearGridTransformer')
  const cdHorizonMeans = cdHorizons ? groupedHorizonValues(cdHorizons, 'MAE') : []
  const lgtHorizonMeans = lgtHorizons ? groupedHorizonValues(lgtHorizons, 'MAE') : []
  const formatValue = (value: unknown) => value == null ? 'Not available' : typeof value === 'object' ? JSON.stringify(value) : String(value)
  return <div className="page">
    <HistoricalStatus resource="experiment" />
    <div className="page-intro"><div><Badge tone="blue">Experiment provenance</Badge><h1>{experiment.run_identifier}</h1><p>Manifest-backed configuration, best validation records, environment, and artifact availability.</p></div><div className="experiment-actions"><Badge>{experiment.run_type ?? 'Experiment'}</Badge><span className="chart-note">Created {experiment.created_utc ?? 'date not recorded'}</span></div></div>
    <Card className="provenance-banner"><div><span className="eyebrow">Experiment identifier</span><strong>{experiment.run_identifier}</strong></div><div><span>Optimizer / loss</span><strong>{experiment.optimizer ?? 'Not available'} / {experiment.loss ?? 'Not available'}</strong></div><div><span>Git commit</span><strong>{experiment.provenance?.git_commit ?? 'Not recorded'}</strong></div><div><span>Test partition</span><Badge tone={experiment.test_partition_used === false ? 'green' : 'amber'}>{experiment.test_partition_used === false ? 'Not used in training' : experiment.test_partition_used === true ? 'Used' : 'Not recorded'}</Badge></div></Card>
    {metrics && <Card className="analysis-card"><SectionHeader title="Held-out evaluation summary" action={<span className="chart-note">Saved report · {experimentId}</span>} />
      <div className="metric-table"><div className="metric-row metric-head"><span>Model or baseline</span><span>{isLatest ? 'MAE · mean ± std' : 'Macro MAE'}</span><span>{isLatest ? 'RMSE · mean ± std' : 'Macro RMSE'}</span></div>{MODEL_ORDER.map((name) => <div className="metric-row" key={name}><span>{name} · {modelInfoRole(name, experiment, historical.models, experimentId)}</span><strong><MeanAndStd mean={metricMean(metrics, name, 'mae')} std={metricStd(metrics, name, 'mae')} /></strong><strong><MeanAndStd mean={metricMean(metrics, name, 'rmse')} std={metricStd(metrics, name, 'rmse')} /></strong></div>)}{Object.entries(metrics.overall.baselines).map(([name]) => <div className="metric-row" key={name}><span>{humanize(name)} · evaluation baseline</span><strong><MetricValue value={baselineMetric(metrics, name, 'mae')} /></strong><strong><MetricValue value={baselineMetric(metrics, name, 'rmse')} /></strong></div>)}</div>
      {isLatest && <p className="muted-note">Seed standard deviations are shown only where supplied by the experiment summary. Persistence has no training seed or standard deviation.</p>}
      <div className="metric-table"><div className="metric-row metric-head"><span>Per-zone MAE{isLatest ? ' · three-seed mean' : ''}</span><span>LinearGridTransformer</span><span>CDLinear</span></div>{zones.map((zone, index) => <div className="metric-row" key={zone}><span>{zoneLabels[index] ?? humanize(zone)}</span><strong><MetricValue value={perZoneMetric(metrics, 'LinearGridTransformer', zone, 'mae')} /></strong><strong><MetricValue value={perZoneMetric(metrics, 'CDLinear', zone, 'mae')} /></strong></div>)}</div>
      {(cdHorizonMeans.length > 0 || lgtHorizonMeans.length > 0) && <div className="metric-table horizon-metric-table"><div className="metric-row metric-head"><span>Forecast lead · saved MAE mean ± seed std</span><span>LinearGridTransformer</span><span>CDLinear</span></div>{cdHorizonMeans.map((row, index) => <div className="metric-row" key={row.step}><span>{row.minutes} minutes</span><strong><MeanAndStd mean={lgtHorizonMeans[index]?.mean} std={lgtHorizonMeans[index]?.std} /></strong><strong><MeanAndStd mean={row.mean} std={row.std} /></strong></div>)}</div>}
      {!isLatest && zones.length > 0 && <div className="metric-table horizon-metric-table"><div className="metric-row metric-head"><span>Forecast lead · target zone · MAE</span><span>LinearGridTransformer</span><span>CDLinear</span></div>{horizonDisplayRows(metrics, 'LinearGridTransformer', 'mae', zones, zoneLabels).map((row) => <div className="metric-row" key={row.key}><span>{row.lead}</span><strong><MetricValue value={row.mean} /></strong><strong><MetricValue value={horizonDisplayRows(metrics, 'CDLinear', 'mae', zones, zoneLabels).find((item) => item.lead === row.lead)?.mean} /></strong></div>)}</div>}
    </Card>}
    <div className="grid-two experiment-grid">
      <Card><SectionHeader title="Training configuration" /><div className="config-grid">{Object.entries(training).map(([key, value]) => <div key={key}><span>{humanize(key)}</span><strong>{formatValue(value)}</strong></div>)}{Object.keys(training).length === 0 && <p className="muted-note">Training configuration is not available.</p>}</div><p className="muted-note">Seed policy: {Object.entries(seedPolicy).map(([key, value]) => `${humanize(key)}: ${formatValue(value)}`).join(' · ') || 'Not recorded'}</p></Card>
      <Card><SectionHeader title="Training environment" /><div className="config-grid">{Object.entries(environment).map(([key, value]) => <div key={key}><span>{humanize(key)}</span><strong>{formatValue(value)}</strong></div>)}<div><span>Backend inference device</span><strong>{historical.models?.map((model) => model.selected_inference_device).filter(Boolean)[0] ?? 'Not available'}</strong></div>{Object.keys(environment).length === 0 && <p className="muted-note">Environment details are not available.</p>}</div></Card>
    </div>
    <div className="accordion">
      <Accordion id="models" title="Model configurations" subtitle="Architecture dimensions and parameter counts" open={open.includes('models')} onToggle={toggle}>
        <div className="model-card-grid">{MODEL_ORDER.map((name) => { const info = historical.models?.find((row) => row.name === name); const record = experiment.models[name]; return <div className="model-config" key={name}><h3>{name} · {modelInfoRole(name, experiment, historical.models, experimentId)}</h3><div className="config-grid">{Object.entries(record.configuration).map(([key, value]) => <div key={key}><span>{humanize(key)}</span><strong>{formatValue(value)}</strong></div>)}<div><span>Input shape</span><strong>{info?.input_shape.join(' × ') ?? 'Not available'}</strong></div><div><span>Forecast output shape</span><strong>{info?.forecast_output_shape.join(' × ') ?? 'Not available'}</strong></div><div><span>Checkpoint seed</span><strong>{record.checkpoint_seed ?? 'Not reported'}</strong></div></div><p>Best validation epoch: {record.best_validation_epoch ?? 'Not available'} · loss: {formatNumber(record.best_validation_loss, 6)}</p></div>})}</div>
      </Accordion>
      <Accordion id="artifacts" title="Artifacts & references" subtitle="Checkpoints, logs, and reports" open={open.includes('artifacts')} onToggle={toggle}>
        <div className="artifact-list">{Object.entries(experiment.artifact_references).map(([key, artifact]) => <span key={key}><Icon name={artifact.available ? 'check' : 'alert'} size={15} />{humanize(key)} · <strong>{artifact.name}</strong> · {artifact.available ? `Available${artifact.size_bytes == null ? '' : ` · ${formatNumber(artifact.size_bytes, 0)} bytes`}` : 'Unavailable'}</span>)}</div>
        <p className="muted-note">The API reports file availability and size. No artifact verification status is supplied. Local paths are intentionally not shown.</p>
      </Accordion>
    </div>
  </div>
}

function Accordion({ id, title, subtitle, open, onToggle, children }: { id: string; title: string; subtitle: string; open: boolean; onToggle: (id: string) => void; children: React.ReactNode }) {
  return <Card className={`accordion-item ${open ? 'open' : ''}`}><button onClick={() => onToggle(id)}><span><strong>{title}</strong><small>{subtitle}</small></span><Icon name="chevron" /></button>{open && <div className="accordion-content">{children}</div>}</Card>
}

function Sidebar({ page, setPage, open, close }: { page: Page; setPage: (p: Page) => void; open: boolean; close: () => void }) {
  return <aside className={`sidebar ${open ? 'mobile-open' : ''}`}>
    <div className="brand"><span className="brand-mark">L</span><div><strong>LinearGrid</strong><span>Transformer</span></div><button className="mobile-close" onClick={close}><Icon name="close" /></button></div>
    <span className="nav-label">Workspace</span>
    <nav>{navItems.map((item) => <button key={item.id} className={page === item.id ? 'active' : ''} onClick={() => { setPage(item.id); close() }}><Icon name={item.icon} /><span>{item.label}</span>{page === item.id && <i />}</button>)}</nav>
    <div className="sidebar-footer">
      <div className="project-mini"><span className="project-icon"><Icon name="activity" size={16} /></span><div><strong>Research project</strong><span>UCI Tetouan dataset</span></div></div>
      <div className="backend-row"><ApiStatus compact /><span>v1.0.0</span></div>
    </div>
  </aside>
}

function AppShell() {
  const { experimentId, setExperimentId } = useAppData()
  const [page, setPage] = useState<Page>('overview')
  const [menuOpen, setMenuOpen] = useState(false)
  const content = useMemo(() => {
    if (page === 'overview') return <Overview goTo={setPage} />
    if (page === 'forecast') return <Forecast />
    if (page === 'models') return <Models />
    if (page === 'dataset') return <Dataset />
    return <Experiment />
  }, [page])
  return <div className="app-shell">
    <Sidebar page={page} setPage={setPage} open={menuOpen} close={() => setMenuOpen(false)} />
    {menuOpen && <div className="mobile-overlay" onClick={() => setMenuOpen(false)} />}
    <main>
      <header className="topbar">
        <div className="topbar-left"><button className="menu-button" onClick={() => setMenuOpen(true)}><Icon name="menu" /></button><div className="breadcrumbs"><span>LinearGridTransformer</span><Icon name="chevron" size={13} /><strong>{pageTitles[page]}</strong></div></div>
        <div className="topbar-actions"><label className="experiment-select"><span>Experiment</span><select aria-label="Select historical experiment" value={experimentId} onChange={(event) => setExperimentId(event.target.value)}>{EXPERIMENT_OPTIONS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}</select><Icon name="chevron" size={13} /></label><ApiStatus /><button className="icon-button" aria-label="Settings"><Icon name="settings" /></button></div>
      </header>
      <div className="content">{content}</div>
    </main>
  </div>
}

function App() {
  return <AppDataProvider><AppShell /></AppDataProvider>
}

export default App
