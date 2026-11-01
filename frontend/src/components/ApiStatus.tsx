import { useAppData } from '../api/context'

export function ApiStatus({ compact = false }: { compact?: boolean }) {
  const { connectionStatus } = useAppData()
  const label = connectionStatus === 'connected'
    ? 'Connected'
    : connectionStatus === 'unavailable'
      ? 'Unavailable'
      : 'Checking'

  return (
    <span className={`status status-${connectionStatus}`} role="status" aria-live="polite">
      <i aria-hidden="true" />{compact ? `API ${label.toLowerCase()}` : label}
    </span>
  )
}
