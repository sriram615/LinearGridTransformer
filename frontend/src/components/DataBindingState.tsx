import { useAppData } from '../api/context'

export function DataBindingState({ source }: { source: string }) {
  const { connectionStatus } = useAppData()
  const state = connectionStatus === 'checking'
    ? 'loading'
    : connectionStatus === 'unavailable'
      ? 'error'
      : 'empty'

  const message = state === 'loading'
    ? 'Checking the backend. This page is showing its saved design snapshot.'
    : state === 'error'
      ? 'Backend unavailable. No live data is loaded; the content below is a saved design snapshot.'
    : `Backend connected. ${source} is not loaded into this saved design snapshot yet; live page binding is planned for WP 3.2.`

  return (
    <div className={`data-binding-state data-binding-${state}`} role="status" aria-live="polite">
      <span className="data-binding-indicator" aria-hidden="true" />
      <span>{message}</span>
    </div>
  )
}
