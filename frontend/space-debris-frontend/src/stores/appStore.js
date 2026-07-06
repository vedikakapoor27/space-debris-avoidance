import { writable, derived } from 'svelte/store'

export const theme = writable('dark')
export const backendOnline = writable(false)
export const prediction = writable(null)
export const selectedObject = writable(null)
export const telemetryFeed = writable([])
export const activePanel = writable('dashboard')
export const globeRotating = writable(true)

export function pushTelemetry(entry) {
  telemetryFeed.update(feed => {
    const next = [{ ...entry, ts: new Date().toISOString() }, ...feed]
    return next.slice(0, 20)
  })
}

export const alertCount = derived(telemetryFeed, $feed =>
  $feed.filter(e => e.risk_level === 'HIGH').length
)

export const formValues = writable({
  distance_km: 50,
  rel_velocity: 7,
  approach_rate: -5
})