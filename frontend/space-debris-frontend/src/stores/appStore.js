import { writable, derived } from 'svelte/store'

// ── Backend connection status ──────────────────────────────────────────────────
export const backendOnline = writable(false)

// ── Latest prediction result from /predict ───────────────────────────────────
export const prediction = writable(null)

// ── Currently selected debris object ─────────────────────────────────────────
export const selectedObject = writable(null)

// ── Live telemetry feed (ring buffer, last 20 entries) ────────────────────────
export const telemetryFeed = writable([])

export function pushTelemetry(entry) {
  telemetryFeed.update(feed => {
    const next = [{ ...entry, ts: new Date().toISOString() }, ...feed]
    return next.slice(0, 20)
  })
}

// ── Active alert count derived from telemetry ────────────────────────────────
export const alertCount = derived(telemetryFeed, $feed =>
  $feed.filter(e => e.risk_level === 'HIGH').length
)

// ── Input form values ────────────────────────────────────────────────────────
export const formValues = writable({
  distance_km:   50,
  rel_velocity:  7,
  approach_rate: -5
})

// ── UI state ─────────────────────────────────────────────────────────────────
export const activePanel = writable('dashboard') // 'dashboard' | 'predict' | 'conjunctions' | 'telemetry'
export const globeRotating = writable(true)
