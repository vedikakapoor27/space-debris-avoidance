import { getAccessToken, getRefreshToken, setSession, clearSession } from '../stores/authStore.js'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

export function apiUrl(path = '') {
  const base = BASE_URL.replace(/\/$/, '')
  const suffix = path.startsWith('/') ? path : `/${path}`
  return `${base}${suffix}`
}

async function parseJson(res) {
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new Error(data.message || `Request failed (${res.status})`)
  }
  return data
}

export async function refreshAccessToken() {
  const refreshToken = getRefreshToken()
  if (!refreshToken) throw new Error('No refresh token')

  const res = await fetch(apiUrl('/auth/refresh'), {
    method: 'POST',
    headers: { Authorization: `Bearer ${refreshToken}` },
  })
  const data = await parseJson(res)

  const current = JSON.parse(localStorage.getItem('astraeus_user') || 'null')
  setSession({
    user: current,
    access_token: data.access_token,
    refresh_token: refreshToken,
  })
  return data.access_token
}

export async function authFetch(path, options = {}) {
  const token = getAccessToken()
  const headers = {
    ...options.headers,
  }

  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = headers['Content-Type'] || 'application/json'
  }

  if (token) {
    headers.Authorization = `Bearer ${token}`
  }

  let res = await fetch(apiUrl(path), { ...options, headers })

  if (res.status === 401 && getRefreshToken()) {
    try {
      const newToken = await refreshAccessToken()
      headers.Authorization = `Bearer ${newToken}`
      res = await fetch(apiUrl(path), { ...options, headers })
    } catch {
      clearSession()
      throw new Error('Session expired. Please log in again.')
    }
  }

  return res
}

export async function checkHealth() {
  const res = await fetch(apiUrl('/health'))
  if (!res.ok) throw new Error('Backend offline')
  return res.json()
}

export async function login({ username, password }) {
  const res = await fetch(apiUrl('/auth/login'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  const data = await parseJson(res)
  setSession(data)
  return data
}

export async function register({ username, email, password }) {
  const res = await fetch(apiUrl('/auth/register'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password }),
  })
  const data = await parseJson(res)
  setSession(data)
  return data
}

export async function getMe(token) {
  const res = await fetch(apiUrl('/auth/me'), {
    headers: { Authorization: `Bearer ${token}` },
  })
  return parseJson(res)
}

export async function predict({ distance_km, rel_velocity, approach_rate }) {
  const res = await authFetch('/predict', {
    method: 'POST',
    body: JSON.stringify({ distance_km, rel_velocity, approach_rate }),
  })
  const data = await res.json()
  if (data.status === 'error') throw new Error(data.message)
  return data
}

export async function getStats() {
  const res = await authFetch('/stats')
  return parseJson(res)
}

export async function getHistory(limit = 50) {
  const res = await authFetch(`/history?limit=${limit}`)
  return parseJson(res)
}

export async function clearHistory() {
  const res = await authFetch('/history/clear', { method: 'DELETE' })
  return parseJson(res)
}

export function generateDebrisField(count = 80) {
  return Array.from({ length: count }, (_, i) => {
    const theta = Math.random() * Math.PI * 2
    const phi   = Math.acos(2 * Math.random() - 1)
    const r     = 1.3 + Math.random() * 1.2
    return {
      id:    i,
      x:     r * Math.sin(phi) * Math.cos(theta),
      y:     r * Math.sin(phi) * Math.sin(theta),
      z:     r * Math.cos(phi),
      speed: 0.001 + Math.random() * 0.004,
      size:  0.006 + Math.random() * 0.012,
      risk:  Math.random(),
      label: i < 5 ? `SAT-${1000 + i}` : `DEB-${2000 + i}`
    }
  })
}

export function getMockConjunctions() {
  return [
    { id: 'CJ-001', object1: 'ISS',         object2: 'DEB-2041', distance: 8.2,  velocity: 12.4, risk: 'HIGH',   time: '00:14:22' },
    { id: 'CJ-002', object1: 'SAT-1001',    object2: 'DEB-2089', distance: 34.7, velocity: 7.1,  risk: 'MEDIUM', time: '01:02:10' },
    { id: 'CJ-003', object1: 'SAT-1002',    object2: 'DEB-2103', distance: 112,  velocity: 4.3,  risk: 'LOW',    time: '02:45:00' },
    { id: 'CJ-004', object1: 'Starlink-12', object2: 'DEB-2200', distance: 19.1, velocity: 9.8,  risk: 'HIGH',   time: '00:31:55' },
    { id: 'CJ-005', object1: 'SAT-1003',    object2: 'DEB-2310', distance: 67.3, velocity: 5.5,  risk: 'MEDIUM', time: '03:18:40' },
  ]
}
