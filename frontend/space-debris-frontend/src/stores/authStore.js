import { writable, get } from 'svelte/store'

const ACCESS_KEY = 'astraeus_access_token'
const REFRESH_KEY = 'astraeus_refresh_token'
const USER_KEY = 'astraeus_user'

function readStored() {
  try {
    return {
      user: JSON.parse(localStorage.getItem(USER_KEY) || 'null'),
      accessToken: localStorage.getItem(ACCESS_KEY),
      refreshToken: localStorage.getItem(REFRESH_KEY),
    }
  } catch {
    return { user: null, accessToken: null, refreshToken: null }
  }
}

export const authStore = writable({
  user: null,
  accessToken: null,
  refreshToken: null,
  loading: true,
  isAuthenticated: false,
})

export function setSession({ user, access_token, refresh_token }) {
  localStorage.setItem(ACCESS_KEY, access_token)
  localStorage.setItem(REFRESH_KEY, refresh_token)
  localStorage.setItem(USER_KEY, JSON.stringify(user))
  authStore.set({
    user,
    accessToken: access_token,
    refreshToken: refresh_token,
    loading: false,
    isAuthenticated: true,
  })
}

export function clearSession() {
  localStorage.removeItem(ACCESS_KEY)
  localStorage.removeItem(REFRESH_KEY)
  localStorage.removeItem(USER_KEY)
  authStore.set({
    user: null,
    accessToken: null,
    refreshToken: null,
    loading: false,
    isAuthenticated: false,
  })
}

export function getAccessToken() {
  return get(authStore).accessToken
}

export function getRefreshToken() {
  return get(authStore).refreshToken
}

export async function initAuth(validateMe) {
  const stored = readStored()
  if (!stored.accessToken) {
    authStore.set({
      user: null,
      accessToken: null,
      refreshToken: null,
      loading: false,
      isAuthenticated: false,
    })
    return
  }

  authStore.set({
    ...stored,
    loading: true,
    isAuthenticated: false,
  })

  try {
    const data = await validateMe(stored.accessToken)
    authStore.set({
      user: data.user,
      accessToken: stored.accessToken,
      refreshToken: stored.refreshToken,
      loading: false,
      isAuthenticated: true,
    })
    localStorage.setItem(USER_KEY, JSON.stringify(data.user))
  } catch {
    clearSession()
  }
}
