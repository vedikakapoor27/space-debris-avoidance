export const ROLES = {
  VIEWER: 'viewer',
  OPERATOR: 'operator',
  ADMIN: 'admin',
}

const PANEL_ACCESS = {
  dashboard: [ROLES.VIEWER, ROLES.OPERATOR, ROLES.ADMIN],
  predict: [ROLES.OPERATOR, ROLES.ADMIN],
  conjunctions: [ROLES.VIEWER, ROLES.OPERATOR, ROLES.ADMIN],
  telemetry: [ROLES.OPERATOR, ROLES.ADMIN],
  history: [ROLES.VIEWER, ROLES.OPERATOR, ROLES.ADMIN],
  admin: [ROLES.ADMIN],
}

export function canAccessPanel(role, panelId) {
  return PANEL_ACCESS[panelId]?.includes(role) ?? false
}

export function canPredict(role) {
  return role === ROLES.OPERATOR || role === ROLES.ADMIN
}

export function canUseTelemetry(role) {
  return canPredict(role)
}

export function canViewFullHistory(role) {
  return role === ROLES.OPERATOR || role === ROLES.ADMIN
}

export function canClearHistory(role) {
  return role === ROLES.ADMIN
}

export function canManageUsers(role) {
  return role === ROLES.ADMIN
}

export function roleLabel(role) {
  return {
    viewer: 'Viewer',
    operator: 'Operator',
    admin: 'Administrator',
  }[role] || role
}

export function firstAllowedPanel(role) {
  const order = ['dashboard', 'conjunctions', 'history', 'predict', 'telemetry', 'admin']
  return order.find((panel) => canAccessPanel(role, panel)) || 'dashboard'
}
