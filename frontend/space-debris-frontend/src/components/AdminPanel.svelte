<script>
  import { onMount } from 'svelte'
  import { fade } from 'svelte/transition'
  import { authStore } from '../stores/authStore.js'
  import { getUsers, updateUser } from '../utils/api.js'
  import { roleLabel } from '../utils/permissions.js'

  let users = []
  let loading = true
  let error = ''
  let savingId = null

  const roles = ['viewer', 'operator', 'admin']

  async function loadUsers() {
    loading = true
    error = ''
    try {
      const data = await getUsers()
      users = data.users || []
    } catch (err) {
      error = err.message
      users = []
    } finally {
      loading = false
    }
  }

  async function changeRole(user, role) {
    if (user.role === role) return
    savingId = user.id
    error = ''
    try {
      const data = await updateUser(user.id, { role })
      users = users.map((u) => (u.id === user.id ? data.user : u))
      if ($authStore.user?.id === user.id) {
        authStore.update((s) => ({ ...s, user: data.user }))
        localStorage.setItem('astraeus_user', JSON.stringify(data.user))
      }
    } catch (err) {
      error = err.message
    } finally {
      savingId = null
    }
  }

  async function toggleActive(user) {
    savingId = user.id
    error = ''
    try {
      const data = await updateUser(user.id, { is_active: !user.is_active })
      users = users.map((u) => (u.id === user.id ? data.user : u))
    } catch (err) {
      error = err.message
    } finally {
      savingId = null
    }
  }

  onMount(loadUsers)

  const fmtDate = (iso) => iso ? new Date(iso).toLocaleDateString() : '--'
</script>

<div class="admin">
  <div class="admin-header">
    <div>
      <div class="eyebrow">Administration</div>
      <h2 class="title">User<br>Management</h2>
    </div>
    <button class="refresh-btn" on:click={loadUsers} disabled={loading}>
      {loading ? 'Loading...' : 'Refresh'}
    </button>
  </div>

  {#if error}
    <div class="error" transition:fade>{error}</div>
  {/if}

  {#if loading}
    <div class="loading">Loading users...</div>
  {:else if !users.length}
    <div class="empty">No users found.</div>
  {:else}
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>User</th>
            <th>Email</th>
            <th>Role</th>
            <th>Status</th>
            <th>Joined</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {#each users as user (user.id)}
            <tr class:inactive={!user.is_active}>
              <td>
                <div class="username">{user.username}</div>
                {#if user.id === $authStore.user?.id}
                  <span class="you-tag">You</span>
                {/if}
              </td>
              <td class="mono">{user.email}</td>
              <td>
                <select
                  value={user.role}
                  disabled={savingId === user.id}
                  on:change={(e) => changeRole(user, e.currentTarget.value)}
                >
                  {#each roles as role}
                    <option value={role}>{roleLabel(role)}</option>
                  {/each}
                </select>
              </td>
              <td>
                <span class="status" class:active={user.is_active}>
                  {user.is_active ? 'Active' : 'Inactive'}
                </span>
              </td>
              <td class="mono">{fmtDate(user.created_at)}</td>
              <td>
                <button
                  class="toggle-btn"
                  disabled={savingId === user.id || user.id === $authStore.user?.id}
                  on:click={() => toggleActive(user)}
                >
                  {user.is_active ? 'Deactivate' : 'Activate'}
                </button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}

  <div class="legend">
  <div class="legend-item"><strong>Viewer</strong> — read dashboard, conjunctions, own history</div>
  <div class="legend-item"><strong>Operator</strong> — run predictions + telemetry + full history</div>
  <div class="legend-item"><strong>Admin</strong> — clear history + manage users</div>
  </div>
</div>

<style>
  .admin {
    padding: 24px;
    height: 100%;
    overflow-y: auto;
  }

  .admin-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
    gap: 16px;
  }

  .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: var(--text-4);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }

  .title {
    font-family: 'Inter', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
  }

  .refresh-btn, .toggle-btn {
    height: 32px;
    padding: 0 12px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-3);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
  }

  .refresh-btn:hover:not(:disabled),
  .toggle-btn:hover:not(:disabled) {
    border-color: var(--border2);
    color: var(--text);
  }

  .refresh-btn:disabled,
  .toggle-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .error {
    margin-bottom: 16px;
    padding: 10px 12px;
    border: 1px solid var(--border2);
    background: var(--danger-bg);
    color: var(--text-2);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
  }

  .loading, .empty {
    padding: 40px;
    text-align: center;
    color: var(--text-4);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
  }

  .table-wrap {
    border: 1px solid var(--border);
    overflow-x: auto;
    margin-bottom: 20px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    min-width: 720px;
  }

  th {
    text-align: left;
    padding: 10px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-4);
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }

  td {
    padding: 12px;
    border-bottom: 1px solid var(--divider);
    color: var(--text-2);
    font-size: 12px;
    vertical-align: middle;
  }

  tr.inactive td {
    opacity: 0.55;
  }

  .username {
    font-weight: 600;
    color: var(--text);
  }

  .you-tag {
    display: inline-block;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-4);
  }

  .mono {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
  }

  select {
    height: 30px;
    padding: 0 8px;
    background: var(--bg2);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
  }

  .status {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-4);
  }

  .status.active {
    color: var(--text-2);
  }

  .legend {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 14px;
    border: 1px solid var(--border);
    background: var(--card);
  }

  .legend-item {
    font-size: 11px;
    color: var(--text-3);
    line-height: 1.5;
  }

  .legend-item strong {
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
</style>
