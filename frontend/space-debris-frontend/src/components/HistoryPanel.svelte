<script>
  import { onMount, onDestroy } from 'svelte'
  import { fly, fade } from 'svelte/transition'
  import { cubicOut } from 'svelte/easing'
  import { getStats, getHistory, clearHistory as clearHistoryApi } from '../utils/api.js'
  import { authStore } from '../stores/authStore.js'
  import { canClearHistory, canViewFullHistory } from '../utils/permissions.js'

  let stats = null
  let history = []
  let loading = true
  let activeTab = 'stats'
  let interval
  let scope = 'all'

  $: role = $authStore.user?.role || 'viewer'
  $: showClear = canClearHistory(role)

  async function fetchStats() {
    try {
      stats = await getStats()
    } catch { stats = null }
  }

  async function fetchHistory() {
    try {
      const data = await getHistory(50)
      history = data.history || []
      scope = data.scope || 'all'
    } catch { history = [] }
  }

  async function clearHistory() {
    try {
      await clearHistoryApi()
      history = []
      await fetchStats()
    } catch {}
  }

  async function load() {
    loading = true
    await Promise.all([fetchStats(), fetchHistory()])
    loading = false
  }

  function exportCSV() {
    if (!history.length) return
    const headers = ['timestamp','risk_level','probability','confidence','distance_km','rel_velocity','approach_rate','action','urgency']
    const rows = history.map(h => [
      h.timestamp,
      h.risk_level,
      h.probability,
      h.confidence,
      h.input?.distance_km,
      h.input?.rel_velocity,
      h.input?.approach_rate,
      h.action,
      h.urgency
    ].join(','))
    const csv = [headers.join(','), ...rows].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url; a.download = 'astraeus_predictions.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  onMount(() => {
    load()
    interval = setInterval(load, 30000)
  })
  onDestroy(() => clearInterval(interval))

  const riskCls = r => r === 'HIGH' ? 'high' : r === 'MEDIUM' ? 'med' : 'low'
  const fmtTime = iso => iso ? new Date(iso).toTimeString().slice(0,8) : '--'
  const fmtDate = iso => iso ? new Date(iso).toLocaleDateString() : '--'
</script>

<div class="hp">

  <div class="hp-header">
    <div>
      <div class="hp-eyebrow">Analytics</div>
      <h2 class="hp-title">Prediction<br>History</h2>
    </div>
    <div class="hp-actions">
      <button class="act-btn export" on:click={exportCSV} disabled={!history.length}>
        ↓ Export CSV
      </button>
      {#if showClear}
        <button class="act-btn clear" on:click={clearHistory} disabled={!history.length}>
          ✕ Clear
        </button>
      {/if}
    </div>
  </div>

  {#if !canViewFullHistory(role)}
    <div class="scope-note">Showing your predictions only</div>
  {/if}

  <!-- TABS -->
  <div class="tabs">
    <button class="tab" class:active={activeTab==='stats'}   on:click={() => activeTab='stats'}>
      Stats Overview
    </button>
    <button class="tab" class:active={activeTab==='history'} on:click={() => activeTab='history'}>
      Prediction Log
      {#if history.length > 0}
        <span class="tab-count">{history.length}</span>
      {/if}
    </button>
  </div>

  {#if loading}
    <div class="loading">
      <span class="spin">◌</span> Loading...
    </div>

  {:else if activeTab === 'stats'}

    {#if !stats || stats.total_predictions === 0}
      <div class="empty">
        <div class="empty-ico">◈</div>
        <p>No predictions yet</p>
        <p class="empty-s">Run a prediction to see stats here</p>
      </div>
    {:else}
      <div class="stats-wrap" in:fly={{ y: 12, duration: 280, easing: cubicOut }}>

        <!-- TOP STATS -->
        <div class="top-stats">
          <div class="top-stat">
            <div class="ts-val">{stats.total_predictions}</div>
            <div class="ts-key">Total Predictions</div>
          </div>
          <div class="top-stat" style="--ac:var(--danger)">
            <div class="ts-val" style="color:var(--danger)">{stats.by_risk?.HIGH ?? 0}</div>
            <div class="ts-key">Critical</div>
          </div>
          <div class="top-stat" style="--ac:var(--warning)">
            <div class="ts-val" style="color:var(--warning)">{stats.by_risk?.MEDIUM ?? 0}</div>
            <div class="ts-key">Warning</div>
          </div>
          <div class="top-stat" style="--ac:var(--safe)">
            <div class="ts-val" style="color:var(--safe)">{stats.by_risk?.LOW ?? 0}</div>
            <div class="ts-key">Nominal</div>
          </div>
        </div>

        <!-- RISK DISTRIBUTION BAR -->
        <div class="section">
          <div class="section-head">
            <span class="section-title">Risk Distribution</span>
          </div>
          <div class="dist-wrap">
            <div class="dist-bar">
              {#if stats.percentages?.HIGH > 0}
                <div class="db-seg high" style="width:{stats.percentages.HIGH}%">
                  {#if stats.percentages.HIGH > 8}{stats.percentages.HIGH}%{/if}
                </div>
              {/if}
              {#if stats.percentages?.MEDIUM > 0}
                <div class="db-seg med" style="width:{stats.percentages.MEDIUM}%">
                  {#if stats.percentages.MEDIUM > 8}{stats.percentages.MEDIUM}%{/if}
                </div>
              {/if}
              {#if stats.percentages?.LOW > 0}
                <div class="db-seg low" style="width:{stats.percentages.LOW}%">
                  {#if stats.percentages.LOW > 8}{stats.percentages.LOW}%{/if}
                </div>
              {/if}
            </div>
            <div class="dist-legend">
              <span class="dl-item high">■ Critical {stats.percentages?.HIGH}%</span>
              <span class="dl-item med">■ Warning {stats.percentages?.MEDIUM}%</span>
              <span class="dl-item low">■ Nominal {stats.percentages?.LOW}%</span>
            </div>
          </div>
        </div>

        <!-- PROBABILITY STATS -->
        <div class="section">
          <div class="section-head">
            <span class="section-title">Probability Analysis</span>
          </div>
          <div class="prob-grid">
            {#each [
              ['Average', stats.probability?.average + '%', 'var(--accent-hi)'],
              ['Maximum', stats.probability?.max + '%',     'var(--danger)'],
              ['Minimum', stats.probability?.min + '%',     'var(--safe)'],
            ] as [k, v, c]}
              <div class="prob-card">
                <div class="prob-val" style="color:{c}">{v}</div>
                <div class="prob-key">{k}</div>
              </div>
            {/each}
          </div>
        </div>

        <!-- ALERT BANNER -->
        {#if stats.alert}
          <div class="alert-banner" in:fly={{ y: 8, duration: 250 }}>
            <span class="alert-dot"></span>
            ⚠ High alert rate detected — {stats.percentages?.HIGH}% of predictions are CRITICAL
          </div>
        {/if}

        <!-- LATEST PREDICTION -->
        {#if stats.latest}
          <div class="section">
            <div class="section-head">
              <span class="section-title">Latest Prediction</span>
              <span class="section-time">{fmtTime(stats.latest.timestamp)}</span>
            </div>
            <div class="latest-wrap">
              <div class="latest-risk">
                <span class="pill {riskCls(stats.latest.risk_level)}">{stats.latest.risk_level}</span>
                <span class="latest-prob">{stats.latest.probability}%</span>
              </div>
              <div class="latest-inputs">
                <span>D: {stats.latest.input?.distance_km} km</span>
                <span>V: {stats.latest.input?.rel_velocity} km/s</span>
                <span>A: {stats.latest.input?.approach_rate}</span>
              </div>
            </div>
          </div>
        {/if}

      </div>
    {/if}

  {:else}

    <!-- HISTORY LOG -->
    {#if history.length === 0}
      <div class="empty">
        <div class="empty-ico">⎍</div>
        <p>No prediction history</p>
        <p class="empty-s">Predictions will appear here automatically</p>
      </div>
    {:else}
      <div class="history-wrap" in:fly={{ y: 12, duration: 280, easing: cubicOut }}>
        {#each history as entry, i}
          <div
            class="h-entry"
            class:h-high={entry.risk_level === 'HIGH'}
            in:fly={{ x: -8, duration: 200, delay: i < 10 ? i * 30 : 0, easing: cubicOut }}
          >
            <div class="he-time">
              <div class="he-t">{fmtTime(entry.timestamp)}</div>
              <div class="he-d">{fmtDate(entry.timestamp)}</div>
            </div>

            <span class="pill {riskCls(entry.risk_level)}">{entry.risk_level}</span>

            <div class="he-prob">{entry.probability}%</div>

            <div class="he-conf">
              <span class="conf-badge conf-{entry.confidence?.toLowerCase()}">{entry.confidence}</span>
            </div>

            <div class="he-inputs">
              <span>D: {entry.input?.distance_km}km</span>
              <span>V: {entry.input?.rel_velocity}</span>
              <span>A: {entry.input?.approach_rate}</span>
            </div>

            <div class="he-action">{entry.action}</div>

            <div class="he-urgency urgency-{riskCls(entry.risk_level)}">{entry.urgency}</div>
          </div>
        {/each}
      </div>
    {/if}

  {/if}

</div>

<style>
  .hp {
    padding: 28px; height: 100%;
    display: flex; flex-direction: column; gap: 20px;
    overflow-y: auto;
  }

  .hp-header {
    display: flex; justify-content: space-between; align-items: flex-start;
  }

  .scope-note {
    margin-top: -8px;
    padding: 8px 10px;
    border: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-4);
  }

  .hp-eyebrow {
    font-family: 'Inter', monospace;
    font-size: 9px; color: var(--text-dim);
    letter-spacing: 0.22em; text-transform: uppercase; margin-bottom: 8px;
  }

  .hp-title {
    font-family: 'Inter', sans-serif;
    font-size: 24px; font-weight: 800; color: var(--text);
    line-height: 1.15; letter-spacing: -0.01em;
  }

  .hp-actions { display: flex; gap: 8px; align-items: flex-start; margin-top: 4px; }

  .act-btn {
    font-family: 'Inter', sans-serif; font-size: 10px; font-weight: 600;
    letter-spacing: 0.08em; padding: 7px 14px; cursor: pointer;
    background: transparent; border: 1px solid; transition: all 0.2s;
  }
  .act-btn.export { color: var(--safe);    border-color: rgba(0,232,160,0.25); }
  .act-btn.clear  { color: var(--danger);  border-color: rgba(255,56,96,0.25); }
  .act-btn:hover:not(:disabled)  { background: var(--glass); }
  .act-btn:disabled { opacity: 0.3; cursor: not-allowed; }

  .tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border-dim); }

  .tab {
    font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 600;
    letter-spacing: 0.06em; padding: 10px 20px;
    background: none; border: none; cursor: pointer;
    color: var(--text-dim); border-bottom: 2px solid transparent;
    margin-bottom: -1px; transition: all 0.2s;
    display: flex; align-items: center; gap: 8px;
  }
  .tab:hover  { color: var(--text-mid); }
  .tab.active { color: var(--accent-hi); border-bottom-color: var(--accent); }

  .tab-count {
    background: var(--accent); color: white;
    font-family: 'Inter', monospace;
    font-size: 8px; padding: 1px 6px; border-radius: 10px;
  }

  .loading {
    display: flex; align-items: center; gap: 8px;
    font-family: 'Inter', monospace;
    font-size: 11px; color: var(--text-dim);
    padding: 40px; justify-content: center;
  }
  .spin { animation: spin 1s linear infinite; display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .empty {
    display: flex; flex-direction: column;
    align-items: center; gap: 10px;
    padding: 60px; opacity: 0.25;
  }
  .empty-ico { font-size: 44px; color: var(--accent); }
  .empty p   { font-family: 'Inter', sans-serif; font-size: 13px; color: var(--text-dim); margin: 0; }
  .empty-s   { font-size: 11px !important; }

  /* STATS */
  .stats-wrap { display: flex; flex-direction: column; gap: 16px; }

  .top-stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; }

  .top-stat {
    background: var(--glass); border: 1px solid var(--border-dim);
    padding: 16px; text-align: center; transition: border-color 0.2s;
  }
  .top-stat:hover { border-color: var(--border); }

  .ts-val {
    font-family: 'Inter', sans-serif; font-size: 28px; font-weight: 800;
    color: var(--accent-hi); line-height: 1; margin-bottom: 6px;
  }
  .ts-key {
    font-family: 'Inter', monospace; font-size: 8px;
    color: var(--text-dim); letter-spacing: 0.14em; text-transform: uppercase;
  }

  .section { border: 1px solid var(--border-dim); overflow: hidden; }

  .section-head {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 16px; border-bottom: 1px solid var(--border-dim);
    background: var(--glass);
  }
  .section-title {
    font-family: 'Inter' sans-serif; font-size: 11px; font-weight: 700;
    color: var(--accent-hi); letter-spacing: 0.08em; text-transform: uppercase;
  }
  .section-time {
    font-family: 'Inter', monospace; font-size: 9px; color: var(--text-dim);
  }

  .dist-wrap { padding: 16px; display: flex; flex-direction: column; gap: 12px; }

  .dist-bar {
    display: flex; height: 28px; overflow: hidden; border-radius: 2px;
    gap: 2px;
  }
  .db-seg {
    display: flex; align-items: center; justify-content: center;
    font-family: 'Inter', monospace; font-size: 9px; color: white;
    transition: width 0.8s ease; font-weight: 500;
  }
  .db-seg.high { background: var(--danger); }
  .db-seg.med  { background: var(--warning); }
  .db-seg.low  { background: var(--safe); }

  .dist-legend { display: flex; gap: 16px; }
  .dl-item {
    font-family: 'Inter', monospace; font-size: 9px; letter-spacing: 0.1em;
  }
  .dl-item.high { color: var(--danger); }
  .dl-item.med  { color: var(--warning); }
  .dl-item.low  { color: var(--safe); }

  .prob-grid { display: grid; grid-template-columns: repeat(3,1fr); }

  .prob-card {
    padding: 16px; text-align: center;
    border-right: 1px solid var(--border-dim);
  }
  .prob-card:last-child { border-right: none; }

  .prob-val {
    font-family: 'Inter', sans-serif; font-size: 22px; font-weight: 800;
    line-height: 1; margin-bottom: 6px;
  }
  .prob-key {
    font-family: 'Inter', monospace; font-size: 8px;
    color: var(--text-dim); letter-spacing: 0.14em; text-transform: uppercase;
  }

  .alert-banner {
    display: flex; align-items: center; gap: 10px;
    background: rgba(255,56,96,0.07); border: 1px solid rgba(255,56,96,0.25);
    border-left: 3px solid var(--danger); padding: 12px 16px;
    font-family: 'Syne', sans-serif; font-size: 11px; font-weight: 600;
    color: var(--danger);
    animation: alertpulse 2s ease-in-out infinite;
  }
  @keyframes alertpulse { 0%,100%{opacity:1} 50%{opacity:0.7} }

  .alert-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--danger); flex-shrink: 0;
    animation: alertpulse 1s ease-in-out infinite;
  }

  .latest-wrap { padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; }

  .latest-risk { display: flex; align-items: center; gap: 12px; }

  .latest-prob {
    font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 800;
    color: var(--text);
  }

  .latest-inputs {
    display: flex; gap: 14px;
    font-family: 'Inter', monospace;
    font-size: 10px; color: var(--text-dim);
  }

  /* HISTORY LOG */
  .history-wrap { display: flex; flex-direction: column; gap: 5px; }

  .h-entry {
    display: grid;
    grid-template-columns: 80px 75px 55px 80px 1fr 1fr 80px;
    align-items: center; gap: 10px;
    padding: 10px 14px;
    background: var(--glass); border: 1px solid var(--border-dim);
    transition: all 0.2s;
  }
  .h-entry:hover    { border-color: var(--border); }
  .h-entry.h-high   { border-color: rgba(255,56,96,0.18); background: rgba(255,56,96,0.03); }

  .he-time { display: flex; flex-direction: column; gap: 2px; }
  .he-t    { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text); }
  .he-d    { font-family: 'JetBrains Mono', monospace; font-size: 8px; color: var(--text-dim); }

  .he-prob {
    font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 700; color: var(--text);
  }

  .conf-badge {
    font-family: 'Inter', monospace; font-size: 8px;
    padding: 2px 7px; letter-spacing: 0.1em; text-transform: uppercase;
  }
  .conf-high   { background: rgba(0,232,160,0.1);  color: var(--safe);    border: 1px solid rgba(0,232,160,0.2); }
  .conf-medium { background: rgba(255,144,32,0.1); color: var(--warning); border: 1px solid rgba(255,144,32,0.2); }
  .conf-low    { background: rgba(255,56,96,0.1);  color: var(--danger);  border: 1px solid rgba(255,56,96,0.2); }

  .he-inputs {
    display: flex; gap: 8px; flex-wrap: wrap;
    font-family: 'Inter', monospace; font-size: 9px; color: var(--text-dim);
  }

  .he-action {
    font-family: 'Inter', sans-serif; font-size: 10px; font-weight: 600; color: var(--text-dim);
  }

  .he-urgency {
    font-family: 'Inter', monospace; font-size: 9px;
    letter-spacing: 0.12em; text-align: right; text-transform: uppercase;
  }
  .urgency-high { color: var(--danger); }
  .urgency-med  { color: var(--warning); }
  .urgency-low  { color: var(--safe); }

  .pill {
    font-family: 'Inter', sans-serif; font-weight: 700;
    font-size: 9px; letter-spacing: 0.08em; padding: 3px 9px;
    text-transform: uppercase;
  }
  .pill.high { background: rgba(255,56,96,0.12);  color: var(--danger);  border: 1px solid rgba(255,56,96,0.3); }
  .pill.med  { background: rgba(255,144,32,0.1);  color: var(--warning); border: 1px solid rgba(255,144,32,0.28); }
  .pill.low  { background: rgba(0,232,160,0.08);  color: var(--safe);    border: 1px solid rgba(0,232,160,0.22); }
</style>