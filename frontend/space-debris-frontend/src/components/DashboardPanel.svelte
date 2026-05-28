<script>
  import { onMount, onDestroy } from 'svelte'
  import { getMockConjunctions } from '../utils/api.js'
  import { selectedObject } from '../stores/appStore.js'

  let conjunctions = getMockConjunctions()
  let interval

  let stats = { tracked: 847, active_sats: 12, high_risk: 2, avg_distance: 156.4 }

  onMount(() => {
    interval = setInterval(() => {
      stats.avg_distance = +(154 + Math.random() * 6).toFixed(1)
    }, 2000)
  })
  onDestroy(() => clearInterval(interval))

  const riskCls = r => r === 'HIGH' ? 'high' : r === 'MEDIUM' ? 'med' : 'low'
</script>

<div class="panel">

  <div class="panel-header">
    <div class="header-left">
      <div class="header-eyebrow">MISSION OVERVIEW</div>
      <h2 class="header-title">Orbital Collision<br>Monitoring</h2>
    </div>
    <div class="live-badge">
      <span class="live-dot"></span>
      LIVE
    </div>
  </div>

  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-num">{stats.tracked}</div>
      <div class="stat-label">Tracked Objects</div>
      <div class="stat-bar" style="--pct:85%; --c:var(--accent)"></div>
    </div>
    <div class="stat-card">
      <div class="stat-num safe">{stats.active_sats}</div>
      <div class="stat-label">Active Satellites</div>
      <div class="stat-bar" style="--pct:40%; --c:var(--safe)"></div>
    </div>
    <div class="stat-card danger-card">
      <div class="stat-num danger">{stats.high_risk}</div>
      <div class="stat-label">High Risk Alerts</div>
      <div class="stat-bar" style="--pct:20%; --c:var(--danger)"></div>
    </div>
    <div class="stat-card">
      <div class="stat-num gold">{stats.avg_distance}<span class="unit"> km</span></div>
      <div class="stat-label">Avg Miss Distance</div>
      <div class="stat-bar" style="--pct:60%; --c:var(--gold)"></div>
    </div>
  </div>

  <div class="section">
    <div class="section-head">
      <span class="section-title">Active Conjunctions</span>
      <span class="section-count">{conjunctions.length} events</span>
    </div>
    <table class="conj-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>OBJECT 1</th>
          <th>OBJECT 2</th>
          <th>DIST</th>
          <th>VEL</th>
          <th>TCA</th>
          <th>RISK</th>
        </tr>
      </thead>
      <tbody>
        {#each conjunctions as c}
          <tr class="conj-row" class:high-row={c.risk === 'HIGH'}>
            <td class="mono dim">{c.id}</td>
            <td class="mono">{c.object1}</td>
            <td class="mono">{c.object2}</td>
            <td class="mono">{c.distance} km</td>
            <td class="mono">{c.velocity}</td>
            <td class="mono dim">{c.time}</td>
            <td><span class="pill {riskCls(c.risk)}">{c.risk}</span></td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  {#if $selectedObject}
    <div class="section">
      <div class="section-head">
        <span class="section-title">Selected Object</span>
      </div>
      <div class="obj-detail">
        {#each [
          ['LABEL',    $selectedObject.label],
          ['POS X',    $selectedObject.x.toFixed(4)],
          ['POS Y',    $selectedObject.y.toFixed(4)],
          ['POS Z',    $selectedObject.z.toFixed(4)],
          ['RISK IDX', ($selectedObject.risk*100).toFixed(1)+'%']
        ] as [k,v]}
          <div class="obj-row">
            <span class="obj-key">{k}</span>
            <span class="obj-val">{v}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

</div>

<style>
  .panel {
    flex: 1; padding: 28px;
    overflow-y: auto;
    display: flex; flex-direction: column; gap: 24px;
    background: var(--panel-bg);
  }

  .panel-header {
    display: flex; align-items: flex-start;
    justify-content: space-between;
  }

  .header-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim);
    letter-spacing: 0.22em; text-transform: uppercase;
    margin-bottom: 6px;
  }

  .header-title {
    font-family: 'Limelight', cursive;
    font-size: 22px; color: var(--text);
    letter-spacing: 0.04em; line-height: 1.15;
    font-weight: 400;
  }

  .live-badge {
    display: flex; align-items: center; gap: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--safe);
    letter-spacing: 0.18em;
    border: 1px solid rgba(0,232,160,0.25);
    padding: 5px 12px;
  }
  .live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--safe);
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.2} }

  .stat-grid {
    display: grid; grid-template-columns: repeat(4,1fr); gap: 12px;
  }

  .stat-card {
    background: var(--glass);
    border: 1px solid var(--border-dim);
    padding: 18px 16px 14px;
    position: relative; overflow: hidden;
    transition: border-color 0.2s;
  }
  .stat-card:hover { border-color: var(--border); }
  .stat-card.danger-card { border-color: rgba(255,56,96,0.15); }

  .stat-num {
    font-family: 'Limelight', cursive;
    font-size: 28px; color: var(--accent-hi);
    line-height: 1; margin-bottom: 6px;
  }
  .stat-num.safe   { color: var(--safe); }
  .stat-num.danger { color: var(--danger); }
  .stat-num.gold   { color: var(--gold); }
  .unit { font-size: 13px; }

  .stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim);
    letter-spacing: 0.14em; text-transform: uppercase;
    margin-bottom: 12px;
  }

  .stat-bar {
    position: absolute; bottom: 0; left: 0;
    width: var(--pct); height: 2px;
    background: var(--c);
    transition: width 1s ease;
  }

  .section {
    border: 1px solid var(--border-dim);
    overflow: hidden;
  }

  .section-head {
    display: flex; justify-content: space-between; align-items: center;
    padding: 11px 16px;
    border-bottom: 1px solid var(--border-dim);
    background: var(--glass);
  }

  .section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--accent-hi);
    letter-spacing: 0.18em; text-transform: uppercase;
  }

  .section-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim);
  }

  .conj-table { width: 100%; border-collapse: collapse; }

  .conj-table th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; letter-spacing: 0.15em;
    color: var(--text-dim);
    padding: 8px 12px; text-align: left;
    border-bottom: 1px solid var(--border-dim);
    text-transform: uppercase;
  }

  .conj-row td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border-dim);
    transition: background 0.15s;
  }
  .conj-row:last-child td { border-bottom: none; }
  .conj-row:hover td { background: var(--glass); }
  .conj-row.high-row td { background: rgba(255,56,96,0.03); }

  .mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--text);
  }
  .dim { color: var(--text-dim) !important; }

  .pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 500;
    letter-spacing: 0.1em; padding: 2px 9px;
    text-transform: uppercase;
  }
  .pill.high   { background: rgba(255,56,96,0.12);  color: var(--danger);  border: 1px solid rgba(255,56,96,0.3); }
  .pill.med    { background: rgba(255,144,32,0.12); color: var(--warning); border: 1px solid rgba(255,144,32,0.3); }
  .pill.low    { background: rgba(0,232,160,0.08);  color: var(--safe);    border: 1px solid rgba(0,232,160,0.25); }

  .obj-detail { padding: 12px 16px; display: flex; flex-direction: column; gap: 6px; }
  .obj-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid var(--border-dim);
  }
  .obj-row:last-child { border-bottom: none; }
  .obj-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim); letter-spacing: 0.15em;
  }
  .obj-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--text);
  }
</style>
