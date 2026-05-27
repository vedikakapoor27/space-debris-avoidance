<script>
  import { onMount, onDestroy } from 'svelte'
  import { getMockConjunctions } from '../utils/api.js'
  import { selectedObject } from '../stores/appStore.js'

  let conjunctions = getMockConjunctions()
  let tick = 0
  let interval

  // Simulated live stats
  let stats = {
    tracked:     847,
    active_sats: 12,
    high_risk:   2,
    avg_distance: 156.4
  }

  // Simulate slight fluctuation for "live" feel
  onMount(() => {
    interval = setInterval(() => {
      stats.avg_distance = +(154 + Math.random() * 6).toFixed(1)
      tick++
    }, 2000)
  })
  onDestroy(() => clearInterval(interval))

  const riskClass = r => r === 'HIGH' ? 'high' : r === 'MEDIUM' ? 'med' : 'low'
</script>

<div class="panel">

  <!-- Header -->
  <div class="panel-header">
    <div>
      <h2 class="panel-title">MISSION OVERVIEW</h2>
      <p class="panel-sub">Real-time orbital collision monitoring</p>
    </div>
    <div class="live-badge">
      <span class="live-dot"></span>
      LIVE
    </div>
  </div>

  <!-- Stat Cards -->
  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-icon blue">◈</div>
      <div class="stat-val">{stats.tracked}</div>
      <div class="stat-key">TRACKED OBJECTS</div>
    </div>
    <div class="stat-card">
      <div class="stat-icon green">◉</div>
      <div class="stat-val">{stats.active_sats}</div>
      <div class="stat-key">ACTIVE SATELLITES</div>
    </div>
    <div class="stat-card danger">
      <div class="stat-icon red">⚠</div>
      <div class="stat-val" style="color:#ff2244">{stats.high_risk}</div>
      <div class="stat-key">HIGH RISK ALERTS</div>
    </div>
    <div class="stat-card">
      <div class="stat-icon orange">⬡</div>
      <div class="stat-val">{stats.avg_distance} <span class="unit">km</span></div>
      <div class="stat-key">AVG MISS DISTANCE</div>
    </div>
  </div>

  <!-- Conjunction Events -->
  <div class="section">
    <div class="section-header">
      <span class="section-title">⚡ ACTIVE CONJUNCTIONS</span>
      <span class="section-count">{conjunctions.length} events</span>
    </div>
    <table class="conj-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>OBJECT 1</th>
          <th>OBJECT 2</th>
          <th>DIST (km)</th>
          <th>VEL (km/s)</th>
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
            <td class="mono">{c.distance}</td>
            <td class="mono">{c.velocity}</td>
            <td class="mono dim">{c.time}</td>
            <td>
              <span class="risk-pill {riskClass(c.risk)}">{c.risk}</span>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <!-- Selected Object Info -->
  {#if $selectedObject}
    <div class="section">
      <div class="section-header">
        <span class="section-title">◎ SELECTED OBJECT</span>
      </div>
      <div class="obj-detail">
        <div class="obj-row"><span>ID</span><span>{$selectedObject.label}</span></div>
        <div class="obj-row"><span>POSITION X</span><span>{$selectedObject.x.toFixed(4)}</span></div>
        <div class="obj-row"><span>POSITION Y</span><span>{$selectedObject.y.toFixed(4)}</span></div>
        <div class="obj-row"><span>POSITION Z</span><span>{$selectedObject.z.toFixed(4)}</span></div>
        <div class="obj-row">
          <span>RISK INDEX</span>
          <span class="risk-pill {riskClass($selectedObject.risk > 0.7 ? 'HIGH' : $selectedObject.risk > 0.3 ? 'MEDIUM' : 'LOW')}">
            {($selectedObject.risk * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  {/if}

</div>

<style>
  .panel {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .panel-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
  }
  .panel-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #00e5ff;
    letter-spacing: 4px;
    margin: 0;
  }
  .panel-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: rgba(255,255,255,0.3);
    margin: 4px 0 0;
  }
  .live-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'Orbitron', sans-serif;
    font-size: 10px;
    font-weight: 700;
    color: #00ff88;
    letter-spacing: 2px;
    border: 1px solid rgba(0,255,136,0.3);
    padding: 4px 10px;
    border-radius: 3px;
  }
  .live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 6px #00ff88;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.2} }

  /* Stat grid */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }
  .stat-card {
    background: rgba(0,229,255,0.03);
    border: 1px solid rgba(0,229,255,0.12);
    border-radius: 6px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    transition: border-color 0.2s;
  }
  .stat-card:hover { border-color: rgba(0,229,255,0.35); }
  .stat-card.danger { border-color: rgba(255,34,68,0.2); background: rgba(255,34,68,0.03); }
  .stat-icon { font-size: 18px; }
  .stat-icon.blue  { color: #00e5ff; text-shadow: 0 0 10px #00e5ff; }
  .stat-icon.green { color: #00ff88; text-shadow: 0 0 10px #00ff88; }
  .stat-icon.red   { color: #ff2244; text-shadow: 0 0 10px #ff2244; }
  .stat-icon.orange{ color: #ff8800; text-shadow: 0 0 10px #ff8800; }
  .stat-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #fff;
  }
  .unit { font-size: 11px; color: rgba(255,255,255,0.4); }
  .stat-key {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: rgba(255,255,255,0.35);
    letter-spacing: 1.5px;
  }

  /* Sections */
  .section {
    background: rgba(0,229,255,0.02);
    border: 1px solid rgba(0,229,255,0.1);
    border-radius: 6px;
    overflow: hidden;
  }
  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid rgba(0,229,255,0.08);
    background: rgba(0,229,255,0.04);
  }
  .section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: #00e5ff;
    letter-spacing: 3px;
  }
  .section-count {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.3);
  }

  /* Table */
  .conj-table {
    width: 100%;
    border-collapse: collapse;
  }
  .conj-table th {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    letter-spacing: 1.5px;
    color: rgba(0,229,255,0.5);
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid rgba(0,229,255,0.06);
  }
  .conj-row td {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    transition: background 0.15s;
  }
  .conj-row:hover td { background: rgba(0,229,255,0.04); }
  .conj-row.high-row td { background: rgba(255,34,68,0.04); }
  .mono {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: rgba(255,255,255,0.75);
  }
  .dim { color: rgba(255,255,255,0.35) !important; }

  .risk-pill {
    font-family: 'Rajdhani', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 2px 8px;
    border-radius: 3px;
  }
  .risk-pill.high   { background: rgba(255,34,68,0.2);  color: #ff2244; border: 1px solid rgba(255,34,68,0.4); }
  .risk-pill.med    { background: rgba(255,136,0,0.2);  color: #ff8800; border: 1px solid rgba(255,136,0,0.4); }
  .risk-pill.low    { background: rgba(0,229,255,0.1);  color: #00e5ff; border: 1px solid rgba(0,229,255,0.3); }

  /* Object detail */
  .obj-detail { padding: 12px 16px; display: flex; flex-direction: column; gap: 8px; }
  .obj-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .obj-row span:first-child {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.35);
    letter-spacing: 1.5px;
  }
  .obj-row span:last-child {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: rgba(255,255,255,0.8);
  }
</style>
