<script>
  import { onMount, onDestroy } from 'svelte'
  import { telemetryFeed, pushTelemetry } from '../stores/appStore.js'
  import { predict } from '../utils/api.js'

  let interval
  const objects = ['ISS', 'SAT-1001', 'SAT-1002', 'Starlink-12', 'CosmosSat-3']

  onMount(() => {
    interval = setInterval(async () => {
      const d = +(5 + Math.random() * 200).toFixed(1)
      const v = +(1 + Math.random() * 14).toFixed(1)
      const a = +((Math.random() * 20 - 10)).toFixed(1)
      try {
        const r = await predict({ distance_km: d, rel_velocity: v, approach_rate: a })
        pushTelemetry({ ...r, object: objects[Math.floor(Math.random() * objects.length)] })
      } catch {
        const risk_level = d < 20 ? 'HIGH' : d < 80 ? 'MEDIUM' : 'LOW'
        pushTelemetry({
          risk_level,
          probability: (Math.random() * 100).toFixed(1),
          action: risk_level === 'HIGH' ? 'MANEUVER REQUIRED' : risk_level === 'MEDIUM' ? 'MONITOR CLOSELY' : 'NO ACTION NEEDED',
          urgency: risk_level === 'HIGH' ? 'IMMEDIATE' : risk_level === 'MEDIUM' ? 'WATCH' : 'CLEAR',
          object: objects[Math.floor(Math.random() * objects.length)],
          input: { distance_km: d, rel_velocity: v, approach_rate: a }
        })
      }
    }, 4000)
  })
  onDestroy(() => clearInterval(interval))

  const riskCls = r => r === 'HIGH' ? 'high' : r === 'MEDIUM' ? 'med' : 'low'
  const fmtTime = iso => iso ? new Date(iso).toTimeString().slice(0,8) : '--:--:--'
</script>

<div class="panel">

  <div class="panel-header">
    <div>
      <div class="header-eyebrow">LIVE DATA STREAM</div>
      <h2 class="header-title">Telemetry<br>Feed</h2>
    </div>
    <div class="polling-badge">
      <span class="poll-dot"></span>
      AUTO-POLLING · 4s
    </div>
  </div>

  <div class="feed">
    {#if $telemetryFeed.length === 0}
      <div class="empty">
        <div class="empty-icon">⎍</div>
        <p>Awaiting telemetry data...</p>
      </div>
    {:else}
      {#each $telemetryFeed as entry, i}
        <div class="entry" class:new={i===0} class:is-high={entry.risk_level==='HIGH'}>
          <div class="entry-time">{fmtTime(entry.ts)}</div>
          <div class="entry-obj">{entry.object ?? 'UNKNOWN'}</div>
          <div class="entry-params">
            <span>D: {entry.input?.distance_km ?? '?'} km</span>
            <span>V: {entry.input?.rel_velocity ?? '?'}</span>
            <span>A: {entry.input?.approach_rate ?? '?'}</span>
          </div>
          <div class="entry-prob">{entry.probability}%</div>
          <span class="pill {riskCls(entry.risk_level)}">{entry.risk_level}</span>
          <div class="entry-action">{entry.action}</div>
          <span class="urgency {riskCls(entry.risk_level)}">{entry.urgency}</span>
        </div>
      {/each}
    {/if}
  </div>

</div>

<style>
  .panel {
    flex: 1; padding: 28px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 22px;
    background: var(--panel-bg);
  }

  .panel-header { display: flex; justify-content: space-between; align-items: flex-start; }

  .header-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim);
    letter-spacing: 0.22em; text-transform: uppercase; margin-bottom: 6px;
  }

  .header-title {
    font-family: 'Limelight', cursive;
    font-size: 22px; color: var(--text);
    letter-spacing: 0.04em; line-height: 1.15; font-weight: 400;
  }

  .polling-badge {
    display: flex; align-items: center; gap: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--violet);
    letter-spacing: 0.14em;
    border: 1px solid rgba(192,132,252,0.22);
    padding: 5px 12px;
  }

  .poll-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--violet);
    animation: pulse 1s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.2} }

  .feed { display: flex; flex-direction: column; gap: 5px; }

  .entry {
    display: grid;
    grid-template-columns: 70px 100px 1fr 50px 75px 1fr 80px;
    align-items: center; gap: 10px;
    padding: 10px 14px;
    background: var(--glass);
    border: 1px solid var(--border-dim);
    transition: all 0.3s;
  }

  .entry.is-high { border-color: rgba(255,56,96,0.18); background: rgba(255,56,96,0.03); }

  .entry.new { animation: slideIn 0.4s ease-out; border-color: var(--border); }
  @keyframes slideIn { from{opacity:0;transform:translateX(-10px)} to{opacity:1;transform:translateX(0)} }

  .entry-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--text-dim);
  }

  .entry-obj {
    font-family: 'Space Grotesk', monospace;
    font-size: 11px; font-weight: 600; color: var(--gold);
  }

  .entry-params {
    display: flex; gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim);
  }

  .entry-prob {
    font-family: 'Limelight', cursive;
    font-size: 13px; color: var(--text); text-align: center;
  }

  .pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; letter-spacing: 0.1em; padding: 2px 7px; text-align: center;
  }
  .pill.high { background: rgba(255,56,96,0.1);  color: var(--danger);  border: 1px solid rgba(255,56,96,0.25); }
  .pill.med  { background: rgba(255,144,32,0.1); color: var(--warning); border: 1px solid rgba(255,144,32,0.25); }
  .pill.low  { background: rgba(0,232,160,0.07); color: var(--safe);    border: 1px solid rgba(0,232,160,0.2); }

  .entry-action {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim);
  }

  .urgency {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.12em; text-align: right;
    text-transform: uppercase;
  }
  .urgency.high { color: var(--danger); }
  .urgency.med  { color: var(--warning); }
  .urgency.low  { color: var(--safe); }

  .empty {
    display: flex; flex-direction: column;
    align-items: center; gap: 12px; padding: 60px; opacity: 0.25;
  }
  .empty-icon { font-size: 40px; color: var(--accent); }
  .empty p {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px; color: var(--text-dim); margin: 0;
  }
</style>
