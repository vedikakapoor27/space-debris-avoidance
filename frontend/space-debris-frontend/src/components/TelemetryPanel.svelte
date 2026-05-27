<script>
  import { onMount, onDestroy } from 'svelte'
  import { telemetryFeed, pushTelemetry } from '../stores/appStore.js'
  import { predict } from '../utils/api.js'

  // Auto-simulate telemetry events every 4 seconds
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
        // backend offline, push simulated
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

  const riskClass = r => r === 'HIGH' ? 'high' : r === 'MEDIUM' ? 'med' : 'low'
  const fmtTime = iso => iso ? new Date(iso).toTimeString().slice(0,8) : '--:--:--'
</script>

<div class="panel">
  <div class="panel-header">
    <div>
      <h2 class="panel-title">TELEMETRY FEED</h2>
      <p class="panel-sub">Live prediction stream — auto-polling every 4 seconds</p>
    </div>
    <div class="live-badge">
      <span class="live-dot"></span>
      AUTO-POLLING
    </div>
  </div>

  <div class="feed-container">
    {#if $telemetryFeed.length === 0}
      <div class="empty">
        <div class="empty-icon">⎍</div>
        <p>Awaiting telemetry data...</p>
      </div>
    {:else}
      {#each $telemetryFeed as entry, i}
        <div class="feed-entry" class:new-entry={i === 0} class:high={entry.risk_level === 'HIGH'}>
          <div class="entry-time">{fmtTime(entry.ts)}</div>
          <div class="entry-object">{entry.object ?? 'UNKNOWN'}</div>
          <div class="entry-stats">
            <span>D: {entry.input?.distance_km ?? '?'} km</span>
            <span>V: {entry.input?.rel_velocity ?? '?'} km/s</span>
            <span>A: {entry.input?.approach_rate ?? '?'}</span>
          </div>
          <div class="entry-prob">{entry.probability}%</div>
          <span class="risk-pill {riskClass(entry.risk_level)}">{entry.risk_level}</span>
          <div class="entry-action">{entry.action}</div>
          <span class="urgency {riskClass(entry.risk_level)}">{entry.urgency}</span>
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  .panel {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
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
    animation: pulse 1s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.2} }

  .feed-container {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .feed-entry {
    display: grid;
    grid-template-columns: 70px 100px 1fr 55px 80px 1fr 80px;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: rgba(0,229,255,0.02);
    border: 1px solid rgba(0,229,255,0.08);
    border-radius: 4px;
    transition: all 0.3s;
  }
  .feed-entry.high { border-color: rgba(255,34,68,0.2); background: rgba(255,34,68,0.03); }
  .feed-entry.new-entry { animation: slideIn 0.4s ease-out; border-color: rgba(0,229,255,0.25); }
  @keyframes slideIn {
    from { opacity: 0; transform: translateX(-10px); }
    to   { opacity: 1; transform: translateX(0); }
  }

  .entry-time {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.3);
  }
  .entry-object {
    font-family: 'Orbitron', sans-serif;
    font-size: 10px;
    color: #00e5ff;
    font-weight: 600;
  }
  .entry-stats {
    display: flex;
    gap: 10px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.4);
  }
  .entry-prob {
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    font-weight: 700;
    color: #fff;
    text-align: center;
  }
  .risk-pill {
    font-family: 'Rajdhani', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 2px 8px;
    border-radius: 3px;
    text-align: center;
  }
  .risk-pill.high  { background: rgba(255,34,68,0.2); color: #ff2244; border: 1px solid rgba(255,34,68,0.4); }
  .risk-pill.med   { background: rgba(255,136,0,0.2); color: #ff8800; border: 1px solid rgba(255,136,0,0.4); }
  .risk-pill.low   { background: rgba(0,229,255,0.1); color: #00e5ff; border: 1px solid rgba(0,229,255,0.3); }
  .entry-action {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.5);
  }
  .urgency {
    font-family: 'Rajdhani', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-align: right;
  }
  .urgency.high  { color: #ff2244; }
  .urgency.med   { color: #ff8800; }
  .urgency.low   { color: #00ff88; }

  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 60px;
    opacity: 0.3;
  }
  .empty-icon { font-size: 40px; color: #00e5ff; }
  .empty p {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: rgba(255,255,255,0.4);
    margin: 0;
  }
</style>
