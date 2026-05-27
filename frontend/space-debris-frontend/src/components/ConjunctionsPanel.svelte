<script>
  import { getMockConjunctions } from '../utils/api.js'
  import { formValues, activePanel } from '../stores/appStore.js'

  let events = getMockConjunctions()
  let selected = null

  function selectEvent(c) {
    selected = c
  }

  function sendToPredict(c) {
    // Pre-fill predict form with this conjunction's data
    formValues.set({
      distance_km:   c.distance,
      rel_velocity:  c.velocity,
      approach_rate: -(c.velocity * 0.6)
    })
    activePanel.set('predict')
  }

  const riskClass = r => r === 'HIGH' ? 'high' : r === 'MEDIUM' ? 'med' : 'low'
</script>

<div class="panel">
  <div class="panel-header">
    <div>
      <h2 class="panel-title">CONJUNCTION EVENTS</h2>
      <p class="panel-sub">Closest approach analysis — Time of Closest Approach (TCA)</p>
    </div>
  </div>

  <div class="stats-row">
    <div class="mini-stat red">
      <span class="mval">{events.filter(e=>e.risk==='HIGH').length}</span>
      <span class="mkey">CRITICAL</span>
    </div>
    <div class="mini-stat orange">
      <span class="mval">{events.filter(e=>e.risk==='MEDIUM').length}</span>
      <span class="mkey">WARNING</span>
    </div>
    <div class="mini-stat blue">
      <span class="mval">{events.filter(e=>e.risk==='LOW').length}</span>
      <span class="mkey">NOMINAL</span>
    </div>
    <div class="mini-stat white">
      <span class="mval">{events.length}</span>
      <span class="mkey">TOTAL</span>
    </div>
  </div>

  <div class="split">
    <!-- Event List -->
    <div class="event-list">
      {#each events as c}
        <button
          class="event-card"
          class:selected={selected?.id === c.id}
          class:high-card={c.risk === 'HIGH'}
          on:click={() => selectEvent(c)}
        >
          <div class="ec-top">
            <span class="ec-id">{c.id}</span>
            <span class="risk-pill {riskClass(c.risk)}">{c.risk}</span>
          </div>
          <div class="ec-objs">
            <span>{c.object1}</span>
            <span class="ec-arrow">↔</span>
            <span>{c.object2}</span>
          </div>
          <div class="ec-bottom">
            <span>TCA: <strong>{c.time}</strong></span>
            <span>{c.distance} km</span>
          </div>
        </button>
      {/each}
    </div>

    <!-- Detail View -->
    <div class="detail-view">
      {#if selected}
        <div class="detail-header">
          <span class="detail-id">{selected.id}</span>
          <span class="risk-pill {riskClass(selected.risk)}">{selected.risk}</span>
        </div>

        <div class="detail-body">
          <div class="obj-pair">
            <div class="obj-box blue">
              <div class="obj-type">PRIMARY</div>
              <div class="obj-name">{selected.object1}</div>
            </div>
            <div class="vs-icon">⟺</div>
            <div class="obj-box orange">
              <div class="obj-type">SECONDARY</div>
              <div class="obj-name">{selected.object2}</div>
            </div>
          </div>

          <div class="detail-stats">
            <div class="ds-row">
              <span>MISS DISTANCE</span>
              <span class="ds-val">{selected.distance} km</span>
            </div>
            <div class="ds-row">
              <span>RELATIVE VELOCITY</span>
              <span class="ds-val">{selected.velocity} km/s</span>
            </div>
            <div class="ds-row">
              <span>TIME OF CLOSEST APPROACH</span>
              <span class="ds-val">{selected.time}</span>
            </div>
            <div class="ds-row">
              <span>RISK LEVEL</span>
              <span class="risk-pill {riskClass(selected.risk)}">{selected.risk}</span>
            </div>
          </div>

          <button class="analyze-btn" on:click={() => sendToPredict(selected)}>
            ⟁ ANALYZE IN RISK PREDICTOR
          </button>
        </div>
      {:else}
        <div class="detail-empty">
          <div class="empty-icon">◎</div>
          <p>Select an event to inspect</p>
        </div>
      {/if}
    </div>
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

  .stats-row {
    display: flex;
    gap: 12px;
  }
  .mini-stat {
    flex: 1;
    background: rgba(0,229,255,0.02);
    border: 1px solid rgba(0,229,255,0.1);
    border-radius: 6px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }
  .mini-stat.red   { border-color: rgba(255,34,68,0.2); }
  .mini-stat.orange{ border-color: rgba(255,136,0,0.2); }
  .mval {
    font-family: 'Orbitron', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: #fff;
  }
  .mini-stat.red   .mval { color: #ff2244; }
  .mini-stat.orange .mval { color: #ff8800; }
  .mkey {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: rgba(255,255,255,0.3);
    letter-spacing: 1.5px;
  }

  .split {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 16px;
    flex: 1;
    min-height: 0;
  }

  .event-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    overflow-y: auto;
  }
  .event-card {
    background: rgba(0,229,255,0.02);
    border: 1px solid rgba(0,229,255,0.1);
    border-radius: 6px;
    padding: 12px;
    cursor: pointer;
    text-align: left;
    display: flex;
    flex-direction: column;
    gap: 6px;
    transition: all 0.2s;
    width: 100%;
  }
  .event-card:hover { border-color: rgba(0,229,255,0.3); background: rgba(0,229,255,0.05); }
  .event-card.selected { border-color: #00e5ff; background: rgba(0,229,255,0.08); }
  .event-card.high-card { border-color: rgba(255,34,68,0.2); }
  .event-card.high-card:hover { border-color: rgba(255,34,68,0.5); }

  .ec-top { display: flex; justify-content: space-between; align-items: center; }
  .ec-id {
    font-family: 'Orbitron', sans-serif;
    font-size: 10px;
    color: rgba(255,255,255,0.4);
    letter-spacing: 1px;
  }
  .ec-objs {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: rgba(255,255,255,0.8);
  }
  .ec-arrow { color: rgba(255,255,255,0.2); }
  .ec-bottom {
    display: flex;
    justify-content: space-between;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.35);
  }
  .ec-bottom strong { color: rgba(255,255,255,0.65); }

  .risk-pill {
    font-family: 'Rajdhani', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 2px 8px;
    border-radius: 3px;
  }
  .risk-pill.high  { background: rgba(255,34,68,0.2); color: #ff2244; border: 1px solid rgba(255,34,68,0.4); }
  .risk-pill.med   { background: rgba(255,136,0,0.2); color: #ff8800; border: 1px solid rgba(255,136,0,0.4); }
  .risk-pill.low   { background: rgba(0,229,255,0.1); color: #00e5ff; border: 1px solid rgba(0,229,255,0.3); }

  /* Detail */
  .detail-view {
    background: rgba(0,229,255,0.02);
    border: 1px solid rgba(0,229,255,0.1);
    border-radius: 8px;
    overflow: hidden;
  }
  .detail-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 18px;
    border-bottom: 1px solid rgba(0,229,255,0.08);
    background: rgba(0,229,255,0.04);
  }
  .detail-id {
    font-family: 'Orbitron', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: #00e5ff;
    letter-spacing: 2px;
  }
  .detail-body {
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .obj-pair {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .obj-box {
    flex: 1;
    padding: 14px;
    border-radius: 6px;
    border: 1px solid;
    text-align: center;
  }
  .obj-box.blue  { border-color: rgba(0,229,255,0.3); background: rgba(0,229,255,0.05); }
  .obj-box.orange{ border-color: rgba(255,136,0,0.3);  background: rgba(255,136,0,0.05); }
  .obj-type {
    font-family: 'Share Tech Mono', monospace;
    font-size: 8px;
    color: rgba(255,255,255,0.3);
    letter-spacing: 2px;
    margin-bottom: 6px;
  }
  .obj-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: #fff;
  }
  .vs-icon {
    font-size: 20px;
    color: rgba(255,255,255,0.2);
  }

  .detail-stats {
    display: flex;
    flex-direction: column;
    gap: 0;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 6px;
    overflow: hidden;
  }
  .ds-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .ds-row span:first-child {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.3);
    letter-spacing: 1px;
  }
  .ds-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: #fff;
  }

  .analyze-btn {
    padding: 12px;
    background: rgba(0,229,255,0.07);
    border: 1px solid rgba(0,229,255,0.3);
    border-radius: 4px;
    color: #00e5ff;
    font-family: 'Orbitron', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    cursor: pointer;
    transition: all 0.2s;
    width: 100%;
  }
  .analyze-btn:hover {
    background: rgba(0,229,255,0.15);
    box-shadow: 0 0 16px rgba(0,229,255,0.15);
  }

  .detail-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 12px;
    opacity: 0.25;
  }
  .empty-icon { font-size: 48px; color: #00e5ff; }
  .detail-empty p {
    font-family: 'Rajdhani', sans-serif;
    font-size: 14px;
    color: rgba(255,255,255,0.5);
    margin: 0;
  }
</style>
