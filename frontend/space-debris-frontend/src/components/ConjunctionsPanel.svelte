<script>
  import { getMockConjunctions } from '../utils/api.js'
  import { formValues, activePanel } from '../stores/appStore.js'

  let events = getMockConjunctions()
  let selected = null

  function sendToPredict(c) {
    formValues.set({ distance_km: c.distance, rel_velocity: c.velocity, approach_rate: -(c.velocity * 0.6) })
    activePanel.set('predict')
  }

  const riskCls = r => r === 'HIGH' ? 'high' : r === 'MEDIUM' ? 'med' : 'low'
</script>

<div class="panel">

  <div class="panel-header">
    <div class="header-eyebrow">TCA ANALYSIS</div>
    <h2 class="header-title">Conjunction<br>Events</h2>
  </div>

  <div class="mini-stats">
    <div class="mstat danger">
      <span class="mval">{events.filter(e=>e.risk==='HIGH').length}</span>
      <span class="mkey">Critical</span>
    </div>
    <div class="mstat warn">
      <span class="mval">{events.filter(e=>e.risk==='MEDIUM').length}</span>
      <span class="mkey">Warning</span>
    </div>
    <div class="mstat safe">
      <span class="mval">{events.filter(e=>e.risk==='LOW').length}</span>
      <span class="mkey">Nominal</span>
    </div>
    <div class="mstat">
      <span class="mval">{events.length}</span>
      <span class="mkey">Total</span>
    </div>
  </div>

  <div class="split">
    <div class="event-list">
      {#each events as c}
        <button
          class="event-card"
          class:active={selected?.id === c.id}
          class:is-high={c.risk === 'HIGH'}
          on:click={() => selected = c}
        >
          <div class="ec-top">
            <span class="ec-id">{c.id}</span>
            <span class="pill {riskCls(c.risk)}">{c.risk}</span>
          </div>
          <div class="ec-pair">
            <span>{c.object1}</span>
            <span class="ec-sep">↔</span>
            <span>{c.object2}</span>
          </div>
          <div class="ec-bottom">
            <span>TCA {c.time}</span>
            <span class="ec-dist">{c.distance} km</span>
          </div>
        </button>
      {/each}
    </div>

    <div class="detail">
      {#if selected}
        <div class="detail-head">
          <span class="detail-id">{selected.id}</span>
          <span class="pill {riskCls(selected.risk)}">{selected.risk}</span>
        </div>
        <div class="detail-body">
          <div class="obj-pair">
            <div class="obj-box primary">
              <div class="obj-type">PRIMARY</div>
              <div class="obj-name">{selected.object1}</div>
            </div>
            <div class="obj-vs">⟺</div>
            <div class="obj-box secondary">
              <div class="obj-type">SECONDARY</div>
              <div class="obj-name">{selected.object2}</div>
            </div>
          </div>
          <div class="detail-rows">
            {#each [
              ['Miss Distance',  selected.distance + ' km'],
              ['Rel Velocity',   selected.velocity + ' km/s'],
              ['Time of CPA',    selected.time],
              ['Risk Level',     selected.risk]
            ] as [k,v]}
              <div class="drow">
                <span class="dkey">{k}</span>
                <span class="dval">{v}</span>
              </div>
            {/each}
          </div>
          <button class="analyze-btn" on:click={() => sendToPredict(selected)}>
            ⟁ Analyze in Risk Predictor
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
    flex: 1; padding: 28px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 22px;
    background: var(--panel-bg);
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
    letter-spacing: 0.04em; line-height: 1.15; font-weight: 400;
  }

  .mini-stats { display: flex; gap: 10px; }

  .mstat {
    flex: 1; border: 1px solid var(--border-dim);
    background: var(--glass); padding: 14px 12px;
    display: flex; flex-direction: column; align-items: center; gap: 4px;
  }
  .mstat.danger { border-color: rgba(255,56,96,0.18); }
  .mstat.warn   { border-color: rgba(255,144,32,0.18); }
  .mstat.safe   { border-color: rgba(0,232,160,0.15); }

  .mval {
    font-family: 'Limelight', cursive; font-size: 26px;
    color: var(--accent-hi);
  }
  .mstat.danger .mval { color: var(--danger); }
  .mstat.warn   .mval { color: var(--warning); }
  .mstat.safe   .mval { color: var(--safe); }

  .mkey {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim);
    letter-spacing: 0.14em; text-transform: uppercase;
  }

  .split {
    display: grid; grid-template-columns: 260px 1fr;
    gap: 16px; flex: 1; min-height: 0;
  }

  .event-list {
    display: flex; flex-direction: column; gap: 7px; overflow-y: auto;
  }

  .event-card {
    background: var(--glass); border: 1px solid var(--border-dim);
    padding: 12px; cursor: pointer; text-align: left;
    display: flex; flex-direction: column; gap: 6px;
    transition: all 0.2s; width: 100%;
  }
  .event-card:hover { border-color: var(--border); }
  .event-card.active { border-color: var(--gold); background: rgba(232,184,75,0.05); }
  .event-card.is-high { border-color: rgba(255,56,96,0.2); }

  .ec-top { display: flex; justify-content: space-between; align-items: center; }

  .ec-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim); letter-spacing: 0.1em;
  }

  .ec-pair {
    display: flex; align-items: center; gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--text);
  }
  .ec-sep { color: var(--text-dim); }

  .ec-bottom {
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim);
  }
  .ec-dist { color: var(--accent-hi); }

  .pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; letter-spacing: 0.1em; padding: 2px 8px;
  }
  .pill.high { background: rgba(255,56,96,0.1);  color: var(--danger);  border: 1px solid rgba(255,56,96,0.25); }
  .pill.med  { background: rgba(255,144,32,0.1); color: var(--warning); border: 1px solid rgba(255,144,32,0.25); }
  .pill.low  { background: rgba(0,232,160,0.07); color: var(--safe);    border: 1px solid rgba(0,232,160,0.2); }

  .detail {
    border: 1px solid var(--border-dim);
    background: var(--glass); overflow: hidden;
    display: flex; flex-direction: column;
  }

  .detail-head {
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 18px; border-bottom: 1px solid var(--border-dim);
    background: var(--glass);
  }

  .detail-id {
    font-family: 'Limelight', cursive;
    font-size: 15px; color: var(--gold); letter-spacing: 0.1em;
  }

  .detail-body {
    padding: 20px; display: flex; flex-direction: column; gap: 18px; flex: 1;
  }

  .obj-pair { display: flex; align-items: center; gap: 10px; }

  .obj-box {
    flex: 1; padding: 12px; border: 1px solid; text-align: center;
  }
  .obj-box.primary   { border-color: rgba(139,92,246,0.3); background: rgba(139,92,246,0.05); }
  .obj-box.secondary { border-color: rgba(232,184,75,0.3); background: rgba(232,184,75,0.04); }

  .obj-type {
    font-family: 'JetBrains Mono', monospace;
    font-size: 7px; color: var(--text-dim);
    letter-spacing: 0.18em; margin-bottom: 6px;
  }

  .obj-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px; font-weight: 600; color: var(--text);
  }

  .obj-vs { font-size: 18px; color: var(--text-dim); }

  .detail-rows {
    border: 1px solid var(--border-dim); overflow: hidden;
  }

  .drow {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 14px; border-bottom: 1px solid var(--border-dim);
  }
  .drow:last-child { border-bottom: none; }

  .dkey {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim); letter-spacing: 0.12em;
  }

  .dval {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--text);
  }

  .analyze-btn {
    padding: 12px; background: var(--glass);
    border: 1px solid var(--accent); color: var(--accent-hi);
    font-family: 'Limelight', cursive; font-size: 11px;
    letter-spacing: 0.14em; cursor: pointer; transition: all 0.2s; width: 100%;
  }
  .analyze-btn:hover {
    background: rgba(139,92,246,0.1);
    box-shadow: var(--glow-accent);
  }

  .detail-empty {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 12px; opacity: 0.2;
  }
  .empty-icon { font-size: 44px; color: var(--accent); }
  .detail-empty p {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px; color: var(--text-dim); margin: 0;
  }
</style>
