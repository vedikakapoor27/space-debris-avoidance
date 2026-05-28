<script>
  import { predict } from '../utils/api.js'
  import { prediction, pushTelemetry } from '../stores/appStore.js'

  let loading = false, error = null
  let dist_km = 50, rel_vel = 7, approach = -5

  async function runPrediction() {
    loading = true; error = null
    try {
      const result = await predict({ distance_km: dist_km, rel_velocity: rel_vel, approach_rate: approach })
      prediction.set(result)
      pushTelemetry(result)
    } catch (e) {
      error = e.message
    } finally {
      loading = false
    }
  }

  $: riskColor = $prediction?.risk_level === 'HIGH'   ? 'var(--danger)'
               : $prediction?.risk_level === 'MEDIUM' ? 'var(--warning)'
               : 'var(--safe)'
</script>

<div class="panel">

  <div class="panel-header">
    <div class="header-eyebrow">ML ANALYSIS</div>
    <h2 class="header-title">Collision Risk<br>Predictor</h2>
    <p class="header-sub">POST /predict · Random Forest · sklearn</p>
  </div>

  <div class="split">

    <div class="form-card">
      <div class="form-section-label">Input Parameters</div>

      <div class="field">
        <div class="field-top">
          <label>Distance</label>
          <span class="field-val">{dist_km} <span class="field-unit">km</span></span>
        </div>
        <input type="range" min="1" max="500" step="0.5" bind:value={dist_km} />
        <div class="range-row"><span>1 km</span><span>500 km</span></div>
      </div>

      <div class="field">
        <div class="field-top">
          <label>Relative Velocity</label>
          <span class="field-val">{rel_vel} <span class="field-unit">km/s</span></span>
        </div>
        <input type="range" min="0.1" max="20" step="0.1" bind:value={rel_vel} />
        <div class="range-row"><span>0.1</span><span>20 km/s</span></div>
      </div>

      <div class="field">
        <div class="field-top">
          <label>Approach Rate</label>
          <span class="field-val" style="color:{approach < 0 ? 'var(--danger)' : 'var(--safe)'}">{approach}</span>
        </div>
        <input type="range" min="-20" max="20" step="0.5" bind:value={approach} />
        <div class="range-row"><span>−20</span><span>+20</span></div>
      </div>

      <div class="scenarios">
        <div class="sc-label">Quick Scenarios</div>
        <div class="sc-row">
          <button class="sc-btn danger" on:click={() => { dist_km=8;    rel_vel=13; approach=-18 }}>CRITICAL</button>
          <button class="sc-btn warn"   on:click={() => { dist_km=35;   rel_vel=7;  approach=-4  }}>MEDIUM</button>
          <button class="sc-btn safe"   on:click={() => { dist_km=9000; rel_vel=2;  approach=3   }}>SAFE</button>
        </div>
      </div>

      <button class="run-btn" class:loading on:click={runPrediction} disabled={loading}>
        {#if loading}
          <span class="spin">◌</span> ANALYZING...
        {:else}
          ▶ RUN PREDICTION
        {/if}
      </button>

      {#if error}
        <div class="err-box">⚠ {error} — Is Flask running on :5000?</div>
      {/if}
    </div>

    <div class="result-card">
      {#if $prediction}
        <div class="result-inner">

          <div class="risk-display" style="--rc:{riskColor}">
            <div class="risk-pct">{$prediction.probability}%</div>
            <div class="risk-pct-label">COLLISION PROBABILITY</div>
          </div>

          <div class="risk-level" style="color:{riskColor}; border-color:{riskColor}">
            {$prediction.risk_level}
          </div>

          <p class="risk-msg">{$prediction.message}</p>

          <div class="avoid-card">
            <div class="avoid-head">Avoidance Plan</div>
            <div class="avoid-rows">
              {#each [
                ['Action',       $prediction.action,        riskColor],
                ['Maneuver',     $prediction.maneuver_type, null],
                ['Adjust by',    $prediction.maneuver_km,   null],
                ['Fuel required',$prediction.fuel_cost_kg,  null],
                ['Time window',  $prediction.time_window,   null],
                ['Urgency',      $prediction.urgency,       riskColor]
              ] as [k,v,c]}
                <div class="avoid-row">
                  <span class="avoid-key">{k}</span>
                  <span class="avoid-val" style={c ? `color:${c}` : ''}>{v}</span>
                </div>
              {/each}
            </div>
          </div>

        </div>
      {:else}
        <div class="empty">
          <div class="empty-icon">⟁</div>
          <p>Configure parameters</p>
          <p class="empty-sub">and run prediction</p>
        </div>
      {/if}
    </div>

  </div>
</div>

<style>
  .panel {
    flex: 1; padding: 28px;
    overflow-y: auto;
    display: flex; flex-direction: column; gap: 24px;
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
    letter-spacing: 0.04em; line-height: 1.15;
    font-weight: 400;
  }

  .header-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--text-dim);
    margin-top: 6px;
  }

  .split {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 20px; flex: 1;
  }

  .form-card {
    border: 1px solid var(--border-dim);
    background: var(--glass);
    padding: 20px;
    display: flex; flex-direction: column; gap: 18px;
  }

  .form-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--accent-hi);
    letter-spacing: 0.2em; text-transform: uppercase;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-dim);
  }

  .field { display: flex; flex-direction: column; gap: 7px; }

  .field-top {
    display: flex; justify-content: space-between; align-items: center;
  }

  label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim);
    letter-spacing: 0.15em; text-transform: uppercase;
  }

  .field-val {
    font-family: 'Limelight', cursive;
    font-size: 14px; color: var(--gold);
  }

  .field-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim);
  }

  input[type=range] {
    -webkit-appearance: none; width: 100%;
    height: 2px; background: var(--border);
    outline: none; border-radius: 1px;
  }

  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--accent);
    box-shadow: var(--glow-accent);
    cursor: pointer;
  }

  .range-row {
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim);
  }

  .scenarios { display: flex; flex-direction: column; gap: 8px; }

  .sc-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim);
    letter-spacing: 0.15em; text-transform: uppercase;
  }

  .sc-row { display: flex; gap: 8px; }

  .sc-btn {
    flex: 1; padding: 7px;
    background: transparent; border: 1px solid;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.12em;
    cursor: pointer; transition: all 0.2s;
    text-transform: uppercase;
  }
  .sc-btn.danger { color: var(--danger); border-color: rgba(255,56,96,0.3); }
  .sc-btn.warn   { color: var(--warning); border-color: rgba(255,144,32,0.3); }
  .sc-btn.safe   { color: var(--safe);    border-color: rgba(0,232,160,0.3); }
  .sc-btn:hover  { background: var(--glass); }

  .run-btn {
    padding: 13px;
    background: var(--glass);
    border: 1px solid var(--accent);
    color: var(--accent-hi);
    font-family: 'Limelight', cursive;
    font-size: 12px; letter-spacing: 0.15em;
    cursor: pointer; transition: all 0.25s;
  }
  .run-btn:hover:not(:disabled) {
    background: rgba(139,92,246,0.12);
    box-shadow: var(--glow-accent);
  }
  .run-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .run-btn.loading  { animation: flicker 1s infinite; }
  @keyframes flicker { 0%,100%{opacity:1} 50%{opacity:0.5} }
  .spin { display: inline-block; animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .err-box {
    background: rgba(255,56,96,0.07);
    border: 1px solid rgba(255,56,96,0.25);
    padding: 10px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--danger);
  }

  .result-card {
    border: 1px solid var(--border-dim);
    background: var(--glass);
    padding: 20px;
    display: flex; align-items: center; justify-content: center;
  }

  .result-inner {
    width: 100%;
    display: flex; flex-direction: column;
    align-items: center; gap: 16px;
  }

  .risk-display {
    text-align: center; padding: 24px 0 16px;
    border-bottom: 1px solid var(--border-dim);
    width: 100%;
  }

  .risk-pct {
    font-family: 'Limelight', cursive;
    font-size: 52px; color: var(--rc);
    line-height: 1;
  }

  .risk-pct-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim);
    letter-spacing: 0.2em; margin-top: 6px;
  }

  .risk-level {
    font-family: 'Limelight', cursive;
    font-size: 16px; letter-spacing: 0.18em;
    padding: 6px 24px; border: 1px solid;
    text-transform: uppercase;
  }

  .risk-msg {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 12px; color: var(--text-dim);
    text-align: center; line-height: 1.5;
  }

  .avoid-card {
    width: 100%;
    border: 1px solid var(--border-dim);
    overflow: hidden;
  }

  .avoid-head {
    padding: 9px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--accent-hi);
    letter-spacing: 0.18em; text-transform: uppercase;
    border-bottom: 1px solid var(--border-dim);
    background: var(--glass);
  }

  .avoid-rows { padding: 4px 0; }

  .avoid-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 14px;
    border-bottom: 1px solid var(--border-dim);
  }
  .avoid-row:last-child { border-bottom: none; }

  .avoid-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim); letter-spacing: 0.12em;
  }

  .avoid-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--text);
  }

  .empty {
    display: flex; flex-direction: column;
    align-items: center; gap: 10px; opacity: 0.25;
  }

  .empty-icon {
    font-size: 48px; color: var(--accent);
    animation: float 3s ease-in-out infinite;
  }
  @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }

  .empty p, .empty-sub {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px; color: var(--text-dim); margin: 0;
  }
  .empty-sub { font-size: 11px !important; }
</style>
