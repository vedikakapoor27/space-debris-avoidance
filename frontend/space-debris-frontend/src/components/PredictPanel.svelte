<script>
  import { predict } from '../utils/api.js'
  import { formValues, prediction, pushTelemetry } from '../stores/appStore.js'

  let loading = false
  let error   = null

  // Reactive sliders
  let dist_km    = 50
  let rel_vel    = 7
  let approach   = -5

  async function runPrediction() {
    loading = true
    error = null
    try {
      const result = await predict({
        distance_km:   dist_km,
        rel_velocity:  rel_vel,
        approach_rate: approach
      })
      prediction.set(result)
      pushTelemetry(result)
    } catch (e) {
      error = e.message
    } finally {
      loading = false
    }
  }

  $: riskColor = $prediction?.risk_level === 'HIGH'   ? '#ff2244'
               : $prediction?.risk_level === 'MEDIUM' ? '#ff8800'
               : '#00ff88'
</script>

<div class="panel">
  <div class="panel-header">
    <h2 class="panel-title">COLLISION RISK PREDICTOR</h2>
    <p class="panel-sub">Powered by ML model — POST /predict</p>
  </div>

  <div class="split">
    <!-- Input Form -->
    <div class="form-card">
      <div class="form-title">⟁ INPUT PARAMETERS</div>

      <div class="field">
        <div class="field-header">
          <label>DISTANCE</label>
          <span class="field-val">{dist_km} km</span>
        </div>
        <input type="range" min="1" max="500" step="0.5" bind:value={dist_km} />
        <div class="range-hint"><span>1 km</span><span>500 km</span></div>
      </div>

      <div class="field">
        <div class="field-header">
          <label>RELATIVE VELOCITY</label>
          <span class="field-val">{rel_vel} km/s</span>
        </div>
        <input type="range" min="0.1" max="20" step="0.1" bind:value={rel_vel} />
        <div class="range-hint"><span>0.1</span><span>20 km/s</span></div>
      </div>

      <div class="field">
        <div class="field-header">
          <label>APPROACH RATE</label>
          <span class="field-val">{approach} km/s</span>
        </div>
        <input type="range" min="-20" max="20" step="0.5" bind:value={approach} />
        <div class="range-hint"><span>−20</span><span>+20</span></div>
      </div>

      <!-- Quick scenarios -->
      <div class="scenarios">
        <div class="sc-label">QUICK SCENARIOS</div>
        <div class="sc-btns">
          <button class="sc-btn red"
            on:click={() => { dist_km=8; rel_vel=13; approach=-18 }}>
            CRITICAL
          </button>
          <button class="sc-btn orange"
            on:click={() => { dist_km=35; rel_vel=7; approach=-4 }}>
            MEDIUM
          </button>
          <button class="sc-btn green"
            on:click={() => { dist_km=9000; rel_vel=2; approach=3 }}>
            SAFE
          </button>
        </div>
      </div>

      <button class="run-btn" class:loading on:click={runPrediction} disabled={loading}>
        {#if loading}
          <span class="spinner">◌</span> ANALYZING...
        {:else}
          ▶ RUN PREDICTION
        {/if}
      </button>

      {#if error}
        <div class="error-box">⚠ {error} — Is the Flask backend running on :5000?</div>
      {/if}
    </div>

    <!-- Result Card -->
    <div class="result-card">
      {#if $prediction}
        <div class="result-inner">
          <!-- Risk Level -->
          <div class="risk-ring" style="--rc:{riskColor}">
            <div class="risk-inner">
              <div class="risk-pct" style="color:{riskColor}">{$prediction.probability}%</div>
              <div class="risk-label">PROBABILITY</div>
            </div>
          </div>

          <div class="risk-level-badge" style="color:{riskColor}; border-color:{riskColor}; box-shadow: 0 0 20px {riskColor}44">
            {$prediction.risk_level}
          </div>

          <p class="risk-message">{$prediction.message}</p>

          <!-- Avoidance Plan -->
          <div class="avoidance-card">
            <div class="av-header">◎ AVOIDANCE PLAN</div>
            <div class="av-grid">
              <div class="av-row">
                <span>ACTION</span>
                <span style="color:{riskColor}">{$prediction.action}</span>
              </div>
              <div class="av-row">
                <span>MANEUVER TYPE</span>
                <span>{$prediction.maneuver_type}</span>
              </div>
              <div class="av-row">
                <span>ADJUST BY</span>
                <span>{$prediction.maneuver_km}</span>
              </div>
              <div class="av-row">
                <span>FUEL REQUIRED</span>
                <span>{$prediction.fuel_cost_kg}</span>
              </div>
              <div class="av-row">
                <span>TIME WINDOW</span>
                <span>{$prediction.time_window}</span>
              </div>
              <div class="av-row">
                <span>URGENCY</span>
                <span style="color:{riskColor}">{$prediction.urgency}</span>
              </div>
            </div>
          </div>
        </div>
      {:else}
        <div class="empty-state">
          <div class="empty-icon">⟁</div>
          <p>Configure parameters and run prediction</p>
          <p class="empty-sub">Results will appear here</p>
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
    gap: 24px;
  }
  .panel-header {}
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

  .split {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    flex: 1;
  }

  /* Form */
  .form-card {
    background: rgba(0,229,255,0.02);
    border: 1px solid rgba(0,229,255,0.12);
    border-radius: 8px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }
  .form-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: #00e5ff;
    letter-spacing: 3px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(0,229,255,0.1);
  }

  .field { display: flex; flex-direction: column; gap: 6px; }
  .field-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.4);
    letter-spacing: 1.5px;
  }
  .field-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: #00e5ff;
  }
  input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    height: 3px;
    background: rgba(0,229,255,0.15);
    border-radius: 2px;
    outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: #00e5ff;
    box-shadow: 0 0 8px #00e5ff;
    cursor: pointer;
  }
  .range-hint {
    display: flex;
    justify-content: space-between;
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: rgba(255,255,255,0.2);
  }

  /* Scenarios */
  .scenarios { display: flex; flex-direction: column; gap: 8px; }
  .sc-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: rgba(255,255,255,0.25);
    letter-spacing: 2px;
  }
  .sc-btns { display: flex; gap: 8px; }
  .sc-btn {
    flex: 1;
    padding: 7px;
    border-radius: 4px;
    border: 1px solid;
    background: transparent;
    font-family: 'Rajdhani', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .sc-btn.red    { color: #ff2244; border-color: rgba(255,34,68,0.4); }
  .sc-btn.orange { color: #ff8800; border-color: rgba(255,136,0,0.4); }
  .sc-btn.green  { color: #00ff88; border-color: rgba(0,255,136,0.4); }
  .sc-btn:hover  { background: rgba(255,255,255,0.05); }

  .run-btn {
    padding: 12px;
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.4);
    border-radius: 4px;
    color: #00e5ff;
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .run-btn:hover:not(:disabled) {
    background: rgba(0,229,255,0.15);
    box-shadow: 0 0 20px rgba(0,229,255,0.2);
  }
  .run-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .run-btn.loading { animation: flicker 1s infinite; }
  @keyframes flicker { 0%,100%{opacity:1} 50%{opacity:0.5} }
  .spinner { display: inline-block; animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .error-box {
    background: rgba(255,34,68,0.1);
    border: 1px solid rgba(255,34,68,0.3);
    border-radius: 4px;
    padding: 10px 12px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #ff6688;
  }

  /* Result */
  .result-card {
    background: rgba(0,229,255,0.02);
    border: 1px solid rgba(0,229,255,0.12);
    border-radius: 8px;
    padding: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .result-inner {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
  }

  .risk-ring {
    width: 120px; height: 120px;
    border-radius: 50%;
    border: 3px solid var(--rc, #00e5ff);
    box-shadow: 0 0 30px var(--rc, #00e5ff), inset 0 0 20px color-mix(in srgb, var(--rc, #00e5ff) 10%, transparent);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .risk-inner { text-align: center; }
  .risk-pct {
    font-family: 'Orbitron', sans-serif;
    font-size: 22px;
    font-weight: 700;
  }
  .risk-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 8px;
    color: rgba(255,255,255,0.3);
    letter-spacing: 1.5px;
  }

  .risk-level-badge {
    font-family: 'Orbitron', sans-serif;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 4px;
    padding: 6px 20px;
    border: 1px solid;
    border-radius: 3px;
  }
  .risk-message {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: rgba(255,255,255,0.5);
    text-align: center;
    margin: 0;
  }

  .avoidance-card {
    width: 100%;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 6px;
    overflow: hidden;
  }
  .av-header {
    padding: 10px 14px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    color: rgba(0,229,255,0.6);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    background: rgba(0,229,255,0.04);
  }
  .av-grid { padding: 6px 0; }
  .av-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
  }
  .av-row span:first-child {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: rgba(255,255,255,0.3);
    letter-spacing: 1px;
  }
  .av-row span:last-child {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.8);
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    opacity: 0.3;
  }
  .empty-icon {
    font-size: 48px;
    color: #00e5ff;
    animation: float 3s ease-in-out infinite;
  }
  @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
  .empty-state p {
    font-family: 'Rajdhani', sans-serif;
    font-size: 13px;
    color: rgba(255,255,255,0.5);
    margin: 0;
    letter-spacing: 1px;
  }
  .empty-sub { font-size: 10px !important; color: rgba(255,255,255,0.25) !important; }
</style>
