<script>
  import { fly, scale, fade } from 'svelte/transition'
  import { cubicOut, elasticOut } from 'svelte/easing'
  import { predict } from '../utils/api.js'
  import { prediction, pushTelemetry } from '../stores/appStore.js'

  let loading = false, error = null
  let dist_km = 50, rel_vel = 7, approach = -5

  async function runPrediction() {
    loading = true; error = null; prediction.set(null)
    try {
      const result = await predict({ distance_km: dist_km, rel_velocity: rel_vel, approach_rate: approach })
      prediction.set(result)
      pushTelemetry(result)
    } catch (e) { error = e.message }
    finally { loading = false }
  }

  $: riskColor = $prediction?.risk_level === 'HIGH'   ? 'var(--danger)'
               : $prediction?.risk_level === 'MEDIUM' ? 'var(--warning)'
               : 'var(--safe)'
</script>

<div class="pp">

  <div class="pp-header">
    <div class="pp-eyebrow">ML Analysis</div>
    <h2 class="pp-title">Collision Risk<br>Predictor</h2>
    <p class="pp-meta">POST /predict · Random Forest · sklearn</p>
  </div>

  <div class="pp-body">

    <div class="form-col">
      <div class="form-header">Input Parameters</div>

      <div class="field">
        <div class="field-row">
          <label>Distance</label>
          <span class="field-val">{dist_km}<span class="field-unit"> km</span></span>
        </div>
        <input type="range" min="1" max="500" step="0.5" bind:value={dist_km} />
        <div class="range-ends"><span>1 km</span><span>500 km</span></div>
      </div>

      <div class="field">
        <div class="field-row">
          <label>Relative Velocity</label>
          <span class="field-val">{rel_vel}<span class="field-unit"> km/s</span></span>
        </div>
        <input type="range" min="0.1" max="20" step="0.1" bind:value={rel_vel} />
        <div class="range-ends"><span>0.1</span><span>20 km/s</span></div>
      </div>

      <div class="field">
        <div class="field-row">
          <label>Approach Rate</label>
          <span class="field-val" style="color:{approach < 0 ? 'var(--danger)' : 'var(--safe)'}">{approach}</span>
        </div>
        <input type="range" min="-20" max="20" step="0.5" bind:value={approach} />
        <div class="range-ends"><span>−20</span><span>+20</span></div>
      </div>

      <div class="scenarios">
        <div class="sc-hd">Quick Scenarios</div>
        <div class="sc-btns">
          <button class="sc-btn d" on:click={() => { dist_km=8;   rel_vel=13; approach=-18 }}>⚠ Critical</button>
          <button class="sc-btn w" on:click={() => { dist_km=35;  rel_vel=7;  approach=-4  }}>◉ Medium</button>
          <button class="sc-btn s" on:click={() => { dist_km=900; rel_vel=2;  approach=3   }}>✓ Safe</button>
        </div>
      </div>

      <button class="run-btn" class:loading on:click={runPrediction} disabled={loading}>
        {#if loading}
          <span class="spin">◌</span> Analyzing...
        {:else}
          ▶ Run Prediction
        {/if}
      </button>

      {#if error}
        <div class="err" in:fly={{ y: 6, duration: 200 }}>
          ⚠ {error} — Is Flask running on :5000?
        </div>
      {/if}
    </div>

    <div class="result-col">
      {#if $prediction}
        <div class="result" in:fly={{ x: 20, duration: 350, easing: cubicOut }}>

          <div class="risk-ring" style="--rc:{riskColor}">
            <div class="risk-num" in:scale={{ duration: 400, delay: 100, easing: elasticOut }}>
              {$prediction.probability}%
            </div>
            <div class="risk-sublabel">Collision Probability</div>
          </div>

          <div class="risk-badge" style="color:{riskColor}; border-color:{riskColor}">
            {$prediction.risk_level}
          </div>

          <p class="risk-msg">{$prediction.message}</p>

          <div class="avoid-block">
            <div class="avoid-hd">› Avoidance Plan</div>
            {#each [
              ['Action',        $prediction.action,        riskColor],
              ['Maneuver',      $prediction.maneuver_type, null],
              ['Adjust by',     $prediction.maneuver_km,   null],
              ['Fuel required', $prediction.fuel_cost_kg,  null],
              ['Time window',   $prediction.time_window,   null],
              ['Urgency',       $prediction.urgency,       riskColor],
            ] as [k,v,c], i}
              <div class="avoid-row" in:fly={{ x: 10, duration: 200, delay: 100 + i*40, easing: cubicOut }}>
                <span class="ak">{k}</span>
                <span class="av" style={c ? `color:${c}` : ''}>{v}</span>
              </div>
            {/each}
          </div>

        </div>
      {:else}
        <div class="empty">
          <div class="empty-ico" in:scale={{ duration: 600, easing: elasticOut }}>⟁</div>
          <p>Configure parameters</p>
          <p class="empty-s">and run prediction</p>
        </div>
      {/if}
    </div>

  </div>
</div>

<style>
  .pp { padding: 28px; height: 100%; display: flex; flex-direction: column; gap: 24px; overflow-y: auto; }

  .pp-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim);
    letter-spacing: 0.22em; text-transform: uppercase; margin-bottom: 8px;
  }

  .pp-title {
    font-family: 'Syne', sans-serif;
    font-size: 24px; font-weight: 800; color: var(--text);
    line-height: 1.15; letter-spacing: -0.01em;
  }

  .pp-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--text-dim); margin-top: 8px;
  }

  .pp-body { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; flex: 1; }

  .form-col {
    border: 1px solid var(--border-dim); background: var(--glass);
    padding: 20px; display: flex; flex-direction: column; gap: 18px;
  }

  .form-header {
    font-family: 'Syne', sans-serif; font-size: 11px; font-weight: 700;
    color: var(--accent-hi); letter-spacing: 0.1em; text-transform: uppercase;
    padding-bottom: 12px; border-bottom: 1px solid var(--border-dim);
  }

  .field { display: flex; flex-direction: column; gap: 7px; }

  .field-row { display: flex; justify-content: space-between; align-items: center; }

  label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim);
    letter-spacing: 0.15em; text-transform: uppercase;
  }

  .field-val {
    font-family: 'Syne', sans-serif; font-size: 14px; font-weight: 700; color: var(--gold);
  }
  .field-unit { font-weight: 400; font-size: 10px; color: var(--text-dim); }

  input[type=range] {
    -webkit-appearance: none; width: 100%; height: 2px;
    background: var(--border); outline: none; border-radius: 1px;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
    background: var(--accent); box-shadow: var(--accent-glow); cursor: pointer;
  }

  .range-ends {
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace; font-size: 8px; color: var(--text-dim);
  }

  .scenarios { display: flex; flex-direction: column; gap: 8px; }
  .sc-hd {
    font-family: 'JetBrains Mono', monospace; font-size: 8px; color: var(--text-dim);
    letter-spacing: 0.15em; text-transform: uppercase;
  }
  .sc-btns { display: flex; gap: 8px; }
  .sc-btn {
    flex: 1; padding: 8px;
    background: transparent; border: 1px solid;
    font-family: 'Syne', sans-serif; font-size: 10px; font-weight: 600;
    cursor: pointer; transition: all 0.2s;
  }
  .sc-btn.d { color: var(--danger);  border-color: rgba(255,56,96,0.28); }
  .sc-btn.w { color: var(--warning); border-color: rgba(255,144,32,0.28); }
  .sc-btn.s { color: var(--safe);    border-color: rgba(0,232,160,0.25); }
  .sc-btn:hover { background: var(--glass); transform: translateY(-1px); }

  .run-btn {
    padding: 13px; background: var(--glass);
    border: 1px solid var(--accent); color: var(--accent-hi);
    font-family: 'Syne', sans-serif; font-size: 12px; font-weight: 700;
    letter-spacing: 0.1em; cursor: pointer; transition: all 0.25s;
  }
  .run-btn:hover:not(:disabled) { background: rgba(124,58,237,0.12); box-shadow: var(--accent-glow); }
  .run-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .run-btn.loading { animation: flicker 1s infinite; }
  @keyframes flicker { 0%,100%{opacity:1} 50%{opacity:0.5} }
  .spin { display: inline-block; animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .err {
    background: rgba(255,56,96,0.06); border: 1px solid rgba(255,56,96,0.22);
    padding: 10px 12px; font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--danger);
  }

  .result-col {
    border: 1px solid var(--border-dim); background: var(--glass);
    padding: 20px; display: flex; align-items: center; justify-content: center;
  }

  .result { width: 100%; display: flex; flex-direction: column; align-items: center; gap: 16px; }

  .risk-ring {
    width: 150px; height: 150px; border-radius: 50%;
    border: 2px solid var(--rc);
    box-shadow: 0 0 30px rgba(var(--rc), 0.2), inset 0 0 30px rgba(0,0,0,0.3);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: rgba(0,0,0,0.3);
    position: relative;
  }

  .risk-ring::before {
    content: '';
    position: absolute; inset: -8px; border-radius: 50%;
    border: 1px solid var(--rc); opacity: 0.2;
    animation: ringpulse 2s ease-in-out infinite;
  }
  @keyframes ringpulse { 0%,100%{transform:scale(1);opacity:0.2} 50%{transform:scale(1.04);opacity:0.1} }

  .risk-num {
    font-family: 'Syne', sans-serif; font-size: 34px; font-weight: 800;
    color: var(--rc); line-height: 1;
  }

  .risk-sublabel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 7px; color: var(--text-dim);
    letter-spacing: 0.14em; text-align: center; margin-top: 4px;
    text-transform: uppercase;
  }

  .risk-badge {
    font-family: 'Syne', sans-serif; font-size: 14px; font-weight: 800;
    letter-spacing: 0.18em; padding: 6px 24px;
    border: 1px solid; text-transform: uppercase;
  }

  .risk-msg {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--text-dim); text-align: center; line-height: 1.6;
  }

  .avoid-block { width: 100%; border: 1px solid var(--border-dim); overflow: hidden; }

  .avoid-hd {
    padding: 9px 14px; font-family: 'Syne', sans-serif;
    font-size: 10px; font-weight: 700; color: var(--accent-hi);
    letter-spacing: 0.1em; text-transform: uppercase;
    border-bottom: 1px solid var(--border-dim); background: var(--glass);
  }

  .avoid-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 14px; border-bottom: 1px solid var(--border-dim);
  }
  .avoid-row:last-child { border-bottom: none; }

  .ak {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim); letter-spacing: 0.12em;
  }
  .av {
    font-family: 'Syne', sans-serif; font-size: 11px; font-weight: 600; color: var(--text);
  }

  .empty {
    display: flex; flex-direction: column;
    align-items: center; gap: 10px; opacity: 0.2;
  }
  .empty-ico {
    font-size: 52px; color: var(--accent);
    animation: float 3s ease-in-out infinite;
  }
  @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
  .empty p { font-family: 'Syne', sans-serif; font-size: 13px; color: var(--text-dim); margin: 0; }
  .empty-s { font-size: 11px !important; }
</style>
