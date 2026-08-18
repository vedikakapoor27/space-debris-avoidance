<script>
  import { fly, scale, fade } from 'svelte/transition'
  import { cubicOut, elasticOut } from 'svelte/easing'
  import { predict } from '../utils/api.js'
  import { prediction, pushTelemetry } from '../stores/appStore.js'
  import { authStore } from '../stores/authStore.js'
  import { canPredict } from '../utils/permissions.js'
  import AccessDenied from './AccessDenied.svelte'

  let loading = false, error = null
  let dist_km = 50, rel_vel = 7, approach = -5

  $: role = $authStore.user?.role || 'viewer'
  $: allowed = canPredict(role)

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
  {#if !allowed}
    <AccessDenied
      title="Prediction Locked"
      message="Only operators and administrators can run collision risk predictions."
      role={role}
    />
  {:else}

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
  {/if}
</div>

<style>
  .pp {
    padding: 24px; height: 100%;
    display: flex; flex-direction: column; gap: 20px;
    overflow-y: auto; background: var(--bg);
    transition: background 0.3s;
  }

  .pp-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; color: var(--text-4);
    letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 8px;
  }

  .pp-title {
    font-family: 'Inter', sans-serif;
    font-size: 20px; font-weight: 700; color: var(--text);
    line-height: 1.2; letter-spacing: -0.01em; text-transform: uppercase;
  }

  .pp-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; color: var(--text-4); margin-top: 6px;
    letter-spacing: 0.06em;
  }

  .pp-body { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; flex: 1; }

  .form-col {
    border: 1px solid var(--border); background: var(--card);
    padding: 16px; display: flex; flex-direction: column; gap: 16px;
  }

  .form-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 600;
    color: var(--text-3); letter-spacing: 0.16em;
    text-transform: uppercase; padding-bottom: 10px;
    border-bottom: 1px solid var(--divider);
  }

  .field { display: flex; flex-direction: column; gap: 7px; }

  .field-row { display: flex; justify-content: space-between; align-items: center; }

  label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; color: var(--text-4);
    letter-spacing: 0.14em; text-transform: uppercase;
  }

  .field-val {
    font-family: 'Inter', monospace;
    font-size: 13px; font-weight: 500; color: var(--text);
  }
  .field-unit { font-weight: 300; font-size: 9px; color: var(--text-4); }

  input[type=range] {
    -webkit-appearance: none; width: 100%;
    height: 1px; background: var(--border);
    outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px; height: 12px;
    background: var(--text); cursor: none;
    border: none;
  }

  .range-ends {
    display: flex; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; color: var(--text-4);
  }

  .scenarios { display: flex; flex-direction: column; gap: 7px; }

  .sc-hd {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; color: var(--text-4);
    letter-spacing: 0.14em; text-transform: uppercase;
  }

  .sc-btns { display: flex; gap: 6px; }

  .sc-btn {
    flex: 1; padding: 7px;
    background: transparent; border: 1px solid var(--border);
    font-family: 'Inter', monospace;
    font-size: 8px; letter-spacing: 0.1em;
    cursor: none; transition: all 0.15s;
    text-transform: uppercase; color: var(--text-3);
  }
  .sc-btn:hover { border-color: var(--border2); color: var(--text); background: var(--surface); }

  .run-btn {
    padding: 11px; background: var(--text);
    border: none; color: var(--bg);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 600;
    letter-spacing: 0.16em; cursor: none;
    transition: opacity 0.2s; text-transform: uppercase;
  }
  .run-btn:hover:not(:disabled) { opacity: 0.85; }
  .run-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .run-btn.loading { animation: flicker 1s infinite; }
  @keyframes flicker { 0%,100%{opacity:1} 50%{opacity:0.5} }
  .spin { display: inline-block; animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .err {
    background: var(--danger-bg); border: 1px solid var(--border);
    padding: 8px 10px; font-family: 'Inter', monospace;
    font-size: 8px; color: var(--text-2); letter-spacing: 0.06em;
  }

  /* RESULT */
  .result-col {
    border: 1px solid var(--border); background: var(--card);
    padding: 16px; display: flex; align-items: center; justify-content: center;
  }

  .result { width: 100%; display: flex; flex-direction: column; align-items: center; gap: 14px; }

  .risk-ring {
    width: 130px; height: 130px;
    border: 1px solid var(--border2);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: var(--surface);
  }

  .risk-num {
    font-family: 'Inter', sans-serif;
    font-size: 32px; font-weight: 700;
    color: var(--text); line-height: 1;
  }

  .risk-sublabel {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 7px; color: var(--text-4);
    letter-spacing: 0.14em; text-align: center;
    margin-top: 4px; text-transform: uppercase;
  }

  .risk-badge {
    font-family: 'Inter', monospace;
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.2em; padding: 5px 20px;
    border: 1px solid var(--border2); color: var(--text);
    text-transform: uppercase;
  }

  .risk-msg {
    font-family: 'Inter', monospace;
    font-size: 9px; color: var(--text-4);
    text-align: center; line-height: 1.6; letter-spacing: 0.04em;
  }

  .avoid-block { width: 100%; border: 1px solid var(--border); }

  .avoid-hd {
    padding: 8px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; font-weight: 600; color: var(--text-3);
    letter-spacing: 0.16em; text-transform: uppercase;
    border-bottom: 1px solid var(--divider); background: var(--card2);
  }

  .avoid-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 12px; border-bottom: 1px solid var(--divider);
  }
  .avoid-row:last-child { border-bottom: none; }

  .ak {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; color: var(--text-4); letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .av {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 500; color: var(--text);
  }

  .empty {
    display: flex; flex-direction: column;
    align-items: center; gap: 10px; opacity: 0.2;
  }
  .empty-ico { font-size: 40px; color: var(--text); }
  .empty p   { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--text-4); margin: 0; letter-spacing: 0.08em; }
  .empty-s   { font-size: 9px !important; }
</style>