<script>
  import { onMount } from 'svelte'
  import Sidebar from './components/Sidebar.svelte'
  import Globe3D from './components/Globe3D.svelte'
  import DashboardPanel from './components/DashboardPanel.svelte'
  import PredictPanel from './components/PredictPanel.svelte'
  import ConjunctionsPanel from './components/ConjunctionsPanel.svelte'
  import TelemetryPanel from './components/TelemetryPanel.svelte'
  import { activePanel, backendOnline, globeRotating } from './stores/appStore.js'
  import { checkHealth } from './utils/api.js'

  // Check backend health on load
  let clockStr = '--:--:--'
onMount(() => {
  const tick = () => {
    const n = new Date()
    clockStr = n.toUTCString().split(' ')[4]
  }
  tick()
  const id = setInterval(tick, 1000)
  return () => clearInterval(id)
})
</script>

<div class="app">
  <!-- Scanline overlay for CRT feel -->
  <div class="scanlines" aria-hidden="true"></div>

  <!-- Sidebar Nav -->
  <Sidebar />

  <!-- Main Content -->
  <div class="main">

    <!-- Top Bar -->
    <header class="topbar">
      <div class="topbar-left">
        <span class="tb-label">UTC</span>
        <span class="tb-time">{clockStr}
          {new Date().toISOString().slice(11,19)}
        </span>
      </div>

      <div class="topbar-center">
        <span class="tb-mission">ORION MISSION CONTROL — DEBRIS AVOIDANCE SYSTEM</span>
      </div>

      <div class="topbar-right">
        <span class="backend-status" class:online={$backendOnline}>
          <span class="bdot"></span>
          BACKEND {$backendOnline ? 'ONLINE' : 'OFFLINE'}
        </span>
        <button
          class="globe-toggle"
          on:click={() => globeRotating.update(v => !v)}
        >
          {$globeRotating ? '⏸ PAUSE' : '▶ RESUME'}
        </button>
      </div>
    </header>

    <!-- Body: Globe + Active Panel side by side -->
    <div class="body">

      <!-- 3D Globe (always visible) -->
      <div class="globe-pane">
        <Globe3D />

        <!-- Overlay stats on globe -->
        <div class="globe-stats">
          <div class="gs-item">
            <span class="gs-val">847</span>
            <span class="gs-key">OBJECTS</span>
          </div>
          <div class="gs-divider"></div>
          <div class="gs-item">
            <span class="gs-val" style="color:#ff2244">2</span>
            <span class="gs-key">HIGH RISK</span>
          </div>
          <div class="gs-divider"></div>
          <div class="gs-item">
            <span class="gs-val" style="color:#ff8800">3</span>
            <span class="gs-key">MEDIUM</span>
          </div>
        </div>

        <!-- Legend -->
        <div class="globe-legend">
          <div class="leg-item"><span class="leg-dot" style="background:#ff2244;box-shadow:0 0 6px #ff2244"></span>HIGH RISK</div>
          <div class="leg-item"><span class="leg-dot" style="background:#ff8800;box-shadow:0 0 6px #ff8800"></span>MEDIUM</div>
          <div class="leg-item"><span class="leg-dot" style="background:#00e5ff;box-shadow:0 0 6px #00e5ff"></span>LOW RISK</div>
        </div>
      </div>

      <!-- Right Panel — swaps based on nav -->
      <div class="right-panel">
        {#if $activePanel === 'dashboard'}
          <DashboardPanel />
        {:else if $activePanel === 'predict'}
          <PredictPanel />
        {:else if $activePanel === 'conjunctions'}
          <ConjunctionsPanel />
        {:else if $activePanel === 'telemetry'}
          <TelemetryPanel />
        {/if}
      </div>
    </div>
  </div>
</div>

<style>
  @import url('https://fonts.googleapis.com/css2?family=Limelight&family=JetBrains+Mono:wght@300;400;500&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

  :global(*) { box-sizing: border-box; margin: 0; padding: 0; }

  :global(:root) {
    /* NEW color palette — deep purple + gold + violet + ice white */
    --void:       #05020f;
    --deep:       #0a0520;
    --surface:    #0f0a24;
    --glass:      rgba(120,60,255,0.04);
    --border:     rgba(140,80,255,0.14);
    --border-dim: rgba(140,80,255,0.07);

    --purple:     #7c3fff;
    --violet:     #b060ff;
    --gold:       #e8b84b;
    --gold-dim:   rgba(232,184,75,0.6);
    --white:      #f0ecff;
    --ice:        #d4c8ff;
    --dim:        rgba(180,160,255,0.4);

    --danger:     #ff3860;
    --warning:    #ff9020;
    --safe:       #00e8a0;

    --glow-purple: 0 0 20px rgba(124,63,255,0.35);
    --glow-gold:   0 0 20px rgba(232,184,75,0.35);
    --glow-danger: 0 0 20px rgba(255,56,96,0.4);
  }

  :global(body) {
    background: var(--void);
    color: var(--white);
    overflow: hidden;
    height: 100vh;
    font-family: 'Space Grotesk', sans-serif;
  }

  :global(#app) { height: 100vh; width: 100vw; }

  :global(::-webkit-scrollbar) { width: 3px; }
  :global(::-webkit-scrollbar-thumb) { background: rgba(124,63,255,0.25); }

  .app {
    display: flex;
    height: 100vh;
    width: 100vw;
    position: relative;
    overflow: hidden;
    background:
      radial-gradient(ellipse 50% 60% at 15% 50%, rgba(80,20,180,0.08) 0%, transparent 65%),
      radial-gradient(ellipse 40% 40% at 85% 20%, rgba(180,80,255,0.05) 0%, transparent 60%),
      var(--void);
  }

  .scanlines {
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 2px,
      rgba(0,0,0,0.025) 2px, rgba(0,0,0,0.025) 4px
    );
    pointer-events: none; z-index: 1000;
  }

  .main { flex: 1; display: flex; flex-direction: column; min-width: 0; height: 100vh; }

  .topbar {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 0 24px; height: 46px; min-height: 46px;
    background: rgba(5,2,15,0.95);
    border-bottom: 1px solid var(--border-dim);
    backdrop-filter: blur(20px);
  }

  .topbar-left { display: flex; align-items: center; gap: 10px; }

  .tb-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--dim);
    letter-spacing: 0.2em;
  }

  .tb-time {
    font-family: 'Limelight', cursive;
    font-size: 15px; color: var(--gold);
    letter-spacing: 0.08em;
  }

  .tb-mission {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 500;
    color: var(--dim); letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  .topbar-right { display: flex; align-items: center; gap: 14px; }

  .backend-status {
    display: flex; align-items: center; gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.12em;
    color: rgba(255,56,96,0.7);
  }

  .backend-status.online { color: rgba(0,232,160,0.8); }

  .bdot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--danger);
    box-shadow: 0 0 8px var(--danger);
    animation: bdot 1.5s ease-in-out infinite;
  }

  .backend-status.online .bdot { background: var(--safe); box-shadow: 0 0 8px var(--safe); }

  @keyframes bdot { 0%,100%{opacity:1} 50%{opacity:0.2} }

  .globe-toggle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.12em;
    color: var(--violet); background: transparent;
    border: 1px solid rgba(176,96,255,0.25);
    padding: 5px 12px; cursor: pointer;
    transition: all 0.2s;
  }

  .globe-toggle:hover {
    border-color: var(--violet);
    background: rgba(124,63,255,0.08);
  }

  .body {
    flex: 1; display: grid;
    grid-template-columns: 440px 1fr;
    min-height: 0; overflow: hidden;
  }

  .globe-pane {
    position: relative;
    background: radial-gradient(ellipse at center, #0d0525 0%, #05020f 100%);
    border-right: 1px solid var(--border-dim);
    overflow: hidden;
  }

  .globe-stats {
    position: absolute; top: 14px; left: 14px;
    display: flex; align-items: center;
    background: rgba(5,2,15,0.8);
    border: 1px solid var(--border);
    backdrop-filter: blur(12px);
    overflow: hidden;
  }

  .gs-item {
    padding: 8px 16px;
    display: flex; flex-direction: column;
    align-items: center; gap: 3px;
  }

  .gs-val {
    font-family: 'Limelight', cursive;
    font-size: 16px; color: var(--gold);
  }

  .gs-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 7px; color: var(--dim);
    letter-spacing: 0.12em; text-transform: uppercase;
  }

  .gs-divider { width: 1px; height: 32px; background: var(--border-dim); }

  .globe-legend {
    position: absolute; bottom: 14px; right: 14px;
    display: flex; flex-direction: column; gap: 7px;
    background: rgba(5,2,15,0.75);
    border: 1px solid var(--border-dim);
    padding: 10px 14px;
    backdrop-filter: blur(10px);
  }

  .leg-item {
    display: flex; align-items: center; gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--dim);
    letter-spacing: 0.1em;
  }

  .leg-dot { width: 7px; height: 7px; border-radius: 50%; }

  .right-panel {
    display: flex; flex-direction: column;
    overflow: hidden; background: var(--void);
  }
</style>
