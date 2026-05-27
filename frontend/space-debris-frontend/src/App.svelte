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
  onMount(async () => {
    try {
      await checkHealth()
      backendOnline.set(true)
    } catch {
      backendOnline.set(false)
    }
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
        <span class="tb-time" id="clock">
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
  :global(*) {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  :global(body) {
    background: #000814;
    color: #fff;
    overflow: hidden;
    height: 100vh;
    width: 100vw;
  }
  :global(#app) {
    height: 100vh;
    width: 100vw;
  }
  :global(::-webkit-scrollbar) { width: 4px; }
  :global(::-webkit-scrollbar-track) { background: transparent; }
  :global(::-webkit-scrollbar-thumb) { background: rgba(0,229,255,0.2); border-radius: 2px; }

  .app {
    display: flex;
    height: 100vh;
    width: 100vw;
    position: relative;
    overflow: hidden;
  }

  /* CRT scanlines */
  .scanlines {
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,0,0,0.03) 2px,
      rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
    z-index: 1000;
  }

  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    height: 100vh;
  }

  /* Top bar */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    height: 44px;
    min-height: 44px;
    background: rgba(0,8,26,0.95);
    border-bottom: 1px solid rgba(0,229,255,0.12);
    backdrop-filter: blur(10px);
  }
  .topbar-left { display: flex; align-items: center; gap: 8px; }
  .tb-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: rgba(0,229,255,0.4);
    letter-spacing: 2px;
  }
  .tb-time {
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    color: #00e5ff;
    letter-spacing: 2px;
  }
  .topbar-center {}
  .tb-mission {
    font-family: 'Orbitron', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: rgba(255,255,255,0.5);
    letter-spacing: 2px;
  }
  .topbar-right { display: flex; align-items: center; gap: 14px; }
  .backend-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    letter-spacing: 1.5px;
    color: rgba(255,34,68,0.7);
  }
  .backend-status.online { color: rgba(0,255,136,0.8); }
  .bdot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #ff2244;
    box-shadow: 0 0 6px #ff2244;
  }
  .backend-status.online .bdot { background: #00ff88; box-shadow: 0 0 6px #00ff88; }
  .globe-toggle {
    font-family: 'Orbitron', sans-serif;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: rgba(0,229,255,0.6);
    background: transparent;
    border: 1px solid rgba(0,229,255,0.2);
    padding: 4px 10px;
    border-radius: 3px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .globe-toggle:hover { border-color: rgba(0,229,255,0.5); color: #00e5ff; }

  /* Body */
  .body {
    flex: 1;
    display: grid;
    grid-template-columns: 420px 1fr;
    min-height: 0;
    overflow: hidden;
  }

  /* Globe pane */
  .globe-pane {
    position: relative;
    background: radial-gradient(ellipse at center, #000d24 0%, #000814 100%);
    border-right: 1px solid rgba(0,229,255,0.1);
    overflow: hidden;
  }
  .globe-stats {
    position: absolute;
    top: 14px;
    left: 14px;
    display: flex;
    align-items: center;
    gap: 0;
    background: rgba(0,8,26,0.75);
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: 4px;
    backdrop-filter: blur(8px);
    overflow: hidden;
  }
  .gs-item {
    padding: 8px 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }
  .gs-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #00e5ff;
  }
  .gs-key {
    font-family: 'Share Tech Mono', monospace;
    font-size: 7px;
    color: rgba(255,255,255,0.3);
    letter-spacing: 1px;
  }
  .gs-divider {
    width: 1px;
    height: 30px;
    background: rgba(0,229,255,0.1);
  }

  .globe-legend {
    position: absolute;
    bottom: 14px;
    right: 14px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: rgba(0,8,26,0.7);
    border: 1px solid rgba(0,229,255,0.1);
    border-radius: 4px;
    padding: 10px 12px;
    backdrop-filter: blur(8px);
  }
  .leg-item {
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: rgba(255,255,255,0.5);
    letter-spacing: 1px;
  }
  .leg-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
  }

  /* Right panel */
  .right-panel {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: #000814;
  }
</style>
