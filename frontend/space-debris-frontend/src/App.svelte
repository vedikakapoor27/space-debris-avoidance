<script>
  import { onMount } from 'svelte'
  import Sidebar from './components/Sidebar.svelte'
  import Globe3D from './components/Globe3D.svelte'
  import DashboardPanel from './components/DashboardPanel.svelte'
  import PredictPanel from './components/PredictPanel.svelte'
  import ConjunctionsPanel from './components/ConjunctionsPanel.svelte'
  import TelemetryPanel from './components/TelemetryPanel.svelte'
  import { activePanel, backendOnline, globeRotating, theme } from './stores/appStore.js'
  import { checkHealth } from './utils/api.js'

  let clockStr = '--:--:--'

  onMount(async () => {
    const tick = () => {
      const n = new Date()
      clockStr = n.toUTCString().split(' ')[4]
    }
    tick()
    const id = setInterval(tick, 1000)

    try {
      await checkHealth()
      backendOnline.set(true)
    } catch {
      backendOnline.set(false)
    }

    return () => clearInterval(id)
  })

  function toggleTheme() {
    theme.update(t => t === 'dark' ? 'void' : 'dark')
  }
</script>

<div class="app" data-theme={$theme}>
  <div class="noise" aria-hidden="true"></div>

  <Sidebar />

  <div class="main">
    <header class="topbar">
      <div class="topbar-left">
        <span class="tb-label">UTC</span>
        <span class="tb-time">{clockStr}</span>
        <span class="tb-sep">·</span>
        <span class="tb-mission">ASTRAEUS — SPACE DEBRIS SENTINEL</span>
      </div>

      <div class="topbar-right">
        <button class="theme-toggle" on:click={toggleTheme} title="Toggle theme">
          {$theme === 'dark' ? '◑ VOID' : '◑ DARK'}
        </button>

        <span class="backend-status" class:online={$backendOnline}>
          <span class="bdot"></span>
          {$backendOnline ? 'BACKEND ONLINE' : 'BACKEND OFFLINE'}
        </span>

        <button
          class="globe-toggle"
          on:click={() => globeRotating.update(v => !v)}
        >
          {$globeRotating ? '⏸ PAUSE' : '▶ RESUME'}
        </button>
      </div>
    </header>

    <div class="body">
      <div class="globe-pane">
        <Globe3D />

        <div class="globe-stats">
          <div class="gs-item">
            <span class="gs-val">847</span>
            <span class="gs-key">OBJECTS</span>
          </div>
          <div class="gs-divider"></div>
          <div class="gs-item">
            <span class="gs-val danger">2</span>
            <span class="gs-key">HIGH RISK</span>
          </div>
          <div class="gs-divider"></div>
          <div class="gs-item">
            <span class="gs-val warn">3</span>
            <span class="gs-key">MEDIUM</span>
          </div>
        </div>

        <div class="globe-legend">
          <div class="leg-item"><span class="leg-dot" style="background:#ff3860"></span>CRITICAL</div>
          <div class="leg-item"><span class="leg-dot" style="background:#ff9020"></span>WARNING</div>
          <div class="leg-item"><span class="leg-dot" style="background:#00e8a0"></span>NOMINAL</div>
        </div>
      </div>

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
  @import url('https://fonts.googleapis.com/css2?family=Limelight&family=JetBrains+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

  /* ── DARK THEME (default) ── */
  :global([data-theme="dark"]) {
    --bg:          #05020f;
    --bg2:         #0a0520;
    --surface:     #0f0a24;
    --glass:       rgba(120,60,255,0.04);
    --border:      rgba(140,80,255,0.16);
    --border-dim:  rgba(140,80,255,0.07);

    --accent:      #8b5cf6;
    --accent-hi:   #a78bfa;
    --gold:        #e8b84b;
    --gold-dim:    rgba(232,184,75,0.55);
    --violet:      #c084fc;
    --text:        #ede9fe;
    --text-dim:    rgba(196,181,253,0.45);

    --danger:      #ff3860;
    --warning:     #ff9020;
    --safe:        #00e8a0;

    --sidebar-bg:  #06031a;
    --panel-bg:    #07041a;
    --topbar-bg:   rgba(5,2,15,0.97);

    --glow-accent: 0 0 24px rgba(139,92,246,0.3);
    --glow-gold:   0 0 20px rgba(232,184,75,0.3);
    --glow-danger: 0 0 20px rgba(255,56,96,0.35);
  }

  /* ── VOID WHITE THEME ── */
  :global([data-theme="void"]) {
    --bg:          #f8f8f0;
    --bg2:         #f0eee8;
    --surface:     #ffffff;
    --glass:       rgba(0,0,0,0.03);
    --border:      rgba(0,0,0,0.1);
    --border-dim:  rgba(0,0,0,0.06);

    --accent:      #7c3fff;
    --accent-hi:   #5b21b6;
    --gold:        #b8860b;
    --gold-dim:    rgba(184,134,11,0.6);
    --violet:      #6d28d9;
    --text:        #0a0a0a;
    --text-dim:    rgba(10,10,10,0.45);

    --danger:      #dc2626;
    --warning:     #d97706;
    --safe:        #059669;

    --sidebar-bg:  #0a0a0a;
    --panel-bg:    #ffffff;
    --topbar-bg:   rgba(10,10,10,0.98);

    --glow-accent: none;
    --glow-gold:   none;
    --glow-danger: none;
  }

  :global(*) { box-sizing: border-box; margin: 0; padding: 0; }

  :global(body) {
    background: var(--bg);
    color: var(--text);
    overflow: hidden;
    height: 100vh;
    font-family: 'Space Grotesk', sans-serif;
    transition: background 0.4s, color 0.4s;
  }

  :global(#app) { height: 100vh; width: 100vw; }

  :global(::-webkit-scrollbar) { width: 3px; }
  :global(::-webkit-scrollbar-thumb) { background: var(--border); }

  .app {
    display: flex;
    height: 100vh;
    width: 100vw;
    position: relative;
    overflow: hidden;
    background: var(--bg);
    transition: background 0.4s;
  }

  /* subtle noise texture overlay */
  .noise {
    position: fixed; inset: 0;
    opacity: 0.025;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 999;
  }

  [data-theme="dark"] .noise { opacity: 0.04; }

  .main {
    flex: 1; display: flex;
    flex-direction: column;
    min-width: 0; height: 100vh;
  }

  .topbar {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 0 24px; height: 46px; min-height: 46px;
    background: var(--topbar-bg);
    border-bottom: 1px solid var(--border-dim);
    backdrop-filter: blur(20px);
    z-index: 10;
  }

  .topbar-left { display: flex; align-items: center; gap: 10px; }

  .tb-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim);
    letter-spacing: 0.2em; text-transform: uppercase;
  }

  .tb-time {
    font-family: 'Limelight', cursive;
    font-size: 14px; color: var(--gold);
    letter-spacing: 0.06em;
  }

  .tb-sep { color: var(--border); font-size: 12px; }

  .tb-mission {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim);
    letter-spacing: 0.18em; text-transform: uppercase;
  }

  .topbar-right { display: flex; align-items: center; gap: 12px; }

  .theme-toggle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.12em;
    color: var(--violet); background: transparent;
    border: 1px solid var(--border);
    padding: 4px 12px; cursor: pointer;
    transition: all 0.2s;
  }
  .theme-toggle:hover { background: var(--glass); border-color: var(--accent); }

  .backend-status {
    display: flex; align-items: center; gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.1em;
    color: var(--danger);
  }
  .backend-status.online { color: var(--safe); }

  .bdot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--danger);
    animation: bdot 1.5s ease-in-out infinite;
  }
  .backend-status.online .bdot { background: var(--safe); }
  @keyframes bdot { 0%,100%{opacity:1} 50%{opacity:0.2} }

  .globe-toggle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.12em;
    color: var(--accent-hi); background: transparent;
    border: 1px solid var(--border);
    padding: 4px 12px; cursor: pointer;
    transition: all 0.2s;
  }
  .globe-toggle:hover { background: var(--glass); border-color: var(--accent); }

  .body {
    flex: 1; display: grid;
    grid-template-columns: 440px 1fr;
    min-height: 0; overflow: hidden;
  }

  .globe-pane {
    position: relative;
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border-dim);
    overflow: hidden;
  }

  [data-theme="dark"] .globe-pane {
    background: radial-gradient(ellipse at 50% 60%, #0e0530 0%, #05020f 100%);
  }

  .globe-stats {
    position: absolute; top: 14px; left: 14px;
    display: flex; align-items: center;
    background: rgba(5,2,15,0.85);
    border: 1px solid var(--border);
    backdrop-filter: blur(14px);
    overflow: hidden;
  }

  [data-theme="void"] .globe-stats {
    background: rgba(255,255,255,0.92);
    border-color: rgba(0,0,0,0.12);
  }

  .gs-item {
    padding: 8px 16px;
    display: flex; flex-direction: column;
    align-items: center; gap: 3px;
  }
  .gs-val {
    font-family: 'Limelight', cursive;
    font-size: 15px; color: var(--gold);
  }
  .gs-val.danger { color: var(--danger); }
  .gs-val.warn   { color: var(--warning); }
  .gs-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 7px; color: var(--text-dim);
    letter-spacing: 0.12em; text-transform: uppercase;
  }
  .gs-divider { width: 1px; height: 32px; background: var(--border-dim); }

  .globe-legend {
    position: absolute; bottom: 14px; right: 14px;
    display: flex; flex-direction: column; gap: 7px;
    background: rgba(5,2,15,0.8);
    border: 1px solid var(--border-dim);
    padding: 10px 14px; backdrop-filter: blur(10px);
  }

  [data-theme="void"] .globe-legend {
    background: rgba(255,255,255,0.92);
    border-color: rgba(0,0,0,0.1);
  }

  .leg-item {
    display: flex; align-items: center; gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim);
    letter-spacing: 0.1em;
  }
  .leg-dot { width: 7px; height: 7px; border-radius: 50%; }

  .right-panel {
    display: flex; flex-direction: column;
    overflow: hidden; background: var(--panel-bg);
    transition: background 0.4s;
  }
</style>
