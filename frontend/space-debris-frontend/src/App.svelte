<script>
  import { onMount } from 'svelte'
  import { fly, fade } from 'svelte/transition'
  import { cubicOut } from 'svelte/easing'
  import Sidebar from './components/Sidebar.svelte'
  import Globe3D from './components/Globe3D.svelte'
  import DashboardPanel from './components/DashboardPanel.svelte'
  import PredictPanel from './components/PredictPanel.svelte'
  import ConjunctionsPanel from './components/ConjunctionsPanel.svelte'
  import TelemetryPanel from './components/TelemetryPanel.svelte'
  import { activePanel, backendOnline, globeRotating, theme } from './stores/appStore.js'
  import { checkHealth } from './utils/api.js'

  let clockStr = '--:--:--'
  let prevPanel = 'dashboard'

  $: if ($activePanel !== prevPanel) { prevPanel = $activePanel }

  onMount(async () => {
    const tick = () => {
      const n = new Date()
      clockStr = n.toUTCString().split(' ')[4]
    }
    tick()
    const id = setInterval(tick, 1000)
    try { await checkHealth(); backendOnline.set(true) }
    catch { backendOnline.set(false) }
    return () => clearInterval(id)
  })

  function toggleTheme() {
    theme.update(t => t === 'dark' ? 'midnight' : 'dark')
  }

  const panelLabels = {
    dashboard:    'Mission Overview',
    predict:      'Risk Analysis',
    conjunctions: 'Conjunction Events',
    telemetry:    'Live Telemetry'
  }
</script>

<div class="app" data-theme={$theme}>
  <div class="grain" aria-hidden="true"></div>

  <Sidebar />

  <div class="main">

    <header class="topbar">
      <div class="topbar-left">
        <div class="time-block">
          <span class="time-label">UTC</span>
          <span class="time-val">{clockStr}</span>
        </div>
        <div class="divider-v"></div>
        <span class="mission-name">ASTRAEUS · SPACE DEBRIS SENTINEL</span>
      </div>

      <div class="topbar-center">
        <span class="panel-crumb">
          {panelLabels[$activePanel] ?? ''}
        </span>
      </div>

      <div class="topbar-right">
        <button class="theme-btn" on:click={toggleTheme}>
          <span class="theme-icon">{$theme === 'dark' ? '◐' : '◑'}</span>
          {$theme === 'dark' ? 'MIDNIGHT' : 'DARK'}
        </button>

        <div class="status-pill" class:online={$backendOnline}>
          <span class="status-dot"></span>
          {$backendOnline ? 'BACKEND ONLINE' : 'BACKEND OFFLINE'}
        </div>

        <button class="ctrl-btn" on:click={() => globeRotating.update(v => !v)}>
          {$globeRotating ? '⏸' : '▶'}
        </button>
      </div>
    </header>

    <div class="body">

      <div class="globe-pane">
        <Globe3D />

        <div class="globe-hud-tl">
          <div class="hud-val">{847}</div>
          <div class="hud-key">OBJECTS</div>
        </div>
        <div class="globe-hud-tr">
          <div class="hud-val danger">2</div>
          <div class="hud-key">CRITICAL</div>
        </div>

        <div class="globe-legend">
          <div class="leg-row"><span class="leg-dot" style="background:#ff3860"></span>CRITICAL</div>
          <div class="leg-row"><span class="leg-dot" style="background:#ff9020"></span>WARNING</div>
          <div class="leg-row"><span class="leg-dot" style="background:#00e8a0"></span>NOMINAL</div>
        </div>
      </div>

      <div class="right-pane">
        {#key $activePanel}
          <div
            class="panel-wrap"
            in:fly={{ x: 24, duration: 320, easing: cubicOut, delay: 60 }}
            out:fade={{ duration: 120 }}
          >
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
        {/key}
      </div>

    </div>
  </div>
</div>

<style>

  /* ══════════════════════════════════════
     DARK THEME  —  deep void purple
  ══════════════════════════════════════ */
  :global([data-theme="dark"]) {
    --bg:           #07041a;
    --bg2:          #0b0726;
    --surface:      #0f0b2e;
    --glass:        rgba(110,50,255,0.05);
    --border:       rgba(130,70,255,0.18);
    --border-dim:   rgba(130,70,255,0.08);

    --accent:       #7c3aed;
    --accent-hi:    #a78bfa;
    --accent-glow:  0 0 28px rgba(124,58,237,0.35);
    --gold:         #e2e8f0;
    --gold-dim:     rgba(240,192,64,0.5);
    --violet:       #c084fc;
    --text:         #ede9fe;
    --text-dim:     rgba(196,181,253,0.42);
    --text-mid:     rgba(220,210,255,0.7);

    --danger:       #ff3860;
    --warning:      #ff9020;
    --safe:         #00e8a0;

    --sidebar-bg:   #050315;
    --panel-bg:     #07041a;
    --topbar-bg:    rgba(5,3,20,0.97);
    --hud-bg:       rgba(5,3,20,0.82);

    --globe-bg:     radial-gradient(ellipse at 50% 55%, #110535 0%, #07041a 100%);
  }

  /* ══════════════════════════════════════
     MIDNIGHT THEME  —  deep blue + ice + gold
  ══════════════════════════════════════ */
  :global([data-theme="midnight"]) {
    --bg:           #0d1117;
    --bg2:          #0a0f1a;
    --surface:      #111827;
    --glass:        rgba(30,100,200,0.06);
    --border:       rgba(80,160,255,0.18);
    --border-dim:   rgba(80,160,255,0.08);

    --accent:       #2563eb;
    --accent-hi:    #93c5fd;
    --accent-glow:  0 0 28px rgba(37,99,235,0.35);
    --gold:         #e2e8f0;
    --gold-dim:     rgba(245,158,11,0.5);
    --violet:       #60a5fa;
    --text:         #e0f2fe;
    --text-dim:     rgba(147,197,253,0.45);
    --text-mid:     rgba(186,230,253,0.75);

    --danger:       #f43f5e;
    --warning:      #f97316;
    --safe:         #10b981;

    --sidebar-bg:   #080d14;
    --panel-bg:     #0d1117;
    --topbar-bg:    rgba(8,13,20,0.98);
    --hud-bg:       rgba(8,13,20,0.85);

    --globe-bg:     radial-gradient(ellipse at 50% 55%, #0a1f3d 0%, #0d1117 100%);
  }

  /* ── GLOBAL RESET ── */
  :global(*) { box-sizing: border-box; margin: 0; padding: 0; }

  :global(body) {
    background: var(--bg);
    color: var(--text);
    overflow: hidden;
    height: 100vh;
    font-family: 'Syne', sans-serif;
    transition: background 0.5s, color 0.5s;
  }

  :global(#app) { height: 100vh; width: 100vw; }

  :global(::-webkit-scrollbar) { width: 3px; }
  :global(::-webkit-scrollbar-thumb) { background: var(--border); border-radius: 2px; }

  /* grain texture */
  .grain {
    position: fixed; inset: 0; pointer-events: none; z-index: 900;
    opacity: 0.035;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E");
  }

  /* ── LAYOUT ── */
  .app {
    display: flex; height: 100vh; width: 100vw;
    overflow: hidden; background: var(--bg);
    transition: background 0.5s;
  }

  .main { flex: 1; display: flex; flex-direction: column; min-width: 0; height: 100vh; }

  /* ── TOPBAR ── */
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 24px; height: 48px; min-height: 48px;
    background: var(--topbar-bg);
    border-bottom: 1px solid var(--border-dim);
    backdrop-filter: blur(24px);
    z-index: 10; position: relative;
  }

  .topbar-left { display: flex; align-items: center; gap: 14px; }

  .time-block { display: flex; align-items: baseline; gap: 6px; }

  .time-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim); letter-spacing: 0.2em;
  }
  .time-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;   color: #e2e8f0; letter-spacing: 0.06em;
    font-weight: 500;
  }

  .divider-v { width: 1px; height: 22px; background: var(--border-dim); }

  .mission-name {
    font-family: 'Syne', sans-serif;
    font-size: 11px; font-weight: 700;
    color: var(--text-dim); letter-spacing: 0.18em;
    text-transform: uppercase;
  }

  .topbar-center {
    position: absolute; left: 50%; transform: translateX(-50%);
  }

  .panel-crumb {
    font-family: 'Syne', sans-serif;
    font-size: 12px; font-weight: 600;
    color: var(--text-mid); letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .topbar-right { display: flex; align-items: center; gap: 10px; }

  .theme-btn {
    display: flex; align-items: center; gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 500; letter-spacing: 0.14em;
    color: var(--violet); background: transparent;
    border: 1px solid var(--border-dim);
    padding: 5px 12px; cursor: pointer;
    transition: all 0.25s; text-transform: uppercase;
  }
  .theme-btn:hover { border-color: var(--accent); background: var(--glass); color: var(--accent-hi); }
  .theme-icon { font-size: 11px; }

  .status-pill {
    display: flex; align-items: center; gap: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.12em;
    color: var(--danger); padding: 5px 12px;
    border: 1px solid rgba(255,56,96,0.2);
    transition: all 0.3s;
  }
  .status-pill.online { color: var(--safe); border-color: rgba(0,232,160,0.2); }

  .status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--danger);
    animation: blink 1.5s ease-in-out infinite;
    flex-shrink: 0;
  }
  .status-pill.online .status-dot { background: var(--safe); }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

  .ctrl-btn {
    width: 32px; height: 32px;
    background: var(--glass); border: 1px solid var(--border-dim);
    color: var(--text-dim); font-size: 12px; cursor: pointer;
    transition: all 0.2s; display: flex; align-items: center; justify-content: center;
  }
  .ctrl-btn:hover { border-color: var(--border); color: var(--text); }

  /* ── BODY ── */
  .body {
    flex: 1; display: grid;
    grid-template-columns: 440px 1fr;
    min-height: 0; overflow: hidden;
  }

  /* ── GLOBE PANE ── */
  .globe-pane {
    position: relative; overflow: hidden;
    background: var(--globe-bg);
    border-right: 1px solid var(--border-dim);
    transition: background 0.5s;
  }

  .globe-hud-tl {
    position: absolute; top: 14px; left: 14px;
    background: var(--hud-bg);
    border: 1px solid var(--border);
    padding: 10px 16px; backdrop-filter: blur(14px);
    display: flex; flex-direction: column; align-items: center; gap: 3px;
  }

  .globe-hud-tr {
    position: absolute; top: 14px; right: 14px;
    background: var(--hud-bg);
    border: 1px solid rgba(255,56,96,0.25);
    padding: 10px 16px; backdrop-filter: blur(14px);
    display: flex; flex-direction: column; align-items: center; gap: 3px;
  }

  .hud-val {
    font-family: 'Syne', sans-serif;
    font-size: 22px; font-weight: 800;
    color: var(--gold); line-height: 1;
  }
  .hud-val.danger { color: var(--danger); }

  .hud-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim);
    letter-spacing: 0.15em; text-transform: uppercase;
  }

  .globe-legend {
    position: absolute; bottom: 14px; right: 14px;
    background: var(--hud-bg);
    border: 1px solid var(--border-dim);
    padding: 10px 14px; backdrop-filter: blur(12px);
    display: flex; flex-direction: column; gap: 8px;
  }

  .leg-row {
    display: flex; align-items: center; gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim); letter-spacing: 0.12em;
  }

  .leg-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

  /* ── RIGHT PANE ── */
  .right-pane {
    position: relative; overflow: hidden;
    background: var(--panel-bg);
    transition: background 0.5s;
  }

  .panel-wrap {
    position: absolute; inset: 0;
    overflow-y: auto; overflow-x: hidden;
  }
</style>
