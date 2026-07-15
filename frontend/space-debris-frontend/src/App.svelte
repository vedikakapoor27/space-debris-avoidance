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
  import HistoryPanel from './components/HistoryPanel.svelte'
  import { activePanel, backendOnline, globeRotating } from './stores/appStore.js'
  import { checkHealth } from './utils/api.js'
  import { theme } from './stores/appStore.js'

  let clockStr = '--:--:--'
  let cursorX = 0, cursorY = 0
  let cursorHover = false

  onMount(async () => {
    const tick = () => {
      const n = new Date()
      clockStr = n.toUTCString().split(' ')[4]
    }
    tick()
    const id = setInterval(tick, 1000)

    try { await checkHealth(); backendOnline.set(true) }
    catch { backendOnline.set(false) }

    const move = (e) => { cursorX = e.clientX; cursorY = e.clientY }
    const over = (e) => { cursorHover = !!e.target.closest('button, a, .hoverable') }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseover', over)

    return () => {
      clearInterval(id)
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseover', over)
    }
  })

  const panelLabels = {
    dashboard:    'Dashboard',
    predict:      'Risk Analysis',
    conjunctions: 'Conjunctions',
    telemetry:    'Telemetry',
    history:      'History'
  }
</script>

<!-- Custom cursor -->
<div
  class="cursor"
  class:cursor-hover={cursorHover}
  style="left:{cursorX}px; top:{cursorY}px"
></div>
<div
  class="cursor-dot"
  style="left:{cursorX}px; top:{cursorY}px"
></div>

<div class="app" data-theme={$theme}>
  <Sidebar />

  <div class="main">
    <header class="topbar">
      <div class="topbar-left">
        <div class="time-block">
          <span class="time-label">UTC</span>
          <span class="time-val">{clockStr}</span>
        </div>
        <div class="divider-v"></div>
        <span class="mission-label">ASTRAEUS — SPACE DEBRIS SENTINEL</span>
      </div>

      <div class="topbar-center">
        <span class="breadcrumb">
          {panelLabels[$activePanel] ?? ''}
        </span>
      </div>

      <div class="topbar-right">
        <div class="status-indicator" class:online={$backendOnline}>
          <span class="status-dot"></span>
          <span>{$backendOnline ? 'Backend Online' : 'Backend Offline'}</span>
        </div>
        <button class="live-btn hoverable" on:click={() => globeRotating.update(v => !v)}>
          <span class="live-dot"></span>
          {$globeRotating ? 'Live' : 'Paused'}
        </button>
<button class="theme-btn hoverable" on:click={() => theme.update(t => t === 'dark' ? 'light' : 'dark')}>
  {$theme === 'dark' ? '☀' : '☾'}
</button>
        <div class="avatar hoverable">SD</div>
      </div>
    </header>

    <div class="body">
      <div class="globe-pane">
        <Globe3D />
        <div class="globe-overlay-tl">
          <div class="ov-title">LIVE ORBITAL TRACKING</div>
          <div class="ov-sub">Drag to rotate · Scroll to zoom</div>
        </div>
        <div class="globe-legend">
          <div class="leg-item"><span class="leg-dot" style="background:#3b82f6"></span>Active Satellite</div>
          <div class="leg-item"><span class="leg-dot" style="background:#94a3b8"></span>Debris</div>
          <div class="leg-item"><span class="leg-dot" style="background:#ef4444"></span>Critical</div>
          <div class="leg-item"><span class="leg-dot" style="background:#f59e0b"></span>Warning</div>
        </div>
      </div>

      <div class="right-pane">
        {#key $activePanel}
          <div
            class="panel-wrap"
            in:fly={{ x: 16, duration: 280, easing: cubicOut, delay: 40 }}
            out:fade={{ duration: 100 }}
          >
            {#if $activePanel === 'dashboard'}
              <DashboardPanel />
            {:else if $activePanel === 'predict'}
              <PredictPanel />
            {:else if $activePanel === 'conjunctions'}
              <ConjunctionsPanel />
            {:else if $activePanel === 'telemetry'}
              <TelemetryPanel />
            {:else if $activePanel === 'history'}
              <HistoryPanel />
            {/if}
          </div>
        {/key}
      </div>
    </div>
  </div>
</div>

<style>
  :global(*) { box-sizing: border-box; margin: 0; padding: 0; }

  :global(:root) {
    --bg:        #0f172a;
    --bg2:       #111827;
    --surface:   #1e293b;
    --card:      #1f2937;
    --card2:     #243447;
    --border:    rgba(255,255,255,0.07);
    --border2:   rgba(255,255,255,0.12);
    --blue:      #3b82f6;
    --blue-dark: #2563eb;
    --success:   #10b981;
    --warning:   #f59e0b;
    --danger:    #ef4444;
    --text:      #f8fafc;
    --text-2:    #cbd5e1;
    --text-3:    #94a3b8;
    --text-4:    #64748b;
    --shadow:    0 1px 3px rgba(0,0,0,0.4), 0 4px 12px rgba(0,0,0,0.25);
    --shadow-lg: 0 4px 6px rgba(0,0,0,0.4), 0 10px 30px rgba(0,0,0,0.3);
    --radius:    8px;
    --radius-lg: 12px;
  }

  :global([data-theme="light"]) {
    --bg:        #f1f5f9;
    --bg2:       #ffffff;
    --surface:   #f8fafc;
    --card:      #ffffff;
    --card2:     #f8fafc;
    --border:    #e2e8f0;
    --border2:   #cbd5e1;
    --blue:      #2563eb;
    --blue-dark: #1d4ed8;
    --success:   #059669;
    --warning:   #d97706;
    --danger:    #dc2626;
    --text:      #0f172a;
    --text-2:    #1e293b;
    --text-3:    #475569;
    --text-4:    #94a3b8;
    --shadow:    0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
    --shadow-lg: 0 4px 10px rgba(0,0,0,0.08), 0 10px 30px rgba(0,0,0,0.06);
  }

  :global(body) {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    height: 100vh;
    overflow: hidden;
    -webkit-font-smoothing: antialiased;
    transition: background 0.3s, color 0.3s;
  }

  :global(#app) { height: 100vh; width: 100vw; }

  :global(::-webkit-scrollbar) { width: 4px; }
  :global(::-webkit-scrollbar-track) { background: transparent; }
  :global(::-webkit-scrollbar-thumb) { background: var(--border2); border-radius: 2px; }
  :global([data-theme="light"]) .stat-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

:global([data-theme="light"]) .section {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

:global([data-theme="light"]) .section-head {
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
}

:global([data-theme="light"]) .ctable th {
  background: #f8fafc;
  color: #475569;
}

:global([data-theme="light"]) .crow td {
  color: #1e293b;
  border-bottom: 1px solid #f1f5f9;
}

:global([data-theme="light"]) .crow:hover td {
  background: #f8fafc;
}

  /* CURSOR */
  .cursor {
    position: fixed; pointer-events: none; z-index: 9999;
    width: 28px; height: 28px;
    border: 1.5px solid rgba(255,255,255,0.5);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: width 0.2s, height 0.2s, border-color 0.2s, background 0.2s;
    mix-blend-mode: difference;
  }

  :global([data-theme="light"]) .cursor {
    border-color: rgba(0,0,0,0.4);
    mix-blend-mode: normal;
  }

  .cursor.cursor-hover {
    width: 40px; height: 40px;
    border-color: var(--blue);
    background: rgba(59,130,246,0.08);
  }

  .cursor-dot {
    position: fixed; pointer-events: none; z-index: 9999;
    width: 4px; height: 4px;
    background: var(--blue);
    border-radius: 50%;
    transform: translate(-50%, -50%);
  }

  /* LAYOUT */
  .app {
    display: flex; height: 100vh; width: 100vw;
    overflow: hidden; background: var(--bg);
    transition: background 0.3s;
  }

  .main {
    flex: 1; display: flex;
    flex-direction: column;
    min-width: 0; height: 100vh;
    overflow: hidden;
  }

  /* TOPBAR */
  .topbar {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    height: 52px; min-height: 52px;
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    z-index: 50;
    position: relative;
    flex-shrink: 0;
    box-shadow: var(--shadow);
  }

  .topbar-left { display: flex; align-items: center; gap: 14px; min-width: 0; }

  .time-block { display: flex; align-items: baseline; gap: 6px; flex-shrink: 0; }

  .time-label {
    font-size: 10px; font-weight: 500;
    color: var(--text-3); letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  .time-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 16px; font-weight: 600;
    color: var(--text); letter-spacing: 0.04em;
  }

  .divider-v {
    width: 1px; height: 20px;
    background: var(--border2);
    flex-shrink: 0;
  }

  .mission-label {
    font-size: 11px; font-weight: 500;
    color: var(--text-3);
    letter-spacing: 0.08em; text-transform: uppercase;
    white-space: nowrap;
  }

  .topbar-center {
    position: absolute; left: 50%;
    transform: translateX(-50%);
    pointer-events: none;
  }

  .breadcrumb {
    font-size: 13px; font-weight: 600;
    color: var(--text-2); letter-spacing: 0.02em;
  }

  .topbar-right {
    display: flex; align-items: center;
    gap: 8px; flex-shrink: 0;
  }

  .status-indicator {
    display: flex; align-items: center; gap: 6px;
    font-size: 12px; font-weight: 500;
    color: var(--danger);
    padding: 5px 12px; height: 32px;
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 6px;
    white-space: nowrap;
    transition: all 0.3s;
  }

  .status-indicator.online {
    color: var(--success);
    background: rgba(16,185,129,0.08);
    border-color: rgba(16,185,129,0.2);
  }

  .status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor;
    animation: blink 2s ease-in-out infinite;
    flex-shrink: 0;
  }

  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

  .live-btn {
    display: flex; align-items: center; gap: 6px;
    font-size: 12px; font-weight: 500;
    color: var(--text-2);
    padding: 5px 12px; height: 32px;
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 6px;
    cursor: none; transition: all 0.2s;
    white-space: nowrap;
  }

  .live-btn:hover { background: var(--card2); }

  .live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--success);
    animation: blink 1.5s infinite;
  }

  .theme-btn {
    width: 32px; height: 32px;
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 6px; cursor: none;
    font-size: 15px; color: var(--text-3);
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s; flex-shrink: 0;
  }

  .theme-btn:hover { background: var(--card2); color: var(--text); }

  .avatar {
    width: 32px; height: 32px; border-radius: 50%;
    background: var(--blue-dark);
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; color: white;
    cursor: none; flex-shrink: 0;
  }

  /* BODY */
  .body {
    flex: 1; display: grid;
    grid-template-columns: 420px 1fr;
    min-height: 0; overflow: hidden;
  }

  /* GLOBE PANE */
  .globe-pane {
    position: relative; overflow: hidden;
    background: #080e1a;
    border-right: 1px solid var(--border);
  }

  .globe-overlay-tl {
    position: absolute;
    top: 14px; left: 14px;
    pointer-events: none;
    z-index: 5;
  }

  .ov-title {
    font-size: 10px; font-weight: 600;
    color: rgba(148,163,184,0.7);
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 2px;
  }

  .ov-sub { font-size: 10px; color: rgba(100,116,139,0.6); }

  .globe-legend {
    position: absolute; bottom: 16px; right: 16px;
    display: flex; flex-direction: column; gap: 7px;
    background: rgba(8,14,26,0.88);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: var(--radius);
    padding: 10px 14px;
    backdrop-filter: blur(12px);
    z-index: 5;
  }

  .leg-item {
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; color: rgba(148,163,184,0.8);
  }

  .leg-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

  /* RIGHT PANE */
  .right-pane {
    position: relative; overflow: hidden;
    background: var(--bg);
    transition: background 0.3s;
  }

  .panel-wrap {
    position: absolute; inset: 0;
    overflow-y: auto; overflow-x: hidden;
  }
</style>