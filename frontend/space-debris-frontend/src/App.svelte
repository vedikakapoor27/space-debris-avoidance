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
    {$backendOnline ? 'ONLINE' : 'OFFLINE'}
  </div>
  <button class="topbar-btn hoverable" on:click={() => globeRotating.update(v => !v)}>
    <span class="live-dot"></span>
    {$globeRotating ? 'LIVE' : 'HOLD'}
  </button>
  <button class="theme-btn hoverable" on:click={() => theme.update(t => t === 'dark' ? 'light' : 'dark')}>
    {$theme === 'dark' ? '○' : '●'}
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
  <div class="leg-item"><span class="leg-dot" style="background:#ffffff"></span>CRITICAL</div>
  <div class="leg-item"><span class="leg-dot" style="background:#888888"></span>WARNING</div>
  <div class="leg-item"><span class="leg-dot" style="background:#444444"></span>NOMINAL</div>
  <div class="leg-item"><span class="leg-dot" style="background:#aaaaaa"></span>SATELLITE</div>
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
  /* ══════════════════════════════════
     DARK THEME — matte black
  ══════════════════════════════════ */
  :global(:root),
  :global([data-theme="dark"]) {
    --bg:         #000000;
    --bg2:        #0b0b0b;
    --surface:    #111111;
    --card:       #0d0d0d;
    --card2:      #141414;
    --border:     #2b2b2b;
    --border2:    #3a3a3a;
    --divider:    #1c1c1c;

    --text:       #ffffff;
    --text-2:     #d4d4d4;
    --text-3:     #a8a8a8;
    --text-4:     #5a5a5a;

    --danger:     #ffffff;
    --warning:    #d4d4d4;
    --success:    #a8a8a8;

    --danger-bg:  rgba(255,255,255,0.06);
    --warning-bg: rgba(255,255,255,0.03);
    --success-bg: rgba(255,255,255,0.02);

    --shadow:     0 1px 4px rgba(0,0,0,0.8);
    --shadow-lg:  0 4px 20px rgba(0,0,0,0.9);
    --radius:     6px;

    --globe-bg:   #000000;
    --sidebar-bg: #080808;
    --topbar-bg:  #050505;
  }

  /* ══════════════════════════════════
     LIGHT THEME — white aerospace
  ══════════════════════════════════ */
  :global([data-theme="light"]) {
    --bg:         #f5f5f5;
    --bg2:        #ffffff;
    --surface:    #eeeeee;
    --card:       #ffffff;
    --card2:      #f9f9f9;
    --border:     #d0d0d0;
    --border2:    #b8b8b8;
    --divider:    #e4e4e4;

    --text:       #000000;
    --text-2:     #1a1a1a;
    --text-3:     #555555;
    --text-4:     #999999;

    --danger:     #000000;
    --warning:    #333333;
    --success:    #555555;

    --danger-bg:  rgba(0,0,0,0.06);
    --warning-bg: rgba(0,0,0,0.03);
    --success-bg: rgba(0,0,0,0.02);

    --shadow:     0 1px 4px rgba(0,0,0,0.08);
    --shadow-lg:  0 4px 20px rgba(0,0,0,0.12);

    --globe-bg:   #0a0a12;
    --sidebar-bg: #fafafa;
    --topbar-bg:  #ffffff;
  }

  /* ── GLOBAL ── */
  :global(*) { box-sizing: border-box; margin: 0; padding: 0; }

  :global(body) {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter' ,sans-serif;
    font-size: 13px;
    height: 100vh;
    overflow: hidden;
    -webkit-font-smoothing: antialiased;
    letter-spacing: 0.01em;
    transition: background 0.3s, color 0.3s;
  }

  :global(#app) { height: 100vh; width: 100vw; }

  :global(::-webkit-scrollbar) { width: 3px; }
  :global(::-webkit-scrollbar-track) { background: transparent; }
  :global(::-webkit-scrollbar-thumb) { background: var(--border); }

  /* ── CURSOR ── */
  .cursor {
    position: fixed; pointer-events: none; z-index: 9999;
    width: 24px; height: 24px;
    border: 1px solid var(--text);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: width 0.2s, height 0.2s, opacity 0.2s;
    opacity: 0.7;
  }

  .cursor.cursor-hover {
    width: 36px; height: 36px; opacity: 1;
  }

  .cursor-dot {
    position: fixed; pointer-events: none; z-index: 9999;
    width: 3px; height: 3px;
    background: var(--text);
    border-radius: 50%;
    transform: translate(-50%, -50%);
  }

  /* ── LAYOUT ── */
  .app {
    display: flex; height: 100vh; width: 100vw;
    overflow: hidden; background: var(--bg);
    transition: background 0.3s;
  }

  .main {
    flex: 1; display: flex; flex-direction: column;
    min-width: 0; height: 100vh; overflow: hidden;
  }

  /* ── TOPBAR ── */
  .topbar {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 0 20px; height: 44px; min-height: 44px;
    background: var(--topbar-bg);
    border-bottom: 1px solid var(--border);
    z-index: 50; position: relative; flex-shrink: 0;
    transition: background 0.3s;
  }

  .topbar-left { display: flex; align-items: center; gap: 16px; }

  .time-block { display: flex; align-items: baseline; gap: 6px; }

  .time-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; color: var(--text-4);
    letter-spacing: 0.15em; text-transform: uppercase;
  }

  .time-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px; font-weight: 500;
    color: var(--text); letter-spacing: 0.08em;
  }

  .divider-v {
    width: 1px; height: 16px; background: var(--border);
  }

  .mission-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; color: var(--text-4);
    letter-spacing: 0.15em; text-transform: uppercase;
  }

  .topbar-center {
    position: absolute; left: 50%;
    transform: translateX(-50%);
    pointer-events: none;
  }

  .breadcrumb {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px; font-weight: 600;
    color: var(--text-3); letter-spacing: 0.14em;
    text-transform: uppercase;
  }

  .topbar-right { display: flex; align-items: center; gap: 8px; }

  .status-indicator {
    display: flex; align-items: center; gap: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; letter-spacing: 0.12em;
    color: var(--text-3); padding: 4px 10px; height: 28px;
    border: 1px solid var(--border);
    text-transform: uppercase;
    transition: all 0.3s;
  }

  .status-indicator.online { color: var(--text-2); border-color: var(--border2); }

  .status-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--text-4);
    animation: blink 3s ease-in-out infinite;
  }
  .status-indicator.online .status-dot { background: var(--text-2); }

  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

  .topbar-btn {
    display: flex; align-items: center; gap: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 500;
    color: var(--text-3); padding: 4px 10px; height: 28px;
    background: transparent;
    border: 1px solid var(--border);
    cursor: none; transition: all 0.2s;
    letter-spacing: 0.1em; text-transform: uppercase;
  }

  .topbar-btn:hover { border-color: var(--border2); color: var(--text); }

  .live-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--text);
    animation: blink 2s infinite;
  }

  .theme-btn {
    width: 28px; height: 28px;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-3); font-size: 13px;
    display: flex; align-items: center; justify-content: center;
    cursor: none; transition: all 0.2s;
  }
  .theme-btn:hover { border-color: var(--border2); color: var(--text); }

  .avatar {
    width: 28px; height: 28px;
    border: 1px solid var(--border2);
    display: flex; align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; font-weight: 500; color: var(--text-2);
    cursor: none; letter-spacing: 0.05em;
  }

  /* ── BODY ── */
  .body {
    flex: 1; display: grid;
    grid-template-columns: 400px 1fr;
    min-height: 0; overflow: hidden;
  }

  /* ── GLOBE PANE ── */
  .globe-pane {
    position: relative; overflow: hidden;
    background: var(--globe-bg);
    border-right: 1px solid var(--border);
  }

  .globe-overlay-tl {
    position: absolute; top: 12px; left: 12px;
    pointer-events: none; z-index: 5;
  }

  .ov-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; font-weight: 500;
    color: rgba(168,168,168,0.6);
    letter-spacing: 0.2em; text-transform: uppercase;
    margin-bottom: 2px;
  }

  .ov-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; color: rgba(90,90,90,0.8);
    letter-spacing: 0.08em;
  }

  .globe-legend {
    position: absolute; bottom: 12px; right: 12px;
    display: flex; flex-direction: column; gap: 6px;
    background: rgba(0,0,0,0.85);
    border: 1px solid #2b2b2b;
    padding: 10px 12px; z-index: 5;
  }

  :global([data-theme="light"]) .globe-legend {
    background: rgba(10,10,18,0.92);
  }

  .leg-item {
    display: flex; align-items: center; gap: 7px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; color: rgba(168,168,168,0.8);
    letter-spacing: 0.1em; text-transform: uppercase;
  }

  .leg-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }

  /* ── RIGHT PANE ── */
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