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
  <button class="theme-btn" on:click={() => theme.update(t => t === 'dark' ? 'light' : 'dark')}>
  {#if $theme === 'dark'}
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
  {:else}
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <circle cx="12" cy="12" r="5"/>
      <line x1="12" y1="1" x2="12" y2="3"/>
      <line x1="12" y1="21" x2="12" y2="23"/>
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
      <line x1="1" y1="12" x2="3" y2="12"/>
      <line x1="21" y1="12" x2="23" y2="12"/>
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
    </svg>
  {/if}
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
  --bg:         #ffffff;
  --bg2:        #fafafa;
  --surface:    #f4f4f4;
  --card:       #ffffff;
  --card2:      #f9f9f9;
  --border:     #e0e0e0;
  --border2:    #c8c8c8;
  --divider:    #ebebeb;

  --text:       #0a0a0a;
  --text-2:     #1a1a1a;
  --text-3:     #444444;
  --text-4:     #888888;

  --danger:     #0a0a0a;
  --warning:    #2a2a2a;
  --success:    #444444;

  --danger-bg:  rgba(0,0,0,0.05);
  --warning-bg: rgba(0,0,0,0.03);
  --success-bg: rgba(0,0,0,0.02);

  --shadow:     0 1px 4px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.04);
  --shadow-lg:  0 4px 16px rgba(0,0,0,0.1);

  --globe-bg:   #08080f;
  --sidebar-bg: #ffffff;
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
    font-family: 'Inter', sans-serif;
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
  }
  .theme-btn:hover { border-color: var(--border2); color: var(--text); }

  .avatar {
    width: 28px; height: 28px;
    border: 1px solid var(--border2);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Inter', monospace;
    font-size: 10px; font-weight: 500; color: var(--text-2);
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
    font-family: 'Inter', monospace;
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
  /* Light mode — stronger contrast everywhere */
:global([data-theme="light"]) .stat-card {
  background: #ffffff;
  border: 1px solid #e0e0e0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

:global([data-theme="light"]) .section {
  background: #ffffff;
  border: 1px solid #e0e0e0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

:global([data-theme="light"]) .section-head {
  background: #f4f4f4;
  border-bottom: 1px solid #e0e0e0;
}

:global([data-theme="light"]) .ctable th {
  background: #f4f4f4;
  color: #444444;
  border-bottom: 1px solid #e0e0e0;
}

:global([data-theme="light"]) .crow td {
  color: #1a1a1a;
  border-bottom: 1px solid #f0f0f0;
}

:global([data-theme="light"]) .crow:hover td {
  background: #f9f9f9;
}

:global([data-theme="light"]) .pill.high {
  color: #0a0a0a;
  border-color: #0a0a0a;
  background: rgba(0,0,0,0.06);
}

:global([data-theme="light"]) .pill.med {
  color: #333333;
  border-color: #888888;
  background: rgba(0,0,0,0.03);
}

:global([data-theme="light"]) .pill.low {
  color: #666666;
  border-color: #bbbbbb;
  background: rgba(0,0,0,0.02);
}

:global([data-theme="light"]) .sidebar {
  border-right: 1px solid #e0e0e0;
}

:global([data-theme="light"]) .nav-item.active {
  background: #f0f0f0;
  color: #0a0a0a;
}

:global([data-theme="light"]) .topbar {
  border-bottom: 1px solid #e0e0e0;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
:global(*) {
  cursor: auto !important;
}
</style>