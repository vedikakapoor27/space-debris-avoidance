<script>
  import { activePanel, alertCount } from '../stores/appStore.js'

 const navItems = [
  { 
    id: 'dashboard',
    icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <circle cx="12" cy="12" r="3"/>
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>
      <path d="M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/>
    </svg>`,
    label: 'Dashboard', sub: 'Overview'
  },
  {
    id: 'predict',
    icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/>
      <path d="M12 8v4l3 3"/>
    </svg>`,
    label: 'Risk Predict', sub: 'AI Analysis'
  },
  {
    id: 'conjunctions',
    icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/>
      <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
    </svg>`,
    label: 'Conjunctions', sub: 'TCA Events'
  },
  {
    id: 'telemetry',
    icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
    </svg>`,
    label: 'Telemetry', sub: 'Live Feed'
  },
  {
    id: 'history',
    icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path d="M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z"/>
    </svg>`,
    label: 'History', sub: 'Analytics'
  },
]
</script>

<nav class="sidebar">
  <div class="logo">
    <div class="logo-icon">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
  <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="1.5" opacity="0.8"/>
  <circle cx="12" cy="12" r="4"  stroke="currentColor" stroke-width="1.5" opacity="0.6"/>
  <circle cx="12" cy="12" r="1"  fill="currentColor"/>
  <line x1="2" y1="12" x2="8"  y2="12" stroke="currentColor" stroke-width="1" opacity="0.4"/>
  <line x1="16" y1="12" x2="22" y2="12" stroke="currentColor" stroke-width="1" opacity="0.4"/>
</svg>
    </div>
    <div>
      <div class="logo-name">ASTRAEUS</div>
      <div class="logo-sub">Space Debris Sentinel</div>
    </div>
  </div>

  <div class="sys-status">
    <span class="sys-dot"></span>
    <span class="sys-text">System Nominal</span>
  </div>

  <div class="nav-group-label">Navigation</div>

  <ul class="nav-list">
    {#each navItems as item}
      <li>
        <button
          class="nav-item hoverable"
          class:active={$activePanel === item.id}
          on:click={() => activePanel.set(item.id)}
        >
          <span class="nav-icon">{@html item.icon}</span>
          <div class="nav-text">
            <span class="nav-label">{item.label}</span>
            <span class="nav-sub">{item.sub}</span>
          </div>
          {#if item.id === 'conjunctions' && $alertCount > 0}
            <span class="badge">{$alertCount}</span>
          {/if}
        </button>
      </li>
    {/each}
  </ul>

  <div class="sidebar-footer">
    <div class="footer-grid">
      <div class="fstat">
        <div class="fval">847</div>
        <div class="fkey">Tracked</div>
      </div>
      <div class="fstat">
        <div class="fval danger">{$alertCount}</div>
        <div class="fkey">Alerts</div>
      </div>
      <div class="fstat">
        <div class="fval success">12</div>
        <div class="fkey">Satellites</div>
      </div>
    </div>
    <div class="footer-version">v2.4.1 · ASTRAEUS</div>
  </div>
</nav>

<style>
  .sidebar {
    width: 200px; min-width: 200px; height: 100%;
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    transition: background 0.3s;
    z-index: 40;
  }

  .logo {
    display: flex; align-items: center; gap: 10px;
    padding: 14px 14px;
    border-bottom: 1px solid var(--border);
    min-height: 58px; flex-shrink: 0;
  }

  .logo-icon {
  width: 30px; height: 30px;
  border: 1px solid var(--border2);
  color: var(--text-3);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
  .logo-name {
    font-family: 'Inter', sans-serif;
    font-size: 13px; font-weight: 700;
    color: var(--text); letter-spacing: 0.1em;
    line-height: 1; white-space: nowrap;
    text-transform: uppercase;
  }

  .logo-sub {
    font-family: 'Inter', monospace;
    font-size: 8px; color: var(--text-4);
    line-height: 1; white-space: nowrap;
    letter-spacing: 0.08em; margin-top: 3px;
    text-transform: uppercase;
  }

  .sys-status {
    display: flex; align-items: center; gap: 7px;
    padding: 7px 14px;
    border-bottom: 1px solid var(--divider);
    flex-shrink: 0;
  }

  .sys-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--text-3); flex-shrink: 0;
    animation: pulse 3s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

  .sys-text {
    font-family: 'Inter', monospace;
    font-size: 8px; font-weight: 500;
    color: var(--text-3); letter-spacing: 0.14em;
    text-transform: uppercase;
  }

  .nav-group-label {
    font-family: 'Inter', monospace;
    font-size: 8px; color: var(--text-4);
    letter-spacing: 0.18em; text-transform: uppercase;
    padding: 12px 14px 5px;
  }

  .nav-list {
    list-style: none; flex: 1;
    padding: 2px 6px;
    display: flex; flex-direction: column; gap: 1px;
    overflow-y: auto;
  }

  .nav-item {
    width: 100%; display: flex; align-items: center; gap: 9px;
    padding: 8px 10px; background: none; border: none;
    color: var(--text-4); text-align: left; cursor: none;
    transition: background 0.15s, color 0.15s;
    position: relative;
  }

  .nav-item:hover { background: var(--surface); color: var(--text-3); }

  .nav-item.active {
    background: var(--surface);
    color: var(--text);
    border-left: 1px solid var(--text);
  }

  .nav-item.active::before {
    content: '';
    position: absolute; left: 0; top: 0; bottom: 0;
    width: 1px; background: var(--text);
  }

  .nav-icon {
    width: 16px; flex-shrink: 0;
    display: flex; align-items: center;
    opacity: 0.6;
  }
  .nav-item.active .nav-icon { opacity: 1; }
  .nav-item:hover .nav-icon  { opacity: 0.8; }

  .nav-text { display: flex; flex-direction: column; gap: 1px; flex: 1; }

  .nav-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; font-weight: 500;
    letter-spacing: 0.1em; line-height: 1;
    text-transform: uppercase;
  }

  .nav-sub {
    font-family: 'Inter', monospace;
    font-size: 8px; opacity: 0.5; line-height: 1;
    letter-spacing: 0.06em;
  }

  .badge {
    background: var(--text); color: var(--bg);
    font-family: 'Inter', monospace;
    font-size: 8px; font-weight: 600;
    padding: 1px 5px;
  }

  .sidebar-footer {
    padding: 12px 14px;
    border-top: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 10px;
    flex-shrink: 0;
  }

  .footer-grid {
    display: grid; grid-template-columns: repeat(3,1fr); gap: 6px;
  }

  .fstat { text-align: center; }

  .fval {
    font-family: 'Inter', sans-serif;
    font-size: 16px; font-weight: 700;
    color: var(--text); line-height: 1;
  }
  .fval.danger  { color: var(--danger); }
  .fval.success { color: var(--text-3); }

  .fkey {
    font-family: 'Inter', monospace;
    font-size: 7px; color: var(--text-4);
    margin-top: 3px; letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  .footer-version {
    font-family: 'Inter', monospace;
    font-size: 8px; color: var(--text-4);
    text-align: center; letter-spacing: 0.08em;
  }
</style> 