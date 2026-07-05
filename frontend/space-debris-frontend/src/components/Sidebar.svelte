<script>
  import { activePanel, alertCount } from '../stores/appStore.js'

  const navItems = [
    { id: 'dashboard',    icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>`, label: 'Dashboard',    sub: 'Overview' },
    { id: 'predict',      icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>`, label: 'Risk Predict', sub: 'AI Analysis' },
    { id: 'conjunctions', icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="8"/><line x1="12" y1="16" x2="12" y2="22"/></svg>`, label: 'Conjunctions', sub: 'TCA Events' },
    { id: 'telemetry',    icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>`, label: 'Telemetry',    sub: 'Live Feed' },
    { id: 'history',      icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`, label: 'History',      sub: 'Analytics' },
  ]
</script>

<nav class="sidebar">
  <div class="logo">
    <div class="logo-icon">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="#3b82f6" stroke-width="1.5"/>
        <circle cx="12" cy="12" r="4"  stroke="#3b82f6" stroke-width="1.5"/>
        <circle cx="12" cy="12" r="1"  fill="#3b82f6"/>
        <line x1="2" y1="12" x2="8"  y2="12" stroke="#3b82f6" stroke-width="1"/>
        <line x1="16" y1="12" x2="22" y2="12" stroke="#3b82f6" stroke-width="1"/>
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
    width: 220px; min-width: 220px; height: 100%;
    background: var(--bg2);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
  }

  .logo {
    display: flex; align-items: center; gap: 10px;
    padding: 18px 16px;
    border-bottom: 1px solid var(--border);
  }

  .logo-icon {
    width: 36px; height: 36px; border-radius: 8px;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.2);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }

  .logo-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14px; font-weight: 700; color: var(--text);
    letter-spacing: 0.08em; line-height: 1;
  }

  .logo-sub {
    font-size: 10px; color: var(--text-4);
    margin-top: 2px; letter-spacing: 0.02em;
  }

  .sys-status {
    display: flex; align-items: center; gap: 7px;
    padding: 9px 16px;
    border-bottom: 1px solid var(--border);
  }

  .sys-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--success); flex-shrink: 0;
    animation: pulse 2.5s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .sys-text {
    font-size: 11px; font-weight: 500; color: var(--success);
  }

  .nav-group-label {
    font-size: 10px; font-weight: 600; color: var(--text-4);
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 14px 16px 6px;
  }

  .nav-list { list-style: none; flex: 1; padding: 4px 8px; display: flex; flex-direction: column; gap: 1px; }

  .nav-item {
    width: 100%; display: flex; align-items: center; gap: 10px;
    padding: 9px 10px; background: none; border: none;
    color: var(--text-3); text-align: left; cursor: none;
    border-radius: var(--radius); transition: all 0.15s;
    position: relative;
  }

  .nav-item:hover { background: var(--surface); color: var(--text-2); }

  .nav-item.active {
    background: rgba(59,130,246,0.1);
    color: var(--blue);
    border: 1px solid rgba(59,130,246,0.15);
  }

  .nav-icon { width: 20px; flex-shrink: 0; display: flex; align-items: center; }

  .nav-text { display: flex; flex-direction: column; gap: 1px; flex: 1; }

  .nav-label { font-size: 13px; font-weight: 500; line-height: 1; }

  .nav-sub { font-size: 10px; opacity: 0.6; line-height: 1; }

  .badge {
    background: var(--danger); color: white;
    font-size: 10px; font-weight: 600;
    padding: 1px 6px; border-radius: 10px;
  }

  .sidebar-footer {
    padding: 14px 16px;
    border-top: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 12px;
  }

  .footer-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; }

  .fstat { text-align: center; }

  .fval {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px; font-weight: 700; color: var(--text); line-height: 1;
  }
  .fval.danger  { color: var(--danger); }
  .fval.success { color: var(--success); }

  .fkey { font-size: 9px; color: var(--text-4); margin-top: 3px; letter-spacing: 0.06em; }

  .footer-version {
    font-size: 10px; color: var(--text-4); text-align: center;
  }
</style>