<script>
  import { activePanel, alertCount, theme } from '../stores/appStore.js'

  const navItems = [
    { id: 'dashboard',    icon: '⬡', label: 'DASHBOARD',    sub: 'Overview' },
    { id: 'predict',      icon: '⟁', label: 'RISK PREDICT', sub: 'ML Analysis' },
    { id: 'conjunctions', icon: '◎', label: 'CONJUNCTIONS', sub: 'TCA Events' },
    { id: 'telemetry',    icon: '⎍', label: 'TELEMETRY',    sub: 'Live Feed' },
  ]
</script>

<nav class="sidebar" data-theme={$theme}>

  <div class="logo">
    <div class="logo-sigil">
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <circle cx="16" cy="16" r="14" stroke="var(--gold)" stroke-width="1" opacity="0.6"/>
        <circle cx="16" cy="16" r="8"  stroke="var(--accent)" stroke-width="1" opacity="0.8"/>
        <circle cx="16" cy="16" r="3"  fill="var(--gold)" opacity="0.9"/>
        <line x1="2"  y1="16" x2="30" y2="16" stroke="var(--accent)" stroke-width="0.5" opacity="0.3"/>
        <line x1="16" y1="2"  x2="16" y2="30" stroke="var(--accent)" stroke-width="0.5" opacity="0.3"/>
      </svg>
    </div>
    <div class="logo-text">
      <span class="logo-name">ASTRAEUS</span>
      <span class="logo-sub">SPACE DEBRIS SENTINEL</span>
    </div>
  </div>

  <div class="sys-status">
    <span class="pulse-dot"></span>
    <span class="sys-text">SYSTEM NOMINAL</span>
  </div>

  <div class="nav-section-label">NAVIGATION</div>

  <ul class="nav-list">
    {#each navItems as item}
      <li>
        <button
          class="nav-btn"
          class:active={$activePanel === item.id}
          on:click={() => activePanel.set(item.id)}
        >
          <span class="nav-icon">{item.icon}</span>
          <div class="nav-text">
            <span class="nav-label">{item.label}</span>
            <span class="nav-sub">{item.sub}</span>
          </div>
          {#if item.id === 'conjunctions' && $alertCount > 0}
            <span class="badge">{$alertCount}</span>
          {/if}
          {#if $activePanel === item.id}
            <span class="active-bar"></span>
          {/if}
        </button>
      </li>
    {/each}
  </ul>

  <div class="sidebar-footer">
    <div class="footer-row">
      <span class="fkey">TRACKED</span>
      <span class="fval">847</span>
    </div>
    <div class="footer-row">
      <span class="fkey">ALERTS</span>
      <span class="fval danger">{$alertCount}</span>
    </div>
    <div class="footer-row">
      <span class="fkey">SATS</span>
      <span class="fval safe">12</span>
    </div>
    <div class="footer-divider"></div>
    <div class="footer-version">v2.4.1 · ASTRAEUS</div>
  </div>

</nav>

<style>
  .sidebar {
    width: 210px; min-width: 210px; height: 100%;
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border-dim);
    display: flex; flex-direction: column;
    padding: 0;
    transition: background 0.4s;
  }

  .logo {
    display: flex; align-items: center; gap: 10px;
    padding: 20px 18px 18px;
    border-bottom: 1px solid var(--border-dim);
  }

  .logo-sigil { flex-shrink: 0; }

  .logo-text { display: flex; flex-direction: column; gap: 2px; }

  .logo-name {
    font-family: 'Limelight', cursive;
    font-size: 15px; color: var(--gold);
    letter-spacing: 0.12em;
    line-height: 1;
  }

  .logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 7px; color: var(--text-dim);
    letter-spacing: 0.12em; text-transform: uppercase;
  }

  .sys-status {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 18px;
    border-bottom: 1px solid var(--border-dim);
  }

  .pulse-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--safe);
    animation: pulse 2s ease-in-out infinite;
    flex-shrink: 0;
  }

  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.85)} }

  .sys-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--safe);
    letter-spacing: 0.14em; text-transform: uppercase;
  }

  .nav-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim);
    letter-spacing: 0.2em; text-transform: uppercase;
    padding: 14px 18px 6px;
  }

  .nav-list {
    list-style: none; flex: 1;
    display: flex; flex-direction: column;
    gap: 2px; padding: 4px 10px;
  }

  .nav-btn {
    width: 100%;
    display: flex; align-items: center; gap: 10px;
    padding: 10px 10px;
    background: none; border: none; cursor: pointer;
    color: var(--text-dim);
    text-align: left; transition: all 0.2s;
    position: relative;
    border-radius: 4px;
  }

  .nav-btn:hover {
    color: var(--accent-hi);
    background: var(--glass);
  }

  .nav-btn.active {
    color: var(--gold);
    background: rgba(232,184,75,0.07);
  }

  .nav-icon {
    font-size: 15px; width: 20px;
    text-align: center; flex-shrink: 0;
  }

  .nav-text {
    display: flex; flex-direction: column; gap: 1px;
    flex: 1;
  }

  .nav-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 500;
    letter-spacing: 0.12em; text-transform: uppercase;
    line-height: 1;
  }

  .nav-sub {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 9px; opacity: 0.5;
    line-height: 1;
  }

  .active-bar {
    position: absolute; left: 0; top: 20%; bottom: 20%;
    width: 2px; background: var(--gold);
    border-radius: 0 2px 2px 0;
  }

  .badge {
    background: var(--danger);
    color: white;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; padding: 1px 6px;
    border-radius: 10px;
  }

  .sidebar-footer {
    padding: 14px 18px;
    border-top: 1px solid var(--border-dim);
    display: flex; flex-direction: column; gap: 8px;
  }

  .footer-row {
    display: flex; justify-content: space-between; align-items: center;
  }

  .fkey {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim);
    letter-spacing: 0.15em; text-transform: uppercase;
  }

  .fval {
    font-family: 'Limelight', cursive;
    font-size: 14px; color: var(--accent-hi);
  }

  .fval.danger { color: var(--danger); }
  .fval.safe   { color: var(--safe); }

  .footer-divider {
    height: 1px; background: var(--border-dim); margin: 2px 0;
  }

  .footer-version {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim);
    letter-spacing: 0.1em; text-align: center;
  }
</style>
