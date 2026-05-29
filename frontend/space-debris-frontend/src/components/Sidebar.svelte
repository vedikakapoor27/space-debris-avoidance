<script>
  import { activePanel, alertCount, theme } from '../stores/appStore.js'
  import { fly } from 'svelte/transition'
  import { cubicOut } from 'svelte/easing'

  const navItems = [
    { id: 'dashboard',    icon: '⬡', label: 'Dashboard',    sub: 'Overview' },
    { id: 'predict',      icon: '⟁', label: 'Risk Predict', sub: 'ML Analysis' },
    { id: 'conjunctions', icon: '◎', label: 'Conjunctions', sub: 'TCA Events' },
    { id: 'telemetry',    icon: '⎍', label: 'Telemetry',    sub: 'Live Feed' },
  ]

  function go(id) { activePanel.set(id) }
</script>

<nav class="sidebar">

  <div class="logo">
    <div class="sigil">
      <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
        <circle cx="18" cy="18" r="16" stroke="var(--gold)" stroke-width="0.8" opacity="0.5"/>
        <circle cx="18" cy="18" r="10" stroke="var(--accent-hi)" stroke-width="0.8" opacity="0.7"/>
        <circle cx="18" cy="18" r="4"  fill="var(--gold)" opacity="0.9"/>
        <line x1="2"  y1="18" x2="34" y2="18" stroke="var(--accent-hi)" stroke-width="0.4" opacity="0.25"/>
        <line x1="18" y1="2"  x2="18" y2="34" stroke="var(--accent-hi)" stroke-width="0.4" opacity="0.25"/>
        <circle cx="18" cy="18" r="16" stroke="var(--accent)" stroke-width="0.4" stroke-dasharray="3 6" opacity="0.3">
          <animateTransform attributeName="transform" type="rotate" from="0 18 18" to="360 18 18" dur="30s" repeatCount="indefinite"/>
        </circle>
      </svg>
    </div>
    <div class="logo-text">
      <span class="logo-name">ASTRAEUS</span>
      <span class="logo-sub">Space Debris Sentinel</span>
    </div>
  </div>

  <div class="sys-pill">
    <span class="sys-dot"></span>
    <span class="sys-label">System Nominal</span>
  </div>

  <div class="nav-label-sec">Navigation</div>

  <ul class="nav-list">
    {#each navItems as item, i}
      <li>
        <button
          class="nav-item"
          class:active={$activePanel === item.id}
          on:click={() => go(item.id)}
          style="animation-delay:{i*60}ms"
        >
          <span class="nav-ico">{item.icon}</span>
          <div class="nav-txt">
            <span class="nav-lbl">{item.label}</span>
            <span class="nav-sub">{item.sub}</span>
          </div>

          {#if item.id === 'conjunctions' && $alertCount > 0}
            <span class="badge">{$alertCount}</span>
          {/if}

          <span class="nav-arrow" class:visible={$activePanel === item.id}>›</span>

          {#if $activePanel === item.id}
            <span class="active-line"></span>
          {/if}
        </button>
      </li>
    {/each}
  </ul>

  <div class="footer">
    <div class="footer-stats">
      <div class="fstat">
        <span class="fval">{847}</span>
        <span class="fkey">Tracked</span>
      </div>
      <div class="fstat-div"></div>
      <div class="fstat">
        <span class="fval" style="color:var(--danger)">{$alertCount}</span>
        <span class="fkey">Alerts</span>
      </div>
      <div class="fstat-div"></div>
      <div class="fstat">
        <span class="fval" style="color:var(--safe)">12</span>
        <span class="fkey">Sats</span>
      </div>
    </div>
    <div class="footer-ver">v2.4.1 · ASTRAEUS</div>
  </div>

</nav>

<style>
  .sidebar {
    width: 214px; min-width: 214px; height: 100%;
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border-dim);
    display: flex; flex-direction: column;
    transition: background 0.5s;
    position: relative; z-index: 20;
  }

  .logo {
    display: flex; align-items: center; gap: 12px;
    padding: 18px 16px;
    border-bottom: 1px solid var(--border-dim);
  }

  .sigil { flex-shrink: 0; }

  .logo-text { display: flex; flex-direction: column; gap: 2px; }

  .logo-name {
    font-family: 'Syne', sans-serif;
    font-size: 15px; font-weight: 800;
    color: var(--gold); letter-spacing: 0.1em; line-height: 1;
  }

  .logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 7.5px; color: var(--text-dim);
    letter-spacing: 0.1em;
  }

  .sys-pill {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 16px;
    border-bottom: 1px solid var(--border-dim);
  }

  .sys-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--safe); flex-shrink: 0;
    animation: pulse 2.5s ease-in-out infinite;
  }

  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.8)} }

  .sys-label {
    font-family: 'Syne', sans-serif;
    font-size: 10px; font-weight: 600;
    color: var(--safe); letter-spacing: 0.06em;
  }

  .nav-label-sec {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim);
    letter-spacing: 0.22em; text-transform: uppercase;
    padding: 14px 16px 6px;
  }

  .nav-list {
    list-style: none; flex: 1;
    display: flex; flex-direction: column;
    gap: 2px; padding: 4px 8px;
  }

  .nav-item {
    width: 100%; display: flex; align-items: center; gap: 10px;
    padding: 11px 10px;
    background: none; border: none; cursor: pointer;
    color: var(--text-dim); text-align: left;
    transition: color 0.2s, background 0.2s;
    position: relative; border-radius: 6px;
  }

  .nav-item:hover {
    color: var(--text-mid);
    background: var(--glass);
  }

  .nav-item.active {
    color: var(--gold);
    background: rgba(240,192,64,0.06);
  }

  .nav-ico {
    font-size: 16px; width: 22px; text-align: center; flex-shrink: 0;
    transition: transform 0.2s;
  }

  .nav-item:hover .nav-ico { transform: scale(1.1); }
  .nav-item.active .nav-ico { color: var(--gold); }

  .nav-txt { display: flex; flex-direction: column; gap: 1px; flex: 1; }

  .nav-lbl {
    font-family: 'Syne', sans-serif;
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.04em; line-height: 1;
    text-transform: uppercase;
  }

  .nav-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; opacity: 0.5; line-height: 1;
  }

  .nav-arrow {
    font-size: 18px; color: var(--text-dim);
    opacity: 0; transform: translateX(-4px);
    transition: opacity 0.2s, transform 0.2s;
    line-height: 1;
  }
  .nav-arrow.visible {
    opacity: 1; color: var(--gold); transform: translateX(0);
  }
  .nav-item:hover .nav-arrow { opacity: 0.4; transform: translateX(0); }

  .active-line {
    position: absolute; left: 0; top: 18%; bottom: 18%;
    width: 2.5px; background: var(--gold);
    border-radius: 0 2px 2px 0;
    animation: linein 0.25s ease-out;
  }
  @keyframes linein { from{height:0;opacity:0} to{opacity:1} }

  .badge {
    background: var(--danger); color: white;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; padding: 1px 6px; border-radius: 10px;
    animation: pulse 1s infinite;
  }

  .footer {
    padding: 14px 16px;
    border-top: 1px solid var(--border-dim);
    display: flex; flex-direction: column; gap: 10px;
  }

  .footer-stats {
    display: flex; align-items: center; justify-content: space-between;
  }

  .fstat {
    display: flex; flex-direction: column;
    align-items: center; gap: 3px; flex: 1;
  }

  .fval {
    font-family: 'Syne', sans-serif;
    font-size: 16px; font-weight: 800;
    color: var(--accent-hi); line-height: 1;
  }

  .fkey {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim);
    letter-spacing: 0.12em; text-transform: uppercase;
  }

  .fstat-div { width: 1px; height: 28px; background: var(--border-dim); }

  .footer-ver {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim);
    letter-spacing: 0.1em; text-align: center;
  }
</style>
