<script>
  import { activePanel, alertCount } from '../stores/appStore.js'

  const navItems = [
    { id: 'dashboard',     icon: '⬡', label: 'DASHBOARD' },
    { id: 'predict',       icon: '⟁', label: 'RISK PREDICT' },
    { id: 'conjunctions',  icon: '◎', label: 'CONJUNCTIONS' },
    { id: 'telemetry',     icon: '⎍', label: 'TELEMETRY' },
  ]
</script>

<nav class="sidebar">
  <div class="logo">
    <div class="logo-mark">⬡</div>
    <div class="logo-text">
      <span class="logo-name">ORION</span>
      <span class="logo-sub">DEBRIS AVOIDANCE v1.0</span>
    </div>
  </div>

  <div class="status-dot">
    <span class="dot pulse"></span>
    <span class="status-text">SYSTEM NOMINAL</span>
  </div>

  <ul class="nav-list">
    {#each navItems as item}
      <li>
        <button
          class="nav-btn"
          class:active={$activePanel === item.id}
          on:click={() => activePanel.set(item.id)}
        >
          <span class="nav-icon">{item.icon}</span>
          <span class="nav-label">{item.label}</span>
          {#if item.id === 'conjunctions' && $alertCount > 0}
            <span class="badge">{$alertCount}</span>
          {/if}
        </button>
      </li>
    {/each}
  </ul>

  <div class="sidebar-footer">
    <div class="footer-stat">
      <span class="fstat-val">847</span>
      <span class="fstat-key">TRACKED</span>
    </div>
    <div class="footer-stat">
      <span class="fstat-val" style="color: #ff2244">{$alertCount}</span>
      <span class="fstat-key">ALERTS</span>
    </div>
    <div class="footer-stat">
      <span class="fstat-val" style="color:#00ff88">12</span>
      <span class="fstat-key">SATS</span>
    </div>
  </div>
</nav>

<style>
  .sidebar {
    width: 200px;
    min-width: 200px;
    height: 100%;
    background: linear-gradient(180deg, #00081a 0%, #000d26 100%);
    border-right: 1px solid rgba(0, 229, 255, 0.1);
    display: flex;
    flex-direction: column;
    padding: 20px 0;
    gap: 0;
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 18px 20px;
    border-bottom: 1px solid rgba(0,229,255,0.08);
  }
  .logo-mark {
    font-size: 24px;
    color: #00e5ff;
    line-height: 1;
    filter: drop-shadow(0 0 8px #00e5ff);
  }
  .logo-text { display: flex; flex-direction: column; }
  .logo-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #00e5ff;
    letter-spacing: 3px;
  }
  .logo-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 7px;
    color: rgba(0,229,255,0.4);
    letter-spacing: 1px;
  }

  .status-dot {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 18px;
  }
  .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 8px #00ff88;
  }
  .dot.pulse { animation: pulse 2s infinite; }
  @keyframes pulse {
    0%,100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
  .status-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: #00ff88;
    letter-spacing: 2px;
  }

  .nav-list {
    list-style: none;
    padding: 10px 0;
    margin: 0;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .nav-btn {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 18px;
    background: none;
    border: none;
    cursor: pointer;
    color: rgba(255,255,255,0.4);
    font-family: 'Rajdhani', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-align: left;
    transition: all 0.2s;
    position: relative;
  }
  .nav-btn:hover {
    color: rgba(0,229,255,0.8);
    background: rgba(0,229,255,0.05);
  }
  .nav-btn.active {
    color: #00e5ff;
    background: rgba(0,229,255,0.08);
    border-left: 2px solid #00e5ff;
  }
  .nav-icon {
    font-size: 14px;
    width: 18px;
    text-align: center;
  }
  .nav-label { flex: 1; }
  .badge {
    background: #ff2244;
    color: white;
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 8px;
    box-shadow: 0 0 6px #ff2244;
  }

  .sidebar-footer {
    display: flex;
    justify-content: space-around;
    padding: 16px 12px;
    border-top: 1px solid rgba(0,229,255,0.08);
    margin-top: auto;
  }
  .footer-stat {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }
  .fstat-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #00e5ff;
  }
  .fstat-key {
    font-family: 'Share Tech Mono', monospace;
    font-size: 7px;
    color: rgba(255,255,255,0.35);
    letter-spacing: 1px;
  }
</style>
