<script>
  import { onMount, onDestroy } from 'svelte'
  import { fly, scale } from 'svelte/transition'
  import { cubicOut } from 'svelte/easing'
  import { getMockConjunctions } from '../utils/api.js'
  import { selectedObject } from '../stores/appStore.js'

  let conjunctions = getMockConjunctions()
  let interval
  let stats = { tracked: 847, active_sats: 12, high_risk: 2, avg_distance: 156.4 }
  let mounted = false

  onMount(() => {
    mounted = true
    interval = setInterval(() => {
      stats.avg_distance = +(152 + Math.random() * 8).toFixed(1)
    }, 2500)
  })
  onDestroy(() => clearInterval(interval))

  const riskCls = r => r === 'HIGH' ? 'high' : r === 'MEDIUM' ? 'med' : 'low'

 const statCards = [
  { key: 'tracked',      label: 'Tracked Objects',  color: 'var(--blue)',    pct: '85%' },
  { key: 'active_sats',  label: 'Active Satellites', color: 'var(--success)', pct: '40%' },
  { key: 'high_risk',    label: 'Critical Alerts',   color: 'var(--danger)',  pct: '20%' },
  { key: 'avg_distance', label: 'Avg Miss Dist',      color: 'var(--warning)', pct: '55%' },
]
  
</script>

<div class="dp">

  <div class="dp-header">
    <div class="dp-eyebrow">Mission Overview</div>
    <h2 class="dp-title">Orbital Collision Monitoring</h2>
    <div class="dp-live">
      <span class="live-dot"></span>LIVE
    </div>
  </div>

  {#if mounted}
    <div class="stat-grid">
      {#each statCards as card, i}
        <div
          class="stat-card"
          in:fly={{ y: 16, duration: 300, delay: i * 70, easing: cubicOut }}
          style="--ac:{card.color}; --pct:{card.pct}"
        >
          <div class="stat-ico" style="color:{card.color}">◈</div>
          <div class="stat-val">
            {#if card.key === 'avg_distance'}
              {stats[card.key]}<span class="stat-unit"> km</span>
            {:else}
              {stats[card.key]}
            {/if}
          </div>
          <div class="stat-lbl">{card.label}</div>
          <div class="stat-bar"></div>
        </div>
      {/each}
    </div>
  {/if}

  <div class="section">
    <div class="section-head">
      <div class="section-title-wrap">
        <span class="section-flash">⚡</span>
        <span class="section-title">Active Conjunctions</span>
      </div>
      <span class="section-badge">{conjunctions.length} events</span>
    </div>
    <div class="table-wrap">
      <table class="ctable">
        <thead>
          <tr>
            {#each ['ID','Object 1','Object 2','Dist (km)','Vel (km/s)','TCA','Risk'] as h}
              <th>{h}</th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each conjunctions as c, i}
            <tr
              class="crow"
              class:crow-high={c.risk === 'HIGH'}
              in:fly={{ x: -10, duration: 250, delay: i * 50, easing: cubicOut }}
            >
              <td class="mono dim">{c.id}</td>
              <td class="mono">{c.object1}</td>
              <td class="mono">{c.object2}</td>
              <td class="mono">{c.distance}</td>
              <td class="mono">{c.velocity}</td>
              <td class="mono dim">{c.time}</td>
              <td><span class="pill {riskCls(c.risk)}">{c.risk}</span></td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>

  {#if $selectedObject}
    <div class="section" in:fly={{ y: 10, duration: 250, easing: cubicOut }}>
      <div class="section-head">
        <span class="section-title">Selected Object</span>
        <button class="clear-btn" on:click={() => selectedObject.set(null)}>✕ Clear</button>
      </div>
      <div class="obj-grid">
        {#each [
          ['Label',    $selectedObject.label],
          ['X',        $selectedObject.x.toFixed(4)],
          ['Y',        $selectedObject.y.toFixed(4)],
          ['Z',        $selectedObject.z.toFixed(4)],
          ['Risk',     ($selectedObject.risk*100).toFixed(1)+'%']
        ] as [k,v]}
          <div class="obj-item">
            <span class="obj-key">{k}</span>
            <span class="obj-val">{v}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

</div>

<style>
  .dp {
    padding: 24px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 20px;
    height: 100%; background: var(--bg);
    transition: background 0.3s;
  }

  .dp-header {
    display: flex; align-items: flex-start;
    justify-content: space-between;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--divider);
  }

  .dp-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; color: var(--text-4);
    letter-spacing: 0.2em; text-transform: uppercase;
    margin-bottom: 8px;
  }

  .dp-title {
  font-family: 'Inter', sans-serif;
  font-size: 20px; font-weight: 800;
  color: var(--text); line-height: 1.1;
  letter-spacing: -0.02em; text-transform: uppercase;
  white-space: nowrap;
}

  .dp-live {
    display: flex; align-items: center; gap: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; color: var(--text-3);
    letter-spacing: 0.16em; text-transform: uppercase;
    border: 1px solid var(--border);
    padding: 5px 10px; flex-shrink: 0; margin-top: 4px;
  }

  .live-dot {
    width: 4px; height: 4px; border-radius: 50%;
    background: var(--text);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.2} }

  /* STAT GRID */
  .stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1px; background: var(--border); }

  .stat-card {
    background: var(--card);
    padding: 16px 14px 14px;
    position: relative; overflow: hidden;
    transition: background 0.15s;
  }
  .stat-card:hover { background: var(--card2); }

  .stat-ico {
    font-size: 11px; margin-bottom: 10px;
    display: block; color: var(--text-4); opacity: 0.8;
  }

  .stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px; font-weight: 700;
    color: var(--text);
    line-height: 1; margin-bottom: 5px;
  }
  .stat-unit { font-size: 12px; font-weight: 400; }

  .stat-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; color: var(--text-4);
    letter-spacing: 0.14em; text-transform: uppercase;
    margin-bottom: 12px;
  }

  .stat-bar {
    position: absolute; bottom: 0; left: 0;
    width: var(--pct); height: 1px;
    background: var(--text-3);
  }

  /* SECTION */
  .section {
    border: 1px solid var(--border); overflow: hidden;
  }

  .section-head {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    background: var(--card2);
  }

  .section-title-wrap { display: flex; align-items: center; gap: 8px; }
  .section-flash { font-size: 10px; color: var(--text-3); }

  .section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 600;
    color: var(--text-2); letter-spacing: 0.16em;
    text-transform: uppercase;
  }

  .section-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; color: var(--text-4);
    letter-spacing: 0.08em;
  }

  /* TABLE */
  .table-wrap { overflow-x: auto; }

  .ctable { width: 100%; border-collapse: collapse; }

  .ctable th {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; letter-spacing: 0.14em; color: var(--text-4);
    padding: 8px 12px; text-align: left;
    border-bottom: 1px solid var(--border);
    text-transform: uppercase; background: var(--card2);
    white-space: nowrap;
  }

  .crow td {
    padding: 9px 12px; border-bottom: 1px solid var(--divider);
    transition: background 0.1s;
  }
  .crow:last-child td { border-bottom: none; }
  .crow:hover td { background: var(--surface); }
  .crow-high td { background: var(--danger-bg); }

  .mono {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: var(--text-2);
  }
  .dim { color: var(--text-4) !important; }

  /* PILLS — monochrome only */
  .pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; font-weight: 600;
    letter-spacing: 0.1em; padding: 2px 8px;
    text-transform: uppercase; border: 1px solid;
  }
  .pill.high { color: var(--text);   border-color: var(--border2); background: var(--danger-bg); }
  .pill.med  { color: var(--text-2); border-color: var(--border);  background: var(--warning-bg); }
  .pill.low  { color: var(--text-3); border-color: var(--divider); background: var(--success-bg); }

  /* OBJECT DETAIL */
  .obj-grid { display: grid; grid-template-columns: repeat(5,1fr); }
  .obj-item {
    display: flex; flex-direction: column; align-items: center; gap: 5px;
    padding: 12px 8px;
    border-right: 1px solid var(--divider);
  }
  .obj-item:last-child { border-right: none; }
  .obj-key {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 7px; color: var(--text-4);
    letter-spacing: 0.14em; text-transform: uppercase;
  }
  .obj-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; font-weight: 500; color: var(--text);
  }

  .clear-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; color: var(--text-4); background: none;
    border: 1px solid var(--border); padding: 3px 8px;
    cursor: none; transition: all 0.2s; letter-spacing: 0.08em;
  }
  .clear-btn:hover { color: var(--text); border-color: var(--border2); }
</style>