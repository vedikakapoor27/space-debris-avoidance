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
    { key: 'tracked',       label: 'Tracked Objects', color: 'var(--accent-hi)', pct: '85%' },
    { key: 'active_sats',   label: 'Active Satellites', color: 'var(--safe)', pct: '40%' },
    { key: 'high_risk',     label: 'Critical Alerts', color: 'var(--danger)', pct: '20%' },
    { key: 'avg_distance',  label: 'Avg Miss Dist', color: 'var(--gold)', pct: '55%' },
  ]
</script>

<div class="dp">

  <div class="dp-header">
    <div class="dp-eyebrow">Mission Overview</div>
    <h2 class="dp-title">Orbital Collision<br>Monitoring</h2>
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
    padding: 28px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 24px;
    height: 100%;
  }

  .dp-header {
    display: flex; align-items: flex-start;
    justify-content: space-between; gap: 16px;
  }

  .dp-eyebrow {
  font-size: 10px; font-weight: 500;
  color: var(--text-3);
  letter-spacing: 0.12em; text-transform: uppercase;
  margin-bottom: 8px;
}
  .dp-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 24px; font-weight: 700;
  color: var(--text);
  line-height: 1.2;
}

  .dp-live {
  display: flex; align-items: center; gap: 7px;
  font-size: 11px; font-weight: 600; color: var(--success);
  letter-spacing: 0.1em;
  border: 1px solid rgba(16,185,129,0.25);
  padding: 5px 12px; border-radius: 20px;
  flex-shrink: 0; margin-top: 4px;
}

.live-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--success);
  animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

  .stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }
.stat-card {
  background: var(--card);
  border: 1px solid var(--border2);
  padding: 18px 16px 14px;
  border-radius: var(--radius);
  position: relative; overflow: hidden;
  box-shadow: var(--shadow);
  transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}
  .stat-card:hover { border-color: var(--ac); transform: translateY(-1px); }

  .stat-ico {
  font-size: 13px; margin-bottom: 10px;
  display: block; color: var(--text-3);
}

  .stat-val {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 30px; font-weight: 700;
  color: var(--text);
  line-height: 1; margin-bottom: 6px;
}
 .stat-unit { font-size: 14px; font-weight: 400; }

  .stat-lbl {
  font-size: 11px; font-weight: 500;
  color: var(--text-3);
  letter-spacing: 0.08em; text-transform: uppercase;
  margin-bottom: 14px;
}
  .stat-bar {
  position: absolute; bottom: 0; left: 0;
  width: var(--pct); height: 2px;
  background: var(--ac);
}

  .section {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  background: var(--card);
  box-shadow: var(--shadow);
}

.section-head {
  display: flex; justify-content: space-between; align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--card2);
}


  .section-title-wrap { display: flex; align-items: center; gap: 8px; }
 .section-flash { font-size: 12px; color: var(--warning); }

  .section-title {
  font-size: 12px; font-weight: 600;
  color: var(--text-2);
  letter-spacing: 0.06em; text-transform: uppercase;
}

  .section-badge {
  font-size: 11px; color: var(--text-4);
}


  .table-wrap { overflow-x: auto; }

  .ctable { width: 100%; border-collapse: collapse; }

  .ctable th {
  font-size: 10px; font-weight: 600;
  letter-spacing: 0.1em; color: var(--text-3);
  padding: 10px 14px; text-align: left;
  border-bottom: 1px solid var(--border);
  text-transform: uppercase;
  background: var(--card2);
}

.crow td {
  padding: 11px 14px;
  border-bottom: 1px solid var(--border);
  color: var(--text-2);
  font-size: 13px;
  transition: background 0.15s;
}
  .crow:hover td { background: var(--surface); }
.crow:last-child td { border-bottom: none; }
.crow-high td { background: rgba(239,68,68,0.04); }

  .mono { font-family: 'Space Grotesk', sans-serif; font-size: 13px; color: var(--text-2); }
.dim  { color: var(--text-4) !important; }
  .pill {
  font-size: 10px; font-weight: 600;
  letter-spacing: 0.06em; padding: 3px 10px;
  border-radius: 4px; text-transform: uppercase;
}
 .pill.high { background: rgba(239,68,68,0.12); color: var(--danger);  border: 1px solid rgba(239,68,68,0.25); }
.pill.med  { background: rgba(245,158,11,0.12); color: var(--warning); border: 1px solid rgba(245,158,11,0.25); }
.pill.low  { background: rgba(16,185,129,0.1);  color: var(--success); border: 1px solid rgba(16,185,129,0.22); }


  .obj-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 0; }
  .obj-item {
    display: flex; flex-direction: column; align-items: center; gap: 5px;
    padding: 14px 10px;
    border-right: 1px solid var(--border-dim);
  }
  .obj-item:last-child { border-right: none; }
  .obj-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: var(--text-dim); letter-spacing: 0.14em; text-transform: uppercase;
  }
  .obj-val {
    font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700; color: var(--text);
  }

  .clear-btn {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text-dim); background: none;
    border: 1px solid var(--border-dim); padding: 4px 10px; cursor: pointer;
    transition: all 0.2s;
  }
  .clear-btn:hover { color: var(--danger); border-color: rgba(255,56,96,0.3); }
</style>
