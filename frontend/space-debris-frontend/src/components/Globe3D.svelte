<script>
  import { onMount, onDestroy } from 'svelte'
  import * as THREE from 'three'
  import { generateDebrisField } from '../utils/api.js'
  import { selectedObject, globeRotating } from '../stores/appStore.js'

  let canvas
  let animId
  let renderer, scene, camera, earth, clouds
  let debrisMeshes = [], orbitRings = []
  let raycaster, mouse
  let debris = generateDebrisField(120)
  let tooltip = null
  let tooltipX = 0, tooltipY = 0

  const riskColor = (r) => {
    if (r > 0.7) return 0xef4444
    if (r > 0.3) return 0xf59e0b
    return 0x3b82f6
  }

  onMount(() => {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
    renderer.setSize(canvas.clientWidth, canvas.clientHeight)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.shadowMap.enabled = true

    scene = new THREE.Scene()
    camera = new THREE.PerspectiveCamera(40, canvas.clientWidth / canvas.clientHeight, 0.1, 100)
    camera.position.set(0, 0, 4.2)

    // Minimal stars — very faint
    const starGeo = new THREE.BufferGeometry()
    const N = 1500
    const starPos = new Float32Array(N * 3)
    for (let i = 0; i < N * 3; i++) starPos[i] = (Math.random() - 0.5) * 80
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3))
    const starMat = new THREE.PointsMaterial({ color: 0x94a3b8, size: 0.025, transparent: true, opacity: 0.4 })
    scene.add(new THREE.Points(starGeo, starMat))

    // Earth with real texture
    const loader = new THREE.TextureLoader()
    const earthGeo = new THREE.SphereGeometry(1, 64, 64)

    // Use a reliable earth texture
    const earthTex = loader.load(
      'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_atmos_2048.jpg',
      undefined, undefined,
      () => {
        // fallback if texture fails — use procedural
        earthMesh.material.color.setHex(0x1a4a6b)
      }
    )
    const earthBump = loader.load(
      'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_normal_2048.jpg'
    )
    const earthSpec = loader.load(
      'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_specular_2048.jpg'
    )

    const earthMat = new THREE.MeshPhongMaterial({
      map: earthTex,
      bumpMap: earthBump,
      bumpScale: 0.015,
      specularMap: earthSpec,
      specular: new THREE.Color(0x222222),
      shininess: 8,
    })
    earth = new THREE.Mesh(earthGeo, earthMat)
    scene.add(earth)

    // Thin atmosphere
    const atmoGeo = new THREE.SphereGeometry(1.02, 32, 32)
    const atmoMat = new THREE.MeshBasicMaterial({
      color: 0x4488ff, transparent: true, opacity: 0.06, side: THREE.FrontSide
    })
    scene.add(new THREE.Mesh(atmoGeo, atmoMat))

    // Orbit rings — thin and clean
    const ringDefs = [
      { r: 1.4, tilt: 0.2,  color: 0x3b82f6, opacity: 0.15 },
      { r: 1.7, tilt: 0.5,  color: 0x64748b, opacity: 0.12 },
      { r: 2.1, tilt: 0.75, color: 0x3b82f6, opacity: 0.1  },
    ]
    ringDefs.forEach(({ r, tilt, color, opacity }) => {
      const geo = new THREE.TorusGeometry(r, 0.002, 8, 140)
      const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity })
      const ring = new THREE.Mesh(geo, mat)
      ring.rotation.x = Math.PI / 2 + tilt
      ring.rotation.z = (Math.random() - 0.5) * 0.3
      scene.add(ring)
      orbitRings.push(ring)
    })

    // Debris — clean dots, no glow
    debris.forEach(d => {
      const size = d.risk > 0.7 ? 0.012 : 0.007
      const geo = new THREE.SphereGeometry(size, 5, 5)
      const mat = new THREE.MeshBasicMaterial({ color: riskColor(d.risk) })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(d.x, d.y, d.z)
      mesh.userData = d
      scene.add(mesh)
      debrisMeshes.push(mesh)
    })

    // Lighting — realistic sun
    scene.add(new THREE.AmbientLight(0x404060, 1.2))
    const sun = new THREE.DirectionalLight(0xffffff, 2.8)
    sun.position.set(8, 3, 4)
    scene.add(sun)

    // Mouse
    raycaster = new THREE.Raycaster()
    mouse = new THREE.Vector2()

    let isDragging = false, prevX = 0, prevY = 0
    let velX = 0, velY = 0

    canvas.addEventListener('mousedown', e => {
      isDragging = true; prevX = e.clientX; prevY = e.clientY; velX = 0; velY = 0
    })
    window.addEventListener('mouseup', () => { isDragging = false })
    canvas.addEventListener('mousemove', e => {
      const rect = canvas.getBoundingClientRect()
      mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1
      mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1
      tooltipX = e.clientX - rect.left
      tooltipY = e.clientY - rect.top

      if (isDragging) {
        velX = (e.clientX - prevX) * 0.005
        velY = (e.clientY - prevY) * 0.003
        earth.rotation.y += velX
        earth.rotation.x += velY
        prevX = e.clientX; prevY = e.clientY
      }

      // hover detection
      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObjects(debrisMeshes)
      tooltip = hits.length ? hits[0].object.userData : null
    })

    // Scroll to zoom
    canvas.addEventListener('wheel', e => {
      camera.position.z = Math.max(2.5, Math.min(7, camera.position.z + e.deltaY * 0.005))
    })

    canvas.addEventListener('click', e => {
      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObjects(debrisMeshes)
      if (hits.length) selectedObject.set(hits[0].object.userData)
    })

    const ro = new ResizeObserver(() => {
      camera.aspect = canvas.clientWidth / canvas.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(canvas.clientWidth, canvas.clientHeight)
    })
    ro.observe(canvas)

    let t = 0
    const tick = () => {
      animId = requestAnimationFrame(tick)
      t += 0.005

      let rotating = true
      globeRotating.subscribe(v => rotating = v)()

      if (rotating) {
        earth.rotation.y += 0.001
        // inertia
        if (!isDragging) {
          velX *= 0.95; velY *= 0.95
          earth.rotation.y += velX
          earth.rotation.x += velY
        }
      }

      debrisMeshes.forEach((mesh, i) => {
        const d = debris[i]
        const a = t * d.speed * 25
        mesh.position.x = d.x * Math.cos(a) - d.z * Math.sin(a)
        mesh.position.z = d.x * Math.sin(a) + d.z * Math.cos(a)
      })

      orbitRings.forEach((r, i) => { r.rotation.z += 0.0002 * (i + 1) })
      renderer.render(scene, camera)
    }
    tick()
  })

  onDestroy(() => { cancelAnimationFrame(animId); renderer?.dispose() })
</script>

<div class="wrap">
  <canvas bind:this={canvas}></canvas>

  {#if tooltip}
    <div class="tooltip" style="left:{tooltipX + 14}px; top:{tooltipY - 10}px">
      <div class="tt-name">{tooltip.label}</div>
      <div class="tt-row"><span>Risk</span><span class="tt-risk" style="color:{tooltip.risk > 0.7 ? '#ef4444' : tooltip.risk > 0.3 ? '#f59e0b' : '#3b82f6'}">{tooltip.risk > 0.7 ? 'CRITICAL' : tooltip.risk > 0.3 ? 'WARNING' : 'NOMINAL'}</span></div>
      <div class="tt-row"><span>Speed</span><span>{(tooltip.speed * 1000).toFixed(1)} km/s</span></div>
    </div>
  {/if}
</div>

<style>
  .wrap { position: relative; width: 100%; height: 100%; }
  canvas { width: 100%; height: 100%; display: block; cursor: none; }

  .tooltip {
    position: absolute; pointer-events: none;
    background: rgba(15,23,42,0.95);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 10px 12px;
    backdrop-filter: blur(10px);
    min-width: 140px; z-index: 100;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  }

  .tt-name {
    font-size: 12px; font-weight: 600; color: #f8fafc;
    margin-bottom: 7px; letter-spacing: 0.02em;
  }

  .tt-row {
    display: flex; justify-content: space-between;
    font-size: 11px; color: #94a3b8; margin-top: 4px;
  }

  .tt-risk { font-weight: 600; }
</style>