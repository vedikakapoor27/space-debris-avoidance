<script>
  import { onMount, onDestroy } from 'svelte'
  import * as THREE from 'three'
  import { generateDebrisField } from '../utils/api.js'
  import { selectedObject, globeRotating } from '../stores/appStore.js'

  let canvas
  let animId
  let renderer, scene, camera
  let earth, earthGlow, debrisMeshes = [], orbitRings = []
  let raycaster, mouse
  let debris = generateDebrisField(100)

  // Risk color palette
  const riskColor = (r) => {
    if (r > 0.7) return 0xff3860
    if (r > 0.3) return 0xff9020
    return 0x00e8a0
  }

  onMount(() => {
    // ── Renderer ────────────────────────────────────────────────────────────
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
    renderer.setSize(canvas.clientWidth, canvas.clientHeight)
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.toneMapping = THREE.ACESFilmicToneMapping

    // ── Scene ───────────────────────────────────────────────────────────────
    scene = new THREE.Scene()

    // ── Camera ──────────────────────────────────────────────────────────────
    camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.1, 100)
    camera.position.set(0, 0, 4.5)

    // ── Stars ───────────────────────────────────────────────────────────────
    const starGeo = new THREE.BufferGeometry()
    const starCount = 3000
    const starPos = new Float32Array(starCount * 3)
    for (let i = 0; i < starCount * 3; i++) starPos[i] = (Math.random() - 0.5) * 80
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3))
    const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.05, transparent: true, opacity: 0.8 })
    scene.add(new THREE.Points(starGeo, starMat))

    // ── Earth ───────────────────────────────────────────────────────────────
    const earthGeo = new THREE.SphereGeometry(1, 64, 64)
    const earthMat = new THREE.MeshPhongMaterial({
      color: 0x1a0a3d,
      emissive: 0x0a0520,
      specular: 0x8040ff,
      shininess: 25,
      wireframe: false,
    })
    earth = new THREE.Mesh(earthGeo, earthMat)
    scene.add(earth)

    // Grid overlay (lat/lon lines)
    const wireGeo = new THREE.SphereGeometry(1.002, 24, 24)
    const wireMat = new THREE.MeshBasicMaterial({ color: 0x5020a0, wireframe: true, transparent: true, opacity: 0.15 })
    earth.add(new THREE.Mesh(wireGeo, wireMat))

    // ── Atmosphere Glow ──────────────────────────────────────────────────────
    const glowGeo = new THREE.SphereGeometry(1.12, 32, 32)
    const glowMat = new THREE.MeshBasicMaterial({
      color:  0x5500cc,
      transparent: true,
      opacity: 0.07,
      side: THREE.FrontSide
    })
    earthGlow = new THREE.Mesh(glowGeo, glowMat)
    scene.add(earthGlow)

    const atmoGeo = new THREE.SphereGeometry(1.08, 32, 32)
    const atmoMat = new THREE.MeshBasicMaterial({ color: 0x3300aa, transparent: true, opacity: 0.12, side: THREE.BackSide })
    scene.add(new THREE.Mesh(atmoGeo, atmoMat))

    // ── Orbit Rings ──────────────────────────────────────────────────────────
    const ringRadii = [1.45, 1.8, 2.2]
    const ringColors = [0xe8b84b, 0xb060ff, 0x7c3fff]
    ringRadii.forEach((r, i) => {
      const geo = new THREE.TorusGeometry(r, 0.003, 8, 120)
      const mat = new THREE.MeshBasicMaterial({ color: ringColors[i], transparent: true, opacity: 0.25 })
      const ring = new THREE.Mesh(geo, mat)
      ring.rotation.x = Math.PI / 2 + (Math.random() - 0.5) * 0.6
      ring.rotation.z = (Math.random() - 0.5) * 0.5
      scene.add(ring)
      orbitRings.push(ring)
    })

    // ── Debris Field ─────────────────────────────────────────────────────────
    debris.forEach(d => {
      const geo = new THREE.SphereGeometry(d.size, 6, 6)
      const mat = new THREE.MeshBasicMaterial({ color: riskColor(d.risk) })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(d.x, d.y, d.z)
      mesh.userData = d
      scene.add(mesh)
      debrisMeshes.push(mesh)

      // Trail
      if (d.risk > 0.7) {
        const trailGeo = new THREE.SphereGeometry(d.size * 1.8, 6, 6)
        const trailMat = new THREE.MeshBasicMaterial({ color: 0xff2244, transparent: true, opacity: 0.2 })
        mesh.add(new THREE.Mesh(trailGeo, trailMat))
      }
    })

    // ── Lighting ─────────────────────────────────────────────────────────────
    scene.add(new THREE.AmbientLight(0x112244, 2))
    const sun = new THREE.DirectionalLight(0x4488ff, 3)
    sun.position.set(5, 3, 5)
    scene.add(sun)
    const rim = new THREE.DirectionalLight(0x0033ff, 1)
    rim.position.set(-3, -2, -3)
    scene.add(rim)

    // ── Raycaster for click ───────────────────────────────────────────────────
    raycaster = new THREE.Raycaster()
    mouse = new THREE.Vector2()

    // ── Mouse drag rotation ───────────────────────────────────────────────────
    let isDragging = false, prevX = 0, prevY = 0
    canvas.addEventListener('mousedown', e => { isDragging = true; prevX = e.clientX; prevY = e.clientY })
    window.addEventListener('mouseup', () => { isDragging = false })
    canvas.addEventListener('mousemove', e => {
      if (!isDragging) return
      const dx = e.clientX - prevX
      const dy = e.clientY - prevY
      earth.rotation.y += dx * 0.005
      earth.rotation.x += dy * 0.003
      prevX = e.clientX; prevY = e.clientY
    })

    canvas.addEventListener('click', e => {
      const rect = canvas.getBoundingClientRect()
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObjects(debrisMeshes)
      if (hits.length > 0) selectedObject.set(hits[0].object.userData)
    })

    // ── Resize ────────────────────────────────────────────────────────────────
    const ro = new ResizeObserver(() => {
      camera.aspect = canvas.clientWidth / canvas.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(canvas.clientWidth, canvas.clientHeight)
    })
    ro.observe(canvas)

    // ── Animate ───────────────────────────────────────────────────────────────
    let t = 0
    const tick = () => {
      animId = requestAnimationFrame(tick)
      t += 0.005

      let rotating = true
      globeRotating.subscribe(v => rotating = v)()

      if (rotating) {
        earth.rotation.y += 0.0015
        earthGlow.rotation.y += 0.0008
      }

      // Orbit debris
      debrisMeshes.forEach((mesh, i) => {
        const d = debris[i]
        const angle = t * d.speed * 30
        mesh.position.x = d.x * Math.cos(angle) - d.z * Math.sin(angle)
        mesh.position.z = d.x * Math.sin(angle) + d.z * Math.cos(angle)
        // Pulse high-risk
        if (d.risk > 0.7) {
          const s = 1 + 0.3 * Math.sin(t * 10 + i)
          mesh.scale.setScalar(s)
        }
      })

      orbitRings.forEach((r, i) => {
        r.rotation.z += 0.0003 * (i + 1)
      })

      renderer.render(scene, camera)
    }
    tick()
  })

  onDestroy(() => {
    cancelAnimationFrame(animId)
    renderer?.dispose()
  })
</script>

<div class="globe-wrap">
  <canvas bind:this={canvas}></canvas>
  <div class="globe-overlay">
    <span class="globe-label">LIVE ORBITAL TRACKING</span>
    <span class="globe-sub">Click debris to inspect</span>
  </div>
</div>

<style>
  .globe-wrap {
    position: relative;
    width: 100%;
    height: 100%;
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
    cursor: grab;
    border-radius: 8px;
  }
  canvas:active { cursor: grabbing; }
  .globe-overlay {
    position: absolute;
    bottom: 16px;
    left: 16px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    pointer-events: none;
  }
  .globe-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 9px;
    letter-spacing: 3px;
    color: rgba(0, 229, 255, 0.6);
  }
  .globe-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.3);
  }
</style>
