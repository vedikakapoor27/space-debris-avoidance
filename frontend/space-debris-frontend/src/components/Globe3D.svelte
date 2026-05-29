<script>
  import { onMount, onDestroy } from 'svelte'
  import * as THREE from 'three'
  import { generateDebrisField } from '../utils/api.js'
  import { selectedObject, globeRotating, theme } from '../stores/appStore.js'

  let canvas
  let animId
  let renderer, scene, camera
  let earth, earthGlow, atmosphere
  let debrisMeshes = [], orbitRings = []
  let raycaster, mouse
  let debris = generateDebrisField(100)
  let currentTheme = 'dark'

  theme.subscribe(t => { currentTheme = t; updateColors() })

  const riskColor = (r) => {
    if (r > 0.7) return 0xff3860
    if (r > 0.3) return 0xff9020
    return 0x00e8a0
  }

  function updateColors() {
    if (!earth) return
    if (currentTheme === 'dark') {
      earth.material.color.setHex(0x1a0a3d)
      earth.material.emissive.setHex(0x0a0520)
      earth.material.specular.setHex(0x8040ff)
      if (earthGlow) earthGlow.material.color.setHex(0x5500cc)
      if (atmosphere) atmosphere.material.color.setHex(0x3300aa)
      orbitRings.forEach((r, i) => {
        r.material.color.setHex([0xf0c040, 0xc084fc, 0x7c3aed][i])
      })
    } else {
      earth.material.color.setHex(0x0a2040)
      earth.material.emissive.setHex(0x050f20)
      earth.material.specular.setHex(0x2060ff)
      if (earthGlow) earthGlow.material.color.setHex(0x1040aa)
      if (atmosphere) atmosphere.material.color.setHex(0x0a2060)
      orbitRings.forEach((r, i) => {
        r.material.color.setHex([0xf59e0b, 0x60a5fa, 0x2563eb][i])
      })
    }
  }

  onMount(() => {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
    renderer.setSize(canvas.clientWidth, canvas.clientHeight)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.2

    scene = new THREE.Scene()
    camera = new THREE.PerspectiveCamera(42, canvas.clientWidth / canvas.clientHeight, 0.1, 100)
    camera.position.set(0, 0.5, 4.8)

    // Stars — warm-tinted
    const starGeo = new THREE.BufferGeometry()
    const N = 3000
    const starPos = new Float32Array(N * 3)
    const starCol = new Float32Array(N * 3)
    for (let i = 0; i < N; i++) {
      starPos[i*3]   = (Math.random() - 0.5) * 80
      starPos[i*3+1] = (Math.random() - 0.5) * 80
      starPos[i*3+2] = (Math.random() - 0.5) * 80
      const warm = Math.random() > 0.7
      starCol[i*3]   = warm ? 1.0 : 0.85
      starCol[i*3+1] = warm ? 0.92 : 0.88
      starCol[i*3+2] = warm ? 0.80 : 1.0
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3))
    starGeo.setAttribute('color', new THREE.BufferAttribute(starCol, 3))
    const starMat = new THREE.PointsMaterial({ size: 0.035, vertexColors: true, transparent: true, opacity: 0.85 })
    scene.add(new THREE.Points(starGeo, starMat))

    // Earth
    const earthGeo = new THREE.SphereGeometry(1, 64, 64)
    const earthMat = new THREE.MeshPhongMaterial({
      color: 0x1a0a3d, emissive: 0x0a0520, specular: 0x8040ff, shininess: 40
    })
    earth = new THREE.Mesh(earthGeo, earthMat)
    scene.add(earth)

    // Wireframe grid
    const wireGeo = new THREE.SphereGeometry(1.003, 28, 28)
    const wireMat = new THREE.MeshBasicMaterial({ color: 0x5020a0, wireframe: true, transparent: true, opacity: 0.1 })
    earth.add(new THREE.Mesh(wireGeo, wireMat))

    // Atmosphere layers
    earthGlow = new THREE.Mesh(
      new THREE.SphereGeometry(1.14, 32, 32),
      new THREE.MeshBasicMaterial({ color: 0x5500cc, transparent: true, opacity: 0.07, side: THREE.FrontSide })
    )
    scene.add(earthGlow)

    atmosphere = new THREE.Mesh(
      new THREE.SphereGeometry(1.09, 32, 32),
      new THREE.MeshBasicMaterial({ color: 0x3300aa, transparent: true, opacity: 0.12, side: THREE.BackSide })
    )
    scene.add(atmosphere)

    // Orbit rings
    const ringDefs = [
      { r: 1.42, tilt: 0.18, color: 0xf0c040, opacity: 0.22 },
      { r: 1.78, tilt: 0.52, color: 0xc084fc, opacity: 0.18 },
      { r: 2.18, tilt: 0.72, color: 0x7c3aed, opacity: 0.15 },
    ]
    ringDefs.forEach(({ r, tilt, color, opacity }) => {
      const geo = new THREE.TorusGeometry(r, 0.003, 8, 140)
      const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity })
      const ring = new THREE.Mesh(geo, mat)
      ring.rotation.x = Math.PI / 2 + tilt
      ring.rotation.z = (Math.random() - 0.5) * 0.4
      scene.add(ring)
      orbitRings.push(ring)
    })

    // Debris objects
    debris.forEach(d => {
      const geo = new THREE.SphereGeometry(d.size, 6, 6)
      const mat = new THREE.MeshBasicMaterial({ color: riskColor(d.risk) })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(d.x, d.y, d.z)
      mesh.userData = d
      scene.add(mesh)
      debrisMeshes.push(mesh)

      if (d.risk > 0.7) {
        const halo = new THREE.Mesh(
          new THREE.SphereGeometry(d.size * 2.5, 6, 6),
          new THREE.MeshBasicMaterial({ color: 0xff3860, transparent: true, opacity: 0.15 })
        )
        mesh.add(halo)
      }
    })

    // Lighting
    scene.add(new THREE.AmbientLight(0x200840, 2.2))
    const sun = new THREE.DirectionalLight(0x9060ff, 3.5)
    sun.position.set(5, 3, 5); scene.add(sun)
    const rim = new THREE.DirectionalLight(0xf0c040, 0.6)
    rim.position.set(-4, -1, -3); scene.add(rim)

    // Mouse interactions
    raycaster = new THREE.Raycaster()
    mouse = new THREE.Vector2()

    let isDragging = false, prevX = 0, prevY = 0
    canvas.addEventListener('mousedown', e => { isDragging = true; prevX = e.clientX; prevY = e.clientY })
    window.addEventListener('mouseup', () => { isDragging = false })
    canvas.addEventListener('mousemove', e => {
      if (!isDragging) return
      earth.rotation.y += (e.clientX - prevX) * 0.005
      earth.rotation.x += (e.clientY - prevY) * 0.003
      prevX = e.clientX; prevY = e.clientY
    })
    canvas.addEventListener('click', e => {
      const rect = canvas.getBoundingClientRect()
      mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1
      mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1
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
        earth.rotation.y      += 0.0014
        earthGlow.rotation.y  += 0.0007
        atmosphere.rotation.y -= 0.0005
      }
      debrisMeshes.forEach((mesh, i) => {
        const d = debris[i]
        const a = t * d.speed * 30
        mesh.position.x = d.x * Math.cos(a) - d.z * Math.sin(a)
        mesh.position.z = d.x * Math.sin(a) + d.z * Math.cos(a)
        if (d.risk > 0.7) mesh.scale.setScalar(1 + 0.25 * Math.sin(t * 10 + i))
      })
      orbitRings.forEach((r, i) => { r.rotation.z += 0.0003 * (i + 1) })
      renderer.render(scene, camera)
    }
    tick()
  })

  onDestroy(() => { cancelAnimationFrame(animId); renderer?.dispose() })
</script>

<div class="wrap">
  <canvas bind:this={canvas}></canvas>
  <div class="label">
    <span class="l1">Live Orbital Tracking</span>
    <span class="l2">Drag to rotate · Click debris to inspect</span>
  </div>
</div>

<style>
  .wrap { position: relative; width: 100%; height: 100%; }
  canvas { width: 100%; height: 100%; display: block; cursor: grab; }
  canvas:active { cursor: grabbing; }
  .label {
    position: absolute; bottom: 16px; left: 16px;
    display: flex; flex-direction: column; gap: 2px;
    pointer-events: none;
  }
  .l1 {
    font-family: 'Syne', sans-serif; font-size: 10px; font-weight: 700;
    letter-spacing: 0.18em; color: rgba(240,192,64,0.45); text-transform: uppercase;
  }
  .l2 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: rgba(196,181,253,0.28);
  }
</style>
