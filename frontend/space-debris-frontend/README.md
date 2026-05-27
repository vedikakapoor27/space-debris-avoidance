# 🛸 ORION — Space Debris Avoidance Dashboard
### Built with **SvelteKit + Three.js** (Svelte v4 + Vite)

---

## 🧠 Why Svelte? (The Underrated Tech)

| Feature | React | Svelte |
|---|---|---|
| Runtime bundle | ~130kb | **~0kb** (compiled away) |
| Reactivity | useState hooks | **Native `$:` reactive declarations** |
| Stores | Redux/Zustand | **Built-in `writable()` stores** |
| Animations | Framer Motion lib | **Built-in `transition:` directives** |
| Learning curve | Medium | **Low — feels like HTML+JS** |
| Performance | Virtual DOM diffing | **Direct DOM updates, no VDOM** |

Svelte compiles your components to vanilla JS at build time — the browser runs zero framework code.

---

## 📂 Full Frontend Structure

```
space-debris-frontend/
│
├── index.html                        ← Entry HTML (loads fonts, mounts app)
├── vite.config.js                    ← Vite + Svelte plugin config
├── package.json                      ← Dependencies
│
└── src/
    ├── main.js                       ← Mounts App.svelte to #app
    │
    ├── App.svelte                    ← ROOT: Layout (Sidebar + Globe + Panel)
    │                                    Checks backend health on load
    │
    ├── components/
    │   ├── Sidebar.svelte            ← Left nav (Dashboard/Predict/Conjunctions/Telemetry)
    │   ├── Globe3D.svelte            ← THREE.js 3D Earth + debris field + orbit rings
    │   │                                Click a debris dot → shows in DashboardPanel
    │   ├── DashboardPanel.svelte     ← Overview: stat cards + conjunction table
    │   ├── PredictPanel.svelte       ← Sliders → POST /predict → shows risk + avoidance plan
    │   ├── ConjunctionsPanel.svelte  ← Event list → select → "Analyze in Predictor" button
    │   └── TelemetryPanel.svelte     ← Live feed, auto-polls /predict every 4s
    │
    ├── stores/
    │   └── appStore.js               ← Global state (Svelte stores):
    │                                    backendOnline, prediction, selectedObject,
    │                                    telemetryFeed, alertCount, activePanel
    │
    └── utils/
        └── api.js                    ← All fetch calls to Flask backend:
                                         checkHealth()    → GET  /health
                                         predict()        → POST /predict
                                         generateDebrisField()  → mock 3D debris
                                         getMockConjunctions()  → mock events
```

---

## 🚀 Running the Frontend

### Step 1 — Install
```bash
cd space-debris-frontend
npm install
```

### Step 2 — Start dev server
```bash
npm run dev
# Opens at http://localhost:5173
```

### Step 3 — Build for production
```bash
npm run build
# Output goes to /dist
```

---

## 🔌 Connecting Frontend ↔ Flask Backend

### Backend (your existing Flask app)

Your `app.py` already has CORS enabled:
```python
from flask_cors import CORS
CORS(app)  # ✅ already done
```

Start the Flask backend:
```bash
cd Backend/ml
pip install flask flask-cors joblib scikit-learn pandas
python app.py
# Runs on http://localhost:5000
```

### Frontend API calls (src/utils/api.js)

The frontend talks to Flask via:

```
GET  http://localhost:5000/health     ← Backend status check (topbar indicator)
POST http://localhost:5000/predict    ← Risk prediction + avoidance plan
```

The `predict()` call sends:
```json
{
  "distance_km": 50,
  "rel_velocity": 7,
  "approach_rate": -5
}
```

Flask responds with everything in `full_assessment()`:
```json
{
  "status": "success",
  "risk_level": "MEDIUM",
  "probability": 62.4,
  "color": "ORANGE",
  "message": "Monitor closely — situation developing",
  "action": "MONITOR CLOSELY",
  "maneuver_type": "Prepare orbit adjustment",
  "maneuver_km": "TBD",
  "fuel_cost_kg": "TBD",
  "time_window": "Review in 2 hours",
  "urgency": "WATCH"
}
```

---

## 🗺️ Data Flow Diagram

```
[3D Globe (Three.js)]
   ↓ click debris object
[appStore.selectedObject]
   ↓ shows in
[DashboardPanel]

[PredictPanel sliders]
   → POST /predict (Flask)
   ← risk + avoidance data
   → appStore.prediction
   → appStore.telemetryFeed (ring buffer, last 20)

[TelemetryPanel]
   ← reads telemetryFeed store
   ← auto-polls /predict every 4s

[ConjunctionsPanel]
   → "Analyze" button
   → sets formValues store
   → switches activePanel to 'predict'
   (PredictPanel pre-fills with conjunction data)
```

---

## 🎨 Tech Stack (Frontend)

| Tech | Purpose | Why |
|---|---|---|
| **Svelte 4** | UI framework | Zero-runtime, reactive, fast |
| **Vite 5** | Dev server + bundler | Near-instant HMR |
| **Three.js** | 3D Earth + debris field | WebGL 3D rendering |
| **Svelte Stores** | Global state | Built-in, no Redux needed |
| **CSS Custom Props** | Theming | Space dark palette |
| **Orbitron font** | HUD typography | NASA/military aesthetic |
| **Share Tech Mono** | Data readouts | Terminal/telemetry feel |

---

## 🔮 Future Backend Endpoints to Add

When your backend grows, update `src/utils/api.js`:

```js
// Add these to api.js as your backend expands:

// GET /conjunctions  → real conjunction data from conjunction_summary.json
export async function getConjunctions() {
  const res = await fetch(`${BASE_URL}/conjunctions`)
  return res.json()
}

// GET /satellites  → list of active satellites from TLE data
export async function getSatellites() {
  const res = await fetch(`${BASE_URL}/satellites`)
  return res.json()
}

// POST /simulate  → run orbit simulation, get trajectory data
export async function simulate(params) {
  const res = await fetch(`${BASE_URL}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  })
  return res.json()
}
```

---

## ⚡ Common Issues

| Problem | Fix |
|---|---|
| Backend OFFLINE shown in topbar | Start Flask: `python app.py` in Backend/ml/ |
| CORS error in browser console | Make sure `CORS(app)` is in your app.py |
| `npm install` fails | Use Node 18+: `node --version` |
| White screen | Check browser console, run `npm run dev` |
| Port conflict | Change Vite port in `vite.config.js` → `port: 3000` |
