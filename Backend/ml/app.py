from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import os
from datetime import datetime
from risk_scorer import score_from_features
from avoidance import full_assessment

app = Flask(__name__)
CORS(app)

# load model once when server starts
model = joblib.load('collision_model.pkl')
print("Model loaded successfully!")

# history file path
HISTORY_FILE = 'prediction_history.json'

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, 'r') as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def add_to_history(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:100]  # keep last 100
    save_history(history)

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'ASTRAEUS — Space Debris Sentinel API',
        'status':  'running',
        'endpoints': {
            '/predict':      'POST - predict collision risk',
            '/health':       'GET  - check if API is running',
            '/history':      'GET  - last 100 predictions',
            '/stats':        'GET  - prediction statistics',
            '/history/clear':'DELETE - clear history',
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':    'OK',
        'model':     'collision_model.pkl loaded',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data          = request.get_json()
        distance_km   = float(data['distance_km'])
        rel_velocity  = float(data['rel_velocity'])
        approach_rate = float(data['approach_rate'])

        result = full_assessment(model, distance_km, rel_velocity, approach_rate)

        # build confidence score
        prob = result['probability']
        if prob >= 85 or prob <= 15:
            confidence = 'HIGH'
        elif prob >= 60 or prob <= 40:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        response = {
            'status':         'success',
            'timestamp':      datetime.utcnow().isoformat(),
            'input': {
                'distance_km':   distance_km,
                'rel_velocity':  rel_velocity,
                'approach_rate': approach_rate
            },
            'risk_level':     result['level'],
            'probability':    result['probability'],
            'confidence':     confidence,
            'color':          result['color'],
            'message':        result['message'],
            'action':         result['action'],
            'maneuver_type':  result['maneuver_type'],
            'maneuver_km':    result['maneuver_km'],
            'fuel_cost_kg':   result['fuel_cost_kg'],
            'time_window':    result['time_window'],
            'urgency':        result['urgency']
        }

        # save to history
        add_to_history(response)

        return jsonify(response)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/history', methods=['GET'])
def history():
    """Returns last 100 predictions"""
    try:
        limit   = int(request.args.get('limit', 100))
        risk    = request.args.get('risk', None)  # filter by HIGH/MEDIUM/LOW

        data = load_history()

        if risk:
            data = [d for d in data if d.get('risk_level') == risk.upper()]

        return jsonify({
            'status':  'success',
            'count':   len(data[:limit]),
            'history': data[:limit]
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/history/clear', methods=['DELETE'])
def clear_history():
    """Clears all prediction history"""
    try:
        save_history([])
        return jsonify({'status': 'success', 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/stats', methods=['GET'])
def stats():
    """Returns prediction statistics"""
    try:
        history = load_history()

        if not history: