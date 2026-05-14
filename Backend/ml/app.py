from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from risk_scorer import score_from_features
from avoidance import full_assessment

app = Flask(__name__)
CORS(app)  # allows frontend to talk to this API

# load model once when server starts
model = joblib.load('collision_model.pkl')
print("Model loaded successfully!")


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Space Debris Collision Avoidance API',
        'status':  'running',
        'endpoints': {
            '/predict': 'POST - predict collision risk',
            '/health':  'GET  - check if API is running'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main endpoint — frontend or Sanidhya's code sends data here
    Returns risk score + avoidance action
    """
    try:
        # get data sent to the API
        data = request.get_json()

        # extract the 3 values
        distance_km   = float(data['distance_km'])
        rel_velocity  = float(data['rel_velocity'])
        approach_rate = float(data['approach_rate'])

        # run full pipeline
        result = full_assessment(model, distance_km, rel_velocity, approach_rate)

        return jsonify({
            'status':         'success',
            'input': {
                'distance_km':   distance_km,
                'rel_velocity':  rel_velocity,
                'approach_rate': approach_rate
            },
            'risk_level':     result['level'],
            'probability':    result['probability'],
            'color':          result['color'],
            'message':        result['message'],
            'action':         result['action'],
            'maneuver_type':  result['maneuver_type'],
            'maneuver_km':    result['maneuver_km'],
            'fuel_cost_kg':   result['fuel_cost_kg'],
            'time_window':    result['time_window'],
            'urgency':        result['urgency']
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)