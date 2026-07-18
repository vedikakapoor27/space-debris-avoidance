import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from flask_migrate import Migrate
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
import joblib
from datetime import datetime

from config import config
from models import db, Prediction, ConjunctionEvent, User
from auth import auth_bp
from risk_scorer import score_from_features
from avoidance import full_assessment

load_dotenv()

def create_app(config_name=None):
    app = Flask(__name__)

    # Load config
    config_name = config_name or os.environ.get('FLASK_ENV', 'development')
    app.config.from_object(config[config_name])

    # Extensions
    db.init_app(app)
    Migrate(app, db)
    JWTManager(app)
    CORS(app, origins=app.config['CORS_ORIGINS'])

    # Rate limiter
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=['200 per day', '50 per hour']
    )

    # Register blueprints
    app.register_blueprint(auth_bp)

    # Load ML model
    model_path = app.config['MODEL_PATH']
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # ─────────────────────────────
    # ROUTES
    # ─────────────────────────────

    @app.route('/', methods=['GET'])
    def home():
        return jsonify({
            'name':    'ASTRAEUS API',
            'version': '2.4.1',
            'status':  'running',
            'endpoints': {
                '/health':          'GET  - health check',
                '/predict':         'POST - collision prediction',
                '/history':         'GET  - prediction history',
                '/stats':           'GET  - statistics',
                '/conjunctions':    'GET  - conjunction events',
                '/auth/register':   'POST - create account',
                '/auth/login':      'POST - login',
                '/auth/me':         'GET  - current user',
            }
        })

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status':    'OK',
            'timestamp': datetime.utcnow().isoformat(),
            'model':     'loaded',
            'database':  'connected'
        })

    @app.route('/predict', methods=['POST'])
    @limiter.limit('30 per minute')
    def predict():
        try:
            data          = request.get_json()
            distance_km   = float(data['distance_km'])
            rel_velocity  = float(data['rel_velocity'])
            approach_rate = float(data['approach_rate'])

            result = full_assessment(model, distance_km, rel_velocity, approach_rate)

            prob = result['probability']
            if prob >= 85 or prob <= 15:
                confidence = 'HIGH'
            elif prob >= 60 or prob <= 40:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'

            # Save to database
            prediction = Prediction(
                distance_km   = distance_km,
                rel_velocity  = rel_velocity,
                approach_rate = approach_rate,
                risk_level    = result['level'],
                probability   = result['probability'],
                confidence    = confidence,
                action        = result['action'],
                maneuver_type = result['maneuver_type'],
                maneuver_km   = result['maneuver_km'],
                fuel_cost_kg  = result['fuel_cost_kg'],
                time_window   = result['time_window'],
                urgency       = result['urgency'],
            )
            db.session.add(prediction)
            db.session.commit()

            return jsonify({
                'status':        'success',
                'timestamp':     datetime.utcnow().isoformat(),
                'input': {
                    'distance_km':   distance_km,
                    'rel_velocity':  rel_velocity,
                    'approach_rate': approach_rate,
                },
                'risk_level':    result['level'],
                'probability':   result['probability'],
                'confidence':    confidence,
                'color':         result['color'],
                'message':       result['message'],
                'action':        result['action'],
                'maneuver_type': result['maneuver_type'],
                'maneuver_km':   result['maneuver_km'],
                'fuel_cost_kg':  result['fuel_cost_kg'],
                'time_window':   result['time_window'],
                'urgency':       result['urgency'],
            })

        except KeyError as e:
            return jsonify({'status': 'error', 'message': f'Missing field: {e}'}), 400
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/history', methods=['GET'])
    def history():
        try:
            limit     = int(request.args.get('limit', 50))
            risk      = request.args.get('risk', None)
            page      = int(request.args.get('page', 1))

            query = Prediction.query.order_by(Prediction.timestamp.desc())

            if risk:
                query = query.filter_by(risk_level=risk.upper())

            total     = query.count()
            items     = query.offset((page-1)*limit).limit(limit).all()

            return jsonify({
                'status':  'success',
                'total':   total,
                'page':    page,
                'limit':   limit,
                'history': [p.to_dict() for p in items]
            })

        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/history/clear', methods=['DELETE'])
    def clear_history():
        try:
            Prediction.query.delete()
            db.session.commit()
            return jsonify({'status': 'success', 'message': 'History cleared'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/stats', methods=['GET'])
    def stats():
        try:
            total  = Prediction.query.count()

            if total == 0:
                return jsonify({'status': 'success', 'total_predictions': 0, 'message': 'No predictions yet'})

            high   = Prediction.query.filter_by(risk_level='HIGH').count()
            medium = Prediction.query.filter_by(risk_level='MEDIUM').count()
            low    = Prediction.query.filter_by(risk_level='LOW').count()

            probs  = [p.probability for p in Prediction.query.all()]
            latest = Prediction.query.order_by(Prediction.timestamp.desc()).first()

            high_pct = round((high / total) * 100, 1)

            return jsonify({
                'status':            'success',
                'total_predictions': total,
                'by_risk': {
                    'HIGH':   high,
                    'MEDIUM': medium,
                    'LOW':    low,
                },
                'percentages': {
                    'HIGH':   high_pct,
                    'MEDIUM': round((medium / total) * 100, 1),
                    'LOW':    round((low    / total) * 100, 1),
                },
                'probability': {
                    'average': round(sum(probs) / len(probs), 2),
                    'max':     round(max(probs), 2),
                    'min':     round(min(probs), 2),
                },
                'latest': latest.to_dict() if latest else None,
                'alert':  high_pct > 30,
            })

        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/conjunctions', methods=['GET'])
    def conjunctions():
        try:
            events = ConjunctionEvent.query.order_by(
                ConjunctionEvent.timestamp.desc()
            ).limit(20).all()
            return jsonify({
                'status': 'success',
                'count':  len(events),
                'data':   [e.to_dict() for e in events]
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    # Create tables on first run
    with app.app_context():
        db.create_all()
        print("Database tables ready!")

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)