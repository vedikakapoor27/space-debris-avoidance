import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
import joblib
from datetime import datetime
from sqlalchemy import text

from config import config
from models import db, Prediction, ConjunctionEvent, User
from auth import auth_bp
from admin import admin_bp
from roles import role_required, get_current_user, predictions_for_user
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

    raw_limits = app.config.get('RATELIMIT_DEFAULT', '1000 per day;300 per hour')
    default_limits = [part.strip() for part in raw_limits.replace(',', ';').split(';') if part.strip()]

    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=default_limits,
        storage_uri=app.config.get('RATELIMIT_STORAGE_URI', 'memory://'),
    )

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)

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
                '/admin/users':     'GET  - list users (admin)',
            }
        })

    @app.route('/health', methods=['GET'])
    @limiter.exempt
    def health():
        db_status = 'disconnected'
        try:
            db.session.execute(text('SELECT 1'))
            db_status = 'connected'
        except Exception as exc:
            db_status = f'error: {exc.__class__.__name__}'

        model_status = 'loaded' if model is not None else 'missing'
        healthy = db_status == 'connected' and model_status == 'loaded'

        return jsonify({
            'status':    'OK' if healthy else 'DEGRADED',
            'timestamp': datetime.utcnow().isoformat(),
            'model':     model_status,
            'database':  db_status,
        }), 200 if healthy else 503

    @app.route('/predict', methods=['POST'])
    @limiter.limit('30 per minute')
    @role_required('operator', 'admin')
    def predict(current_user):
        try:
            user_id       = current_user.id
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
                user_id       = user_id,
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
    @role_required('viewer', 'operator', 'admin')
    def history(current_user):
        try:
            limit     = int(request.args.get('limit', 50))
            risk      = request.args.get('risk', None)
            page      = int(request.args.get('page', 1))

            query = predictions_for_user(current_user)
            query = query.order_by(Prediction.timestamp.desc())

            if risk:
                query = query.filter_by(risk_level=risk.upper())

            total     = query.count()
            items     = query.offset((page-1)*limit).limit(limit).all()

            return jsonify({
                'status':  'success',
                'total':   total,
                'page':    page,
                'limit':   limit,
                'scope':   'own' if current_user.role == 'viewer' else 'all',
                'history': [p.to_dict() for p in items]
            })

        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/history/clear', methods=['DELETE'])
    @role_required('admin')
    def clear_history(current_user):
        try:
            Prediction.query.delete()
            db.session.commit()
            return jsonify({'status': 'success', 'message': 'History cleared'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/stats', methods=['GET'])
    @role_required('viewer', 'operator', 'admin')
    def stats(current_user):
        try:
            base_query = predictions_for_user(current_user)
            total  = base_query.count()

            if total == 0:
                return jsonify({'status': 'success', 'total_predictions': 0, 'message': 'No predictions yet'})

            high   = base_query.filter_by(risk_level='HIGH').count()
            medium = base_query.filter_by(risk_level='MEDIUM').count()
            low    = base_query.filter_by(risk_level='LOW').count()

            items  = base_query.all()
            probs  = [p.probability for p in items]
            latest = base_query.order_by(Prediction.timestamp.desc()).first()

            high_pct = round((high / total) * 100, 1)

            return jsonify({
                'status':            'success',
                'scope':             'own' if current_user.role == 'viewer' else 'all',
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
    @role_required('viewer', 'operator', 'admin')
    def conjunctions(current_user):
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
    # Local only. Prefer: gunicorn --bind 0.0.0.0:5000 --workers 2 app:app
    debug = os.environ.get('FLASK_ENV', 'development') != 'production'
    app.run(host='0.0.0.0', debug=debug, port=int(os.environ.get('PORT', 5000)))