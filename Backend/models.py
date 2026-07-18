from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'users'

    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(80),  unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(256), nullable=False)
    role       = db.Column(db.String(20),  default='viewer')  # admin/operator/viewer
    created_at = db.Column(db.DateTime,    default=datetime.utcnow)
    is_active  = db.Column(db.Boolean,     default=True)

    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def to_dict(self):
        return {
            'id':         self.id,
            'username':   self.username,
            'email':      self.email,
            'role':       self.role,
            'created_at': self.created_at.isoformat(),
            'is_active':  self.is_active,
        }


class Prediction(db.Model):
    __tablename__ = 'predictions'

    id            = db.Column(db.Integer, primary_key=True)
    timestamp     = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # Input features
    distance_km   = db.Column(db.Float, nullable=False)
    rel_velocity  = db.Column(db.Float, nullable=False)
    approach_rate = db.Column(db.Float, nullable=False)

    # ML output
    risk_level    = db.Column(db.String(10),  nullable=False, index=True)
    probability   = db.Column(db.Float,       nullable=False)
    confidence    = db.Column(db.String(10))

    # Avoidance
    action        = db.Column(db.String(100))
    maneuver_type = db.Column(db.String(100))
    maneuver_km   = db.Column(db.String(50))
    fuel_cost_kg  = db.Column(db.String(50))
    time_window   = db.Column(db.String(100))
    urgency       = db.Column(db.String(20))

    # Optional user link
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

    def to_dict(self):
        return {
            'id':            self.id,
            'timestamp':     self.timestamp.isoformat(),
            'input': {
                'distance_km':   self.distance_km,
                'rel_velocity':  self.rel_velocity,
                'approach_rate': self.approach_rate,
            },
            'risk_level':    self.risk_level,
            'probability':   self.probability,
            'confidence':    self.confidence,
            'action':        self.action,
            'maneuver_type': self.maneuver_type,
            'maneuver_km':   self.maneuver_km,
            'fuel_cost_kg':  self.fuel_cost_kg,
            'time_window':   self.time_window,
            'urgency':       self.urgency,
        }


class ConjunctionEvent(db.Model):
    __tablename__ = 'conjunction_events'

    id           = db.Column(db.Integer, primary_key=True)
    timestamp    = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    object1      = db.Column(db.String(50), nullable=False)
    object2      = db.Column(db.String(50), nullable=False)
    distance_km  = db.Column(db.Float,      nullable=False)
    cpa_time_sec = db.Column(db.Float)
    status       = db.Column(db.String(20), nullable=False, index=True)

    def to_dict(self):
        return {
            'id':           self.id,
            'timestamp':    self.timestamp.isoformat(),
            'object1':      self.object1,
            'object2':      self.object2,
            'distance_km':  self.distance_km,
            'cpa_time_sec': self.cpa_time_sec,
            'status':       self.status,
        }