import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'astraeus-dev-secret-change-in-production')
    DEBUG = False
    TESTING = False

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'sqlite:///astraeus.db'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }

    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'astraeus-jwt-secret-change-in-production')
    JWT_ACCESS_TOKEN_EXPIRES = 3600
    JWT_REFRESH_TOKEN_EXPIRES = 86400

    # Dashboard polls /history + /stats; keep room for that while still capping abuse.
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '1000 per day;300 per hour')
    RATELIMIT_STORAGE_URI = os.environ.get('RATELIMIT_STORAGE_URI', 'memory://')

    CORS_ORIGINS = [
        o.strip()
        for o in os.environ.get('CORS_ORIGINS', 'http://localhost:5173').split(',')
        if o.strip()
    ]

    MODEL_PATH = os.environ.get('MODEL_PATH', 'collision_model.pkl')


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False

    @staticmethod
    def init_uri():
        uri = os.environ.get('DATABASE_URL', '')
        return uri.replace('postgres://', 'postgresql://') if uri else 'sqlite:///astraeus.db'


ProductionConfig.SQLALCHEMY_DATABASE_URI = ProductionConfig.init_uri()


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    RATELIMIT_STORAGE_URI = 'memory://'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig,
}
