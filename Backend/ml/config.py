import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'astraeus-dev-secret-change-in-production')
    DEBUG = False
    TESTING = False

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'sqlite:///astraeus.db'   # fallback for local dev
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle':  300,
    }

    # JWT
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'astraeus-jwt-secret-change-in-production')
    JWT_ACCESS_TOKEN_EXPIRES  = 3600   # 1 hour
    JWT_REFRESH_TOKEN_EXPIRES = 86400  # 24 hours

    # Rate limiting
    RATELIMIT_DEFAULT          = '200 per day, 50 per hour'
    RATELIMIT_STORAGE_URL      = 'memory://'

    # CORS
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:5173').split(',')

    # ML Model
    MODEL_PATH = os.environ.get('MODEL_PATH', 'collision_model.pkl')


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', '').replace(
        'postgres://', 'postgresql://'   # Railway fix
    )


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


config = {
    'development': DevelopmentConfig,
    'production':  ProductionConfig,
    'testing':     TestingConfig,
    'default':     DevelopmentConfig,
}