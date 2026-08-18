from functools import wraps

from flask import jsonify
from flask_jwt_extended import get_jwt_identity, jwt_required

from models import User, Prediction

ROLES = ('viewer', 'operator', 'admin')

ROLE_RANK = {
    'viewer': 1,
    'operator': 2,
    'admin': 3,
}


def get_current_user():
    user_id = get_jwt_identity()
    if user_id is None:
        return None
    return User.query.get(user_id)


def role_required(*allowed_roles):
    def decorator(fn):
        @wraps(fn)
        @jwt_required()
        def wrapper(*args, **kwargs):
            user = get_current_user()
            if not user:
                return jsonify({'status': 'error', 'message': 'User not found'}), 404
            if not user.is_active:
                return jsonify({'status': 'error', 'message': 'Account deactivated'}), 403
            if user.role not in allowed_roles:
                return jsonify({
                    'status': 'error',
                    'message': 'Insufficient permissions for this action',
                    'required_roles': list(allowed_roles),
                    'your_role': user.role,
                }), 403
            return fn(*args, current_user=user, **kwargs)

        return wrapper

    return decorator


def predictions_for_user(user):
    query = Prediction.query
    if user.role == 'viewer':
        query = query.filter_by(user_id=user.id)
    return query
