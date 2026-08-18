from flask import Blueprint, request, jsonify
from models import db, User
from roles import role_required, ROLES

admin_bp = Blueprint('admin', __name__)


@admin_bp.route('/admin/users', methods=['GET'])
@role_required('admin')
def list_users(current_user):
    users = User.query.order_by(User.created_at.asc()).all()
    return jsonify({
        'status': 'success',
        'count': len(users),
        'users': [u.to_dict() for u in users],
    })


@admin_bp.route('/admin/users/<int:user_id>', methods=['PATCH'])
@role_required('admin')
def update_user(user_id, current_user):
    try:
        target = User.query.get(user_id)
        if not target:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404

        data = request.get_json() or {}
        new_role = data.get('role')
        new_active = data.get('is_active')

        if new_role is not None:
            if new_role not in ROLES:
                return jsonify({
                    'status': 'error',
                    'message': f'Invalid role. Must be one of: {", ".join(ROLES)}',
                }), 400
            if target.id == current_user.id and new_role != 'admin':
                return jsonify({
                    'status': 'error',
                    'message': 'You cannot remove your own admin role',
                }), 400
            target.role = new_role

        if new_active is not None:
            if target.id == current_user.id and not new_active:
                return jsonify({
                    'status': 'error',
                    'message': 'You cannot deactivate your own account',
                }), 400
            target.is_active = bool(new_active)

        db.session.commit()
        return jsonify({
            'status': 'success',
            'message': 'User updated',
            'user': target.to_dict(),
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
