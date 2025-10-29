from flask import render_template, Blueprint

guide_bp = Blueprint('guide', __name__)

@guide_bp.route('/guide')
def user_guide():
    return render_template('user_guide.html')