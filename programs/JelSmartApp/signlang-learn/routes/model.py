from flask import Blueprint, render_template, request
from routes.home import load_translation

dactyl_bp = Blueprint('dactyl', __name__, template_folder='templates')

@dactyl_bp.route('/dactyl')
def dactyl_page():
    lang = request.args.get("lang", "hu")
    texts = load_translation(lang)
    return render_template('model.html', texts=texts, lang=lang)