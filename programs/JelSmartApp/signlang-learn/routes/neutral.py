from flask import Blueprint,render_template,abort,request
from config import Config, db
from models.tasklist import TaskList
from routes.home import load_translation

neutral_bp = Blueprint("neutral",__name__)

@neutral_bp.route('/lecke/<int:lecke_id>')
def lecke(lecke_id):
    lang = request.args.get("lang", "hu")
    texts = load_translation(lang)

    row = TaskList.query.get(lecke_id)
    if not row:
        abort(404)
    return render_template('lecke.html', task=row, texts=texts, lang=lang)

@neutral_bp.route('/testmodel')
def testmodel():
    return render_template('model.html')