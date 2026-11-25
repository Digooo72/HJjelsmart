from flask import Blueprint, render_template,abort,request
from config import Config, db
from models.tasklist import TaskList
from routes.home import load_translation
from sqlalchemy.exc import OperationalError


semipro_bp = Blueprint("semipro", __name__)

@semipro_bp.route("/semipro")
def semipro():
    lang = request.args.get("lang", "hu")
    texts = load_translation(lang)
    try:
        rows = TaskList.query.filter(
            (TaskList.task_difficulty_id == 2)
        ).all()
    except OperationalError:
        return render_template("kozephalado.html", semipro=None, texts=texts, lang=lang)

    return render_template("kozephalado.html", semipro=rows, texts=texts, lang=lang)

