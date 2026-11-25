from flask import Blueprint, render_template, abort, request
from config import Config, db
from models.tasklist import TaskList
from routes.home import load_translation
from sqlalchemy.exc import OperationalError

pro_bp = Blueprint("pro", __name__)

@pro_bp.route("/pro")
def semipro():
    lang = request.args.get("lang", "hu")
    texts = load_translation(lang)
    try:
        rows = TaskList.query.filter(
            (TaskList.task_difficulty_id == 3)
        ).all()
    except OperationalError:
        return render_template("halado.html", pro=None, texts=texts, lang=lang)

    return render_template("halado.html", pro=rows, texts=texts, lang=lang)

