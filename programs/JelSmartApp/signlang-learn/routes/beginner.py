from flask import Blueprint, render_template, abort, request
from config import Config, db
from models.tasklist import TaskList
from routes.home import load_translation
from sqlalchemy.exc import OperationalError

beginner_bp = Blueprint("beginner", __name__)

@beginner_bp.route("/beginner")
def beginner():
    lang = request.args.get("lang", "hu")
    texts = load_translation(lang)
    try:
        rows = TaskList.query.filter(
            (TaskList.task_difficulty_id == 1)
        ).all()
    except OperationalError:
        return render_template("kezdo.html", beginner=None, texts=texts, lang=lang)
    
    return render_template("kezdo.html", beginner=rows, texts=texts, lang=lang)
