from flask import Blueprint, render_template, abort, request, jsonify
from sqlalchemy.exc import OperationalError
from config import Config, db
from models.category import Category  
from models.tasklist import TaskList
from routes.home import load_translation

category_bp = Blueprint("category", __name__)

@category_bp.route("/category")
def categories():
    lang = request.args.get("lang", "hu")
    texts = load_translation(lang)
    try:
        rows = Category.query.all()
    except OperationalError:
        return render_template("kategoriak.html", categories=None, texts=texts, lang=lang)

    return render_template("kategoriak.html", categories=rows, texts=texts, lang=lang)

@category_bp.route("/categories/<int:category_id>")
def specific_category(category_id):
    lang = request.args.get("lang", "hu")
    texts = load_translation(lang)
    try:
        rows = TaskList.query.filter(
            (TaskList.task_category_id == category_id)
        ).all()
    except OperationalError:
        return render_template("kozephalado.html", semipro=None, texts=texts, lang=lang)

    return render_template("kozephalado.html", semipro=rows, texts=texts, lang=lang)
