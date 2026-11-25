from flask import Blueprint, render_template, request
from routes.home import load_translation

guide_bp = Blueprint("guide", __name__)

@guide_bp.route("/kezikonyvek")
def kezikonyvek():
    lang = request.args.get("lang", "hu")
    texts = load_translation(lang)
    return render_template("kezikonyvek.html", texts=texts, lang=lang)