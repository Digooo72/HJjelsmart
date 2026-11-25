from flask import Blueprint, render_template, request
import os, json

home_bp = Blueprint("home", __name__)


def load_translation(lang):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    translations_path = os.path.join(base_dir, "..", "translations", f"{lang}.json")
    print("Looking for:", translations_path)
    if os.path.exists(translations_path):
        with open(translations_path, "r", encoding="utf-8") as f:
            return json.load(f)
 
    return {}

@home_bp.route("/")
def home():
    lang = request.args.get("lang", "hu")
    texts = load_translation(lang)
    return render_template("index.html", texts=texts, lang=lang)


@home_bp.route("/gyakorlok")
def gyakorlok():
    lang = request.args.get("lang", "hu")  
    texts = load_translation(lang)
    return render_template("gyakorlok.html", texts=texts, lang=lang)
