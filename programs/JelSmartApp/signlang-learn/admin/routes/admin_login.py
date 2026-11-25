from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from config import Config, db
from models import User

login_bp = Blueprint("login_bp", __name__, template_folder="templates")

@login_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        
        user = User.query.filter_by(user_name=username).first()

        if user and user.check_password(password):
            if user.user_role != 1:
                flash("Nem admin.", "danger")
                return redirect(url_for("login_bp.login"))

          
            session["user_id"] = user.user_id
            session["username"] = user.user_name
            session["user_role"] = user.user_role

            flash("done", "success")
            return redirect(url_for("login_bp.home"))
        else:
            flash("nem done.", "danger")
            return redirect(url_for("login_bp.login"))

    return render_template("admin-login.html")


@login_bp.route("/home")
def home():
    if "user_id" not in session:
        flash("Nincs bejelentkezve.", "warning")
        return redirect(url_for("login_bp.login"))

    
    if session.get("user_role") != 1:
        flash("nem admin.", "danger")
        return redirect(url_for("login_bp.logout"))

    return render_template("admin-landing.html", username=session["username"])


@login_bp.route("/logout",methods=["GET", "POST"])
def logout():
    session.clear()
    flash("Kijelentkezve.", "info")
    return redirect(url_for("login_bp.login"))
