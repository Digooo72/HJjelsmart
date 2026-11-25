from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from config import Config, db
from models import User

login_bp = Blueprint("login", __name__, template_folder="templates")

@login_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()

        if not username or not password:
            flash("type blyat.", "danger")
            return redirect(url_for("login.login"))

        user = User.query.filter_by(user_name=username).first()
        if user and user.check_password(password):
           
            session["user_id"] = user.user_id
            session["username"] = user.user_name
            session["user_role"] = user.user_role

            flash("sikeres belepes!", "success")
            return redirect(url_for("login.dashboard"))  
        else:
            flash("hibas data.", "danger")
            return redirect(url_for("login.login"))

    return render_template("login.html")

@login_bp.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Nincs log be ")
        return redirect(url_for("login.login"))
    return render_template("logintest.html", username=session["username"])


@login_bp.route("/logout")
def logout():
   
    session.clear()
    flash("Kijelentkezve")
    return redirect(url_for("login.login"))