from flask import Blueprint, render_template, request, redirect, url_for, flash
from config import Config, db
from models import User

register_bp = Blueprint("register", __name__, template_folder="templates")

@register_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        email = request.form.get("email")

        username = username.strip()
        password = password.strip()
        email = email.strip()


        if User.query.filter_by(user_name=username).first():
            flash("felh nev mar registralt.", "danger")
            return redirect(url_for("register.register"))
        if User.query.filter_by(user_email=email).first():
            flash("email mar registralt.", "danger")
            return redirect(url_for("register.register"))

     
        new_user = User(
            user_name=username,
            user_email=email,
            user_role=1
        )
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        flash("Sikeres reg!", "success")
        return redirect(url_for("register.register"))

    return render_template("regisztracio.html")
