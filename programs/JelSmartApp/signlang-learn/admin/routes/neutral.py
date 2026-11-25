from flask import Blueprint, render_template,abort,redirect,request,flash,url_for,send_from_directory
from config import Config,db
from models.tasklist import TaskList
from models.difficulty import Difficulty
from werkzeug.utils import secure_filename
import os



neutral_bp = Blueprint("neutral_bp",__name__)

def diff_id_to_String(difficulty_id):
     if difficulty_id == 1:
          return "Kezdo"
     elif difficulty_id == 2:
          return "Közephaladó"
     else:
          return "Haladó"




@neutral_bp.route("/admin-beginner")
def show_beginner():
      rows = TaskList.query.filter(
    TaskList.task_difficulty_id == 1
).order_by(TaskList.task_description.asc()).all()
      return render_template("admin-list-tasks.html",task = rows,difficulty = diff_id_to_String(1))


@neutral_bp.route("/admin-semipro")
def show_semipro():
      rows = TaskList.query.filter(
    TaskList.task_difficulty_id == 2
).order_by(TaskList.task_description.asc()).all()
      return render_template("admin-list-tasks.html",task = rows, difficulty = diff_id_to_String(2))


@neutral_bp.route("/admin-pro")
def show_pro():
    rows = TaskList.query.filter(
    TaskList.task_difficulty_id == 3
).order_by(TaskList.task_description.asc()).all()
    return render_template("admin-list-tasks.html",task = rows, difficulty = diff_id_to_String(3))



@neutral_bp.route("/admin-listusers")
def show_users():
    return render_template("admin-listusers.html")

@neutral_bp.route("/admin-showmodel")
def show_model():
    return render_template("admin-showmodel.html")
    

