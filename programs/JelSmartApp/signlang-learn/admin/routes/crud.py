from flask import Blueprint, render_template,abort,redirect,request,flash,url_for,send_from_directory
from config import Config,db
from models.tasklist import TaskList
from models.difficulty import Difficulty
from werkzeug.utils import secure_filename
import os

from admin.routes.media import IMAGES_FOLDER,VIDEOS_FOLDER
from admin.routes.neutral import diff_id_to_String


crud_bp = Blueprint("crud_bp",__name__)

@crud_bp.route("/delete/<int:task_id>/<int:difficulty_id>", methods=["POST"])
def delete(task_id,difficulty_id):
    row = TaskList.query.get(task_id)
    if not row:
        abort(404)
    db.session.delete(row)
    db.session.commit()

    rows = TaskList.query.filter(
        TaskList.task_difficulty_id == difficulty_id
    ).order_by(TaskList.task_description.asc()).all()
    return render_template("admin-list-tasks.html",task = rows, difficulty = diff_id_to_String(difficulty_id))


@crud_bp.route("/create", methods=["GET", "POST"])
def create():
    if request.method == "POST":
        difficulty_id = request.form.get("task_difficulty_id")
        description_letter = request.form.get("description_letter")
        description = request.form.get("description")
        language_id = request.form.get("language_id")

    
        image_file = request.files.get("image")
        video_file = request.files.get("video")

        image_filename = 'default.png'
        video_filename = 'default.mp4'

        if image_file and image_file.filename:
            image_filename = secure_filename(image_file.filename)
            image_file.save(os.path.join(IMAGES_FOLDER, image_filename))

        if video_file and video_file.filename:
            video_filename = secure_filename(video_file.filename)
            video_file.save(os.path.join(VIDEOS_FOLDER, video_filename))

        if TaskList.query.filter_by(task_description_letter=description_letter).first():
            flash("Van már ilyen feladat!", "error")
            
        else:
            new_task = TaskList(
                task_difficulty_id=difficulty_id,
                task_description_letter=description_letter,
                task_description=description,
                task_image=image_filename,
                task_video_url=video_filename,
                task_language_id=language_id,
            )
            db.session.add(new_task)
            db.session.commit()
            flash("Sikeres felvétel!", "success")

    difficulty_id = request.form.get("task_difficulty_id")
    if difficulty_id == "1":
        return redirect(url_for("neutral_bp.show_beginner"))
    elif difficulty_id == "2":
        return redirect(url_for("neutral_bp.show_semipro"))
    elif difficulty_id == "3":
        return redirect(url_for("neutral_bp.show_pro"))
    
    
    return render_template("admin-edit-task.html")

@crud_bp.route("/edit_task")
def edit_task():
    difficulties = Difficulty.query.order_by(Difficulty.difficulty_id).all()

    return render_template("admin-create-task.html", difficulties=difficulties)



@crud_bp.route("/edit/<int:task_id>", methods=["GET", "POST"])
def edit(task_id):
    task = TaskList.query.get_or_404(task_id)  
    difficulties = Difficulty.query.all()      

    if request.method == "POST":
        task.task_difficulty_id = request.form.get("task_difficulty_id")
        task.task_description_letter = request.form.get("description_letter")
        task.task_description = request.form.get("description")
        task.task_language_id = request.form.get("language_id")


        image_file = request.files.get("image")
        if image_file and image_file.filename:
            image_filename = secure_filename(image_file.filename)
            image_file.save(os.path.join(IMAGES_FOLDER, image_filename))
            task.task_image = image_filename

        video_file = request.files.get("video")
        if video_file and video_file.filename:
            video_filename = secure_filename(video_file.filename)
            video_file.save(os.path.join(VIDEOS_FOLDER, video_filename))
            task.task_video_url = video_filename

        db.session.commit()
        flash("Feladat sikeresen frissítve!", "success")

        if task.task_difficulty_id == 1:
            return redirect(url_for("neutral_bp.show_beginner"))
        elif task.task_difficulty_id == 2:
            return redirect(url_for("neutral_bp.show_semipro"))
        elif task.task_difficulty_id == 3:
            return redirect(url_for("neutral_bp.show_pro"))

    return render_template("admin-edit-task.html", task=task, difficulties=difficulties)

