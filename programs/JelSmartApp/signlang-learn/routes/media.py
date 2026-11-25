from flask import Blueprint, render_template,abort,redirect,request,flash,url_for,send_from_directory
from config import Config,db
from models.tasklist import TaskList
from models.difficulty import Difficulty
from werkzeug.utils import secure_filename
import os

media_bp = Blueprint("media_bp",__name__)


IMAGES_FOLDER= "/home/ihne/jelsmart-server/programs/media/images"  
VIDEOS_FOLDER= "/home/ihne/jelsmart-server/programs/media/videos"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "mp4", "avi"}




@media_bp.route("/showmedia")
def show_media():
    images_rows = (
        TaskList.query
        .filter(TaskList.task_image.isnot(None), TaskList.task_image != "")
        .all()
    )
    video_rows = (
        TaskList.query
        .filter(TaskList.task_video_url.isnot(None), TaskList.task_video_url != "")
        .all()
    )
    return render_template("admin-edit-galery.html", media_videos = video_rows, media_images = images_rows)



@media_bp.route("/delete_media/<int:media_id>", methods=["POST"])
def delete_media(media_id):
    row = TaskList.query.get_or_404(media_id)

    if row.task_image:
        image_path = os.path.join(IMAGES_FOLDER, row.task_image)
        if os.path.isfile(image_path):
            os.remove(image_path)
        row.task_image = None  

    if row.task_video_url:
        video_path = os.path.join(VIDEOS_FOLDER, row.task_video_url)
        if os.path.isfile(video_path):
            os.remove(video_path)
        row.task_video_url = None 

    db.session.commit()

    return redirect(request.referrer or url_for("edit.show_media"))

@media_bp.route('/images/<path:filename>')
def images(filename):
    return send_from_directory(IMAGES_FOLDER, filename)


@media_bp.route('/videos/<path:filename>')
def videos(filename):
    return send_from_directory(VIDEOS_FOLDER, filename)
