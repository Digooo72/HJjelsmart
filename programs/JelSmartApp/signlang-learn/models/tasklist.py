from config import Config, db

class TaskList(db.Model):

    __tablename__ = "tasklist"
    
    task_id = db.Column(db.Integer, primary_key = True)
    task_difficulty_id = db.Column(db.Integer)
    task_description_letter = db.Column(db.String(255))
    task_description = db.Column(db.String(255))
    task_image = db.Column(db.String(255))
    task_video_url = db.Column(db.String(255))
    task_language_id = db.Column(db.Integer)
    task_category_id = db.Column(db.Integer)