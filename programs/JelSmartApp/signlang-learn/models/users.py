from werkzeug.security import generate_password_hash, check_password_hash
from config import Config, db

class User(db.Model):
    __tablename__ = "users"
    
    user_id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(255))
    user_username = db.Column(db.String(255))
    user_password = db.Column(db.String(255))
    user_email = db.Column(db.String(255))
    user_role = db.Column(db.Integer)
    user_progress_beginer = db.Column(db.Integer, default=0)
    user_progress_semipro = db.Column(db.Integer, default=0)
    user_progress_pro = db.Column(db.Integer, default=0)
    user_progress_sum = db.Column(db.Integer, default=0)
    user_avatar = db.Column(db.String(255))

    def set_password(self, password):
        self.user_password = generate_password_hash(password, method="pbkdf2:sha256")

    def check_password(self, password):
        return check_password_hash(self.user_password, password)
    