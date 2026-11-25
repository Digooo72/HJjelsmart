from flask_sqlalchemy import SQLAlchemy
from config import Config, db


class Difficulty(db.Model):

    __tablename__ = "difficulty"
    difficulty_id = db.Column(db.Integer,primary_key=True)
    difficulty_name = db.Column (db.String(255),nullable=False)
    difficulty_description = db.Column(db.String(255),nullable=False)
