from flask_sqlalchemy import SQLAlchemy
from config import Config, db


class Category(db.Model):

    __tablename__ = "category"
    id = db.Column(db.Integer,primary_key=True)
    category_name = db.Column(db.String(255),nullable = False)
