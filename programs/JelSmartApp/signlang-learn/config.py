import os
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Config:
    SQLALCHEMY_DATABASE_URI = (
        "postgresql+psycopg2://dev:ihnedev@server.jelsmart.hu/jelsmartdb"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = "supersecretkey123"


