from flask_sqlalchemy import SQLAlchemy
from config import Config, db

from .difficulty import Difficulty
from .users import User
from .tasklist import TaskList 
from .language import Language
from .category import Category
