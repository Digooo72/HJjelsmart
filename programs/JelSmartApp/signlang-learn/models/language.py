from config import Config, db



class Language(db.Model):
    
    __tablename__ = "languages"

    language_id = db.Column(db.Integer, primary_key = True)
    language_name = db.Column(db.String(255))

    