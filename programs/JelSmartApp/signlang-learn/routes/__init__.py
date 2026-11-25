from routes.home import home_bp
from routes.beginner import beginner_bp
from routes.semi_pro import semipro_bp
from routes.pro import pro_bp
from routes.register import register_bp
from routes.login import login_bp
from routes.neutral import neutral_bp
from routes.model import dactyl_bp
from routes.category import category_bp
from routes.guide import guide_bp
from routes.media import media_bp


def register_blueprints(app):
    app.register_blueprint(home_bp)
    app.register_blueprint(beginner_bp)
    app.register_blueprint(semipro_bp)
    app.register_blueprint(pro_bp)
    app.register_blueprint(register_bp)
    app.register_blueprint(login_bp)
    app.register_blueprint(neutral_bp)
    app.register_blueprint(dactyl_bp)
    app.register_blueprint(category_bp)
    app.register_blueprint(guide_bp)
    app.register_blueprint(media_bp)

