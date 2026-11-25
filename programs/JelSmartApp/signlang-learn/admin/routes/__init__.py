from admin.routes.home import admin_bp
from admin.routes.admin_login import login_bp
from admin.routes.crud import crud_bp
from admin.routes.media import media_bp
from admin.routes.neutral import neutral_bp


def register_blueprints(app):
    app.register_blueprint(admin_bp)
    app.register_blueprint(login_bp)
    app.register_blueprint(crud_bp)
    app.register_blueprint(media_bp)
    app.register_blueprint(neutral_bp)

