from flask import Flask
from config import Config
from admin.routes import register_blueprints
from config import Config, db



def create_app():
    admin_app = Flask(
    __name__, 
    template_folder="admin/templates", 
    static_folder="admin/static"
)
    admin_app.config.from_object(Config)

    db.init_app(admin_app)

    register_blueprints(admin_app)

    return admin_app

if __name__ == "__main__":
    admin_app = create_app()

    with admin_app.app_context():
        db.create_all() 
    admin_app.run(debug=True)

    #admin_app.run(host='0.0.0.0', port=6750, debug=True)