from flask import Flask
from routes import register_blueprints
from config import Config, db

def create_app():
    app = Flask(__name__)

    app.config.from_object(Config)

    db.init_app(app)

    register_blueprints(app)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

   # app.run(host='0.0.0.0', port=3000, debug=True)