import os
import sqlite3
from flask import Flask
from flask_login import LoginManager
from .models import User

# ----------------------------
# LOGIN MANAGER
# ----------------------------
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message_category = "info"


# ----------------------------
# CREATE APP FACTORY
# ----------------------------
def create_app():
    app = Flask(__name__)

    # Secret key
    app.config["SECRET_KEY"] = "energy_secret_key"

    # ----------------------------
    # DATABASE SETUP (SAFE PATH)
    # ----------------------------
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    db_path = os.path.join(BASE_DIR, "..", "users.db")

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            login_count INTEGER DEFAULT 0,
            last_login TEXT
        )
    """)
    conn.commit()
    conn.close()

    # ----------------------------
    # INIT LOGIN MANAGER
    # ----------------------------
    login_manager.init_app(app)

    # ----------------------------
    # REGISTER BLUEPRINTS
    # ----------------------------
    from .routes import main
    from .auth import auth
    from .admin import admin

    app.register_blueprint(main)
    app.register_blueprint(auth)
    app.register_blueprint(admin)

    return app


# ----------------------------
# USER LOADER
# ----------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)
