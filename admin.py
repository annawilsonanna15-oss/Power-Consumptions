from flask import Blueprint, render_template
from flask_login import login_required, current_user
from functools import wraps
from flask import abort
import sqlite3
import os

# IMPORTANT: Blueprint name MUST be "admin"
admin = Blueprint("admin", __name__)

# ----------------------------
# ADMIN REQUIRED DECORATOR
# ----------------------------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_user.role != "admin":
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


# ----------------------------
# ADMIN PANEL ROUTE
# ----------------------------
@admin.route("/admin")
@login_required
@admin_required
def admin_panel():

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    db_path = os.path.join(BASE_DIR, "..", "users.db")

    conn = sqlite3.connect(db_path)

    users = conn.execute("""
        SELECT username, role, login_count, last_login
        FROM users
    """).fetchall()

    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_logins = conn.execute("SELECT SUM(login_count) FROM users").fetchone()[0]

    conn.close()

    return render_template(
        "admin.html",
        users=users,
        total_users=total_users,
        total_logins=total_logins
    )
