
import sqlite3
import os
import hashlib
from flask import Blueprint, render_template, request, redirect, url_for
from flask_login import login_user, logout_user, login_required
from .models import User
import datetime

auth = Blueprint("auth", __name__)

def hash_pwd(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_db():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(BASE_DIR, "..", "users.db")


@auth.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        username = request.form["username"]
        password = hash_pwd(request.form["password"])

        conn = sqlite3.connect(get_db())
        cur = conn.cursor()

        # Fetch username + role
        cur.execute(
            "SELECT username, role FROM users WHERE username=? AND password=?",
            (username, password)
        )

        row = cur.fetchone()

        if row:
            # Extract values properly
            username_db = row[0]
            role_db = row[1]

            # Update login stats
            cur.execute("""
                UPDATE users
                SET login_count = login_count + 1,
                    last_login = ?
                WHERE username=?
            """, (datetime.datetime.now(), username_db))

            conn.commit()

            # Create user object correctly
            user = User(username_db, role_db)
            login_user(user)

            conn.close()
            return redirect(url_for("main.dashboard"))

        conn.close()

    return render_template("login.html")


# @auth.route("/login", methods=["GET", "POST"])
# def login():

#     if request.method == "POST":
#         username = request.form["username"]
#         password = hash_pwd(request.form["password"])

#         conn = sqlite3.connect(get_db())
#         cur = conn.cursor()
#         cur.execute("SELECT username, role FROM users WHERE username=? AND password=?",
#                     (username, password))
#         row = cur.fetchone()

#         if row:
#             cur.execute("""
#                 UPDATE users
#                 SET login_count = login_count + 1,
#                     last_login = ?
#                 WHERE username=?
#             """, (datetime.datetime.now(), username))
#             conn.commit()

#             user = User(username, role)
#             login_user(user)

#             #user = User(row[0], row[1])
#             #login_user(user)
#             conn.close()
#             return redirect(url_for("main.dashboard"))

#         conn.close()

#     return render_template("login.html")



@auth.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":
        username = request.form["username"]
        password = hash_pwd(request.form["password"])

        conn = sqlite3.connect(get_db())
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO users (username, password)
                VALUES (?, ?)
            """, (username, password))
            conn.commit()
            conn.close()
            return redirect(url_for("auth.login"))
        except:
            conn.close()
            return "User already exists"

    return render_template("register.html")


@auth.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))

