
# from flask_login import UserMixin
# import sqlite3

# class User(UserMixin):
#     def __init__(self, username, role="user"):
#         self.id = username
#         self.role = role

#     @staticmethod
#     def get(username):
#         conn = sqlite3.connect("users.db")
#         cur = conn.cursor()
#         cur.execute("SELECT username, role FROM users WHERE username=?", (username,))
#         row = cur.fetchone()
#         conn.close()
#         if row:
#             return User(row[0], row[1])
#         return None


import sqlite3
import os
from flask_login import UserMixin

class User(UserMixin):

    def __init__(self, username, role):
        self.id = username
        self.username = username
        self.role = role

    @staticmethod
    def get(username):
        BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        db_path = os.path.join(BASE_DIR, "..", "users.db")

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT username, role FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        conn.close()

        if row:
            return User(row[0], row[1])
        return None
