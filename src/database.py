import sqlite3
import os
from datetime import datetime

DB_PATH = "tracking_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Table for general tracking sessions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            source TEXT
        )
    ''')
    # Table for person detection logs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT,
            person_count INTEGER,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    ''')
    conn.commit()
    conn.close()

def start_session(source="live"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO sessions (start_time, source) VALUES (?, ?)", (start_time, source))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def log_detection(session_id, person_count):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO detections (session_id, timestamp, person_count) VALUES (?, ?, ?)", 
                   (session_id, timestamp, person_count))
    conn.commit()
    conn.close()

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, person_count FROM detections ORDER BY timestamp DESC LIMIT 100")
    data = cursor.fetchall()
    conn.close()
    return [{"timestamp": d[0], "count": d[1]} for d in data]

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
