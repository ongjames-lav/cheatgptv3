import sqlite3

db_path = 'web_app/cheatgpt_sessions.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if test session exists
cursor.execute("SELECT * FROM sessions WHERE session_id = 'single_1761065641'")
session = cursor.fetchone()

if session:
    print("✅ Session found in database!")
    print(f"   ID: {session[1]}")
    print(f"   Path: {session[2]}")
else:
    print("❌ Session NOT found in database")

conn.close()
