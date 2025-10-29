import sqlite3
import json

db_path = 'web_app/cheatgpt_sessions.db'

print("=" * 60)
print("DIRECT DATABASE QUERY")
print("=" * 60)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Check total sessions
cursor.execute("SELECT COUNT(*) as total FROM sessions")
total = cursor.fetchone()['total']
print(f"\nTotal sessions in database: {total}")

# Count by type
cursor.execute("SELECT session_type, COUNT(*) as count FROM sessions GROUP BY session_type")
types = cursor.fetchall()
print(f"\nBreakdown by session_type:")
for row in types:
    print(f"  {row['session_type'] or 'NULL'}: {row['count']}")

# Get the test session
cursor.execute("SELECT * FROM sessions WHERE session_id = 'single_1761065641'")
test_session = cursor.fetchone()
if test_session:
    print(f"\nTest session single_1761065641:")
    print(f"  video_path: {test_session['video_path']}")
    print(f"  video_title: {test_session['video_title']}")
    print(f"  session_type: {test_session['session_type']}")
    print(f"  status: {test_session['status']}")
else:
    print(f"\nTest session single_1761065641 NOT found")

# Check for duplicates
cursor.execute("""
    SELECT session_id, COUNT(*) as count 
    FROM sessions 
    GROUP BY session_id 
    HAVING count > 1
""")
duplicates = cursor.fetchall()
if duplicates:
    print(f"\nDuplicate session_ids found:")
    for dup in duplicates:
        print(f"  {dup['session_id']}: {dup['count']} entries")
else:
    print(f"\nNo duplicate session_ids found")

conn.close()

print("\n" + "=" * 60)
