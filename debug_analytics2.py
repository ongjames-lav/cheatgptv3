import sqlite3
import os

# Check both databases to understand the actual data
db_path = "cheatgpt.db"
sessions_db_path = "cheatgpt_sessions.db"

print("=== DEBUGGING ANALYTICS DATA ===")

# Check main database
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n1. SAMPLE SESSIONS DATA:")
    cursor.execute("SELECT session_id, start_timestamp, end_timestamp FROM sessions LIMIT 10")
    sessions = cursor.fetchall()
    for session in sessions:
        session_id, start_ts, end_ts = session
        duration = (end_ts - start_ts) if end_ts else 0
        print(f"   Session {session_id}: Start={start_ts}, End={end_ts}, Duration={duration:.1f}s")
    
    print("\n2. EVENTS COUNT PER SESSION:")
    cursor.execute("""
        SELECT s.session_id, s.start_timestamp, s.end_timestamp, COUNT(e.id) as event_count
        FROM sessions s 
        LEFT JOIN events e ON s.session_id = e.session_id 
        GROUP BY s.session_id, s.start_timestamp, s.end_timestamp
        LIMIT 10
    """)
    sessions_with_events = cursor.fetchall()
    for session in sessions_with_events:
        session_id, start_ts, end_ts, event_count = session
        duration = (end_ts - start_ts) if end_ts else 0
        events_per_min = (event_count / (duration / 60.0)) if duration > 0 else 0
        print(f"   Session {session_id}: {event_count} events in {duration:.1f}s = {events_per_min:.2f}/min")
    
    print("\n3. SAMPLE EVENTS DATA:")
    cursor.execute("""
        SELECT session_id, event_type, confidence, COUNT(*) as count
        FROM events 
        GROUP BY session_id, event_type, confidence
        LIMIT 10
    """)
    events = cursor.fetchall()
    for event in events:
        print(f"   Session {event[0]}: {event[1]} (confidence: {event[2]}) - {event[3]} occurrences")
    
    conn.close()
else:
    print(f"Main database {db_path} not found")

# Check how the analytics calculates duration in the actual code
print("\n4. CHECKING ANALYTICS CALCULATION IN HTML:")
html_path = "web_app/templates/analytics_home.html"
if os.path.exists(html_path):
    with open(html_path, 'r') as f:
        content = f.read()
        
    # Find the suspicious percentage calculation
    if "calculateSuspiciousPercentage" in content:
        print("   Found calculateSuspiciousPercentage function in HTML")
        # Extract the function
        start = content.find("function calculateSuspiciousPercentage")
        end = content.find("}", start) + 1
        if start != -1 and end != -1:
            func_code = content[start:end]
            print("   Function code:")
            print(func_code)
    else:
        print("   calculateSuspiciousPercentage function not found")
else:
    print(f"   HTML file {html_path} not found")

print("\n=== END DEBUG ===")
