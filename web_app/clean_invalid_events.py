"""
Remove invalid 'unknown' events with 0% confidence and 0:00 timestamp from the database
"""
import sqlite3

def clean_invalid_events():
    """Remove invalid events from hotspots table"""
    db_path = 'cheatgpt_sessions.db'
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count invalid events before deletion
    cursor.execute("""
        SELECT COUNT(*) FROM hotspots 
        WHERE event_type = 'unknown' 
        AND confidence = 0.0 
        AND timestamp_offset = 0.0
    """)
    invalid_count = cursor.fetchone()[0]
    
    print(f"Found {invalid_count} invalid 'unknown' events with 0% confidence and 0:00 timestamp")
    
    if invalid_count > 0:
        # Delete invalid events
        cursor.execute("""
            DELETE FROM hotspots 
            WHERE event_type = 'unknown' 
            AND confidence = 0.0 
            AND timestamp_offset = 0.0
        """)
        
        conn.commit()
        print(f"✅ Removed {invalid_count} invalid events from database")
    else:
        print("✓ No invalid events found")
    
    # Show remaining events summary
    cursor.execute("""
        SELECT event_type, COUNT(*) as count, AVG(confidence) as avg_conf
        FROM hotspots 
        GROUP BY event_type
        ORDER BY count DESC
    """)
    
    print("\n" + "="*60)
    print("Remaining events in database:")
    print("="*60)
    for row in cursor.fetchall():
        event_type, count, avg_conf = row
        print(f"{event_type:30s} Count: {count:4d}  Avg Confidence: {avg_conf:.1%}")
    
    conn.close()

if __name__ == '__main__':
    clean_invalid_events()
