import requests
import json

print("=" * 60)
print("TESTING /api/sessions/list ENDPOINT DIRECTLY")
print("=" * 60)

try:
    response = requests.get('http://localhost:5000/api/sessions/list', timeout=10)
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nAPI Response:")
        print(f"  Success: {data.get('success')}")
        print(f"  Total Count: {data.get('total_count')}")
        
        sessions = data.get('sessions', [])
        
        # Count by type
        recorded = [s for s in sessions if s.get('session_type') != 'uploaded']
        uploaded = [s for s in sessions if s.get('session_type') == 'uploaded']
        
        print(f"\nSession Breakdown:")
        print(f"  Recorded: {len(recorded)}")
        print(f"  Uploaded: {len(uploaded)}")
        
        print(f"\nFirst 5 uploaded sessions:")
        for i, s in enumerate(uploaded[:5]):
            print(f"  {i+1}. {s['session_id']} - {s.get('video_title', 'N/A')}")
            print(f"     Video Path: {s.get('video_path')}")
            print(f"     Status: {s.get('status')}")
        
        # Check for the specific session we've been testing
        test_session = next((s for s in sessions if s['session_id'] == 'single_1761065641'), None)
        if test_session:
            print(f"\nFound test session single_1761065641:")
            print(f"  Type: {test_session.get('session_type')}")
            print(f"  Video Path: {test_session.get('video_path')}")
            print(f"  Status: {test_session.get('status')}")
        else:
            print(f"\nTest session single_1761065641 NOT found in API response!")
    else:
        print(f"Error: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("\nERROR: Cannot connect to http://localhost:5000")
    print("Make sure the Flask app is running with: python web_app/app.py")
except Exception as e:
    print(f"\nError: {e}")

print("\n" + "=" * 60)
