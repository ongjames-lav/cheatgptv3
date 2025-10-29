import requests
import json

print("=" * 60)
print("CHECKING IF single_1761069614 IS IN API RESPONSE")
print("=" * 60)

try:
    response = requests.get('http://localhost:5000/api/sessions/list', timeout=10)
    if response.status_code == 200:
        data = response.json()
        sessions = data.get('sessions', [])
        
        print(f"\nTotal sessions in API: {len(sessions)}")
        
        # Search for the specific session
        target_session = None
        for s in sessions:
            if s['session_id'] == 'single_1761069614':
                target_session = s
                break
        
        if target_session:
            print(f"\n✅ FOUND in API response!")
            print(f"   Session ID: {target_session['session_id']}")
            print(f"   Video Path: {target_session['video_path']}")
            print(f"   Title: {target_session['video_title']}")
            print(f"   Type: {target_session.get('session_type', 'N/A')}")
            print(f"   Status: {target_session.get('status', 'N/A')}")
        else:
            print(f"\n❌ NOT found in API response!")
            print(f"\nAll session IDs in response:")
            for i, s in enumerate(sessions[-10:]):  # Show last 10
                print(f"  {i+1}. {s['session_id']}")
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
