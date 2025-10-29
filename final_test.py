import requests
import json

print("=" * 60)
print("FINAL VERIFICATION - ALL UPLOADED VIDEOS")
print("=" * 60)

try:
    # Test 1: API endpoint
    response = requests.get('http://localhost:5000/api/sessions/list', timeout=10)
    if response.status_code == 200:
        data = response.json()
        uploaded = [s for s in data['sessions'] if s.get('session_type') == 'uploaded']
        print(f"\n✅ API returned {len(uploaded)} uploaded videos")
        
        for video in uploaded:
            print(f"\n   📹 {video['session_id']}")
            print(f"      Path: {video['video_path']}")
            print(f"      Title: {video['video_title']}")
            
            # Test playback
            playback_url = f"http://localhost:5000/playback/{video['session_id']}"
            pb_response = requests.head(playback_url, timeout=5)
            if pb_response.status_code == 200:
                print(f"      ✅ PLAYABLE (200)")
            else:
                print(f"      ❌ ERROR ({pb_response.status_code})")
    else:
        print(f"❌ API error: {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print("\n❌ Cannot connect to Flask app")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "=" * 60)
