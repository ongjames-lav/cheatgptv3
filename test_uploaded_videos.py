import sys
import os
import requests
import time

print("=" * 60)
print("TESTING UNIFIED DATABASE FOR UPLOADED VIDEOS")
print("=" * 60)

# Wait for server to be ready
time.sleep(2)

# Test the /api/sessions/list endpoint
try:
    print("\n1. Testing /api/sessions/list endpoint...")
    response = requests.get('http://localhost:5000/api/sessions/list')
    
    if response.status_code == 200:
        data = response.json()
        sessions = data.get('sessions', [])
        print(f"✅ Success! Found {len(sessions)} total sessions")
        
        # Count by type
        recorded = [s for s in sessions if s.get('session_type') == 'recorded']
        uploaded = [s for s in sessions if s.get('session_type') == 'uploaded']
        
        print(f"\n   📹 Recorded sessions: {len(recorded)}")
        print(f"   📤 Uploaded sessions: {len(uploaded)}")
        
        # Show first few of each type
        if recorded:
            print(f"\n   First 3 recorded sessions:")
            for s in recorded[:3]:
                print(f"      - {s['session_id']}: {s.get('duration', 0):.1f}s, {s.get('hotspot_count', 0)} events")
        
        if uploaded:
            print(f"\n   First 5 uploaded sessions:")
            for s in uploaded[:5]:
                print(f"      - {s['session_id']}: {s.get('video_title', 'N/A')}")
                print(f"        Duration: {s.get('duration', 0):.1f}s, Events: {s.get('hotspot_count', 0)}")
                print(f"        Video path: {s.get('video_path', 'N/A')}")
                print(f"        Status: {s.get('status', 'N/A')}")
        else:
            print(f"\n   ⚠️  No uploaded videos found in webapp database yet")
            print(f"       Upload a video to test the unified database")
    else:
        print(f"❌ Error: HTTP {response.status_code}")
        print(f"   Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Error: Could not connect to http://localhost:5000")
    print("   Make sure the Flask server is running with: cd web_app && python app.py")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("TESTING PLAYBACK AVAILABILITY")
print("=" * 60)

# Test if we can access playback for both types of videos
try:
    response = requests.get('http://localhost:5000/api/sessions/list')
    if response.status_code == 200:
        data = response.json()
        sessions = data.get('sessions', [])
        
        # Test recorded video playback
        recorded = [s for s in sessions if s.get('session_type') == 'recorded']
        if recorded:
            test_session = recorded[0]
            session_id = test_session['session_id']
            
            print(f"\n2. Testing playback for recorded video: {session_id}")
            playback_url = f"http://localhost:5000/playback/{session_id}"
            print(f"   Playback URL: {playback_url}")
            
            # Check if playback page loads
            response = requests.get(playback_url)
            if response.status_code == 200:
                print(f"   ✅ Recorded video playback page loads successfully")
            else:
                print(f"   ❌ Playback page error: HTTP {response.status_code}")
        
        # Test uploaded video playback
        uploaded = [s for s in sessions if s.get('session_type') == 'uploaded']
        if uploaded:
            test_session = uploaded[0]
            session_id = test_session['session_id']
            
            print(f"\n3. Testing playback for uploaded video: {session_id}")
            playback_url = f"http://localhost:5000/playback/{session_id}"
            print(f"   Playback URL: {playback_url}")
            print(f"   Video path: {test_session.get('video_path', 'N/A')}")
            
            # Check if playback page loads
            response = requests.get(playback_url)
            if response.status_code == 200:
                print(f"   ✅ Uploaded video playback page loads successfully")
            else:
                print(f"   ❌ Playback page error: HTTP {response.status_code}")
        else:
            print("\n3. No uploaded videos found in unified database to test playback")
            print("   Upload a video through the web interface to test")
            
except Exception as e:
    print(f"❌ Error testing playback: {e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✅ Changes implemented:")
print("   - Added 'session_type' column to webapp database")
print("   - Created 'create_uploaded_session()' method")
print("   - Updated 'process_video_async()' to use webapp database")
print("   - Modified '/api/sessions/list' to show all videos from one database")
print("   - Removed dependency on separate 'main_db' for uploaded videos")
print("\n📝 Next steps:")
print("   - Upload a video through the web interface")
print("   - Verify it appears in the session list")
print("   - Test that it's playable like recorded videos")
print("\n" + "=" * 60)
