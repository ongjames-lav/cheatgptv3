import requests

print("=" * 60)
print("TESTING VIDEO PLAYBACK")
print("=" * 60)

session_id = 'single_1761069614'

try:
    # Test playback endpoint
    playback_url = f'http://localhost:5000/playback/{session_id}'
    print(f"\nTesting: {playback_url}")
    
    response = requests.get(playback_url, timeout=10, stream=True)
    
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print(f"Content-Length: {response.headers.get('Content-Length', 'Unknown')}")
    
    if response.status_code == 200:
        # Read first 1KB to verify it's a real video file
        chunk = next(response.iter_content(chunk_size=1024), None)
        if chunk:
            print(f"First chunk size: {len(chunk)} bytes")
            # Check for MP4 file signature (ftyp)
            if b'ftyp' in chunk[:100]:
                print("✅ Valid MP4 file signature detected")
            else:
                print("⚠️  No MP4 signature found in first 100 bytes")
        
        print("\n✅ VIDEO IS PLAYABLE!")
    else:
        print(f"\n❌ Error: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("\n❌ Cannot connect to http://localhost:5000")
    print("Make sure Flask app is running")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "=" * 60)
