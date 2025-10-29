import requests

print("=" * 60)
print("TESTING THUMBNAIL GENERATION")
print("=" * 60)

session_id = 'single_1761065641'

try:
    # Test thumbnail endpoint
    thumbnail_url = f'http://localhost:5000/api/thumbnail/{session_id}'
    print(f"\nTesting: {thumbnail_url}")
    
    response = requests.get(thumbnail_url, timeout=10)
    
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    
    if response.status_code == 200:
        content_type = response.headers.get('Content-Type', '')
        if 'image' in content_type:
            print(f"Content-Length: {len(response.content)} bytes")
            print("✅ THUMBNAIL GENERATED SUCCESSFULLY!")
        else:
            print(f"⚠️  Unexpected content type: {content_type}")
    else:
        print(f"❌ Error: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("\n❌ Cannot connect to http://localhost:5000")
    print("Make sure Flask app is running")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "=" * 60)
