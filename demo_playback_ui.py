#!/usr/bin/env python3
"""
Demo script for the CheatGPT Session Playback UI
"""

import os
import sys
import webbrowser
import subprocess
import time
from pathlib import Path

def main():
    print("🎬 CheatGPT Session Playback UI Demo")
    print("=" * 50)
    
    # Check if recordings exist
    recordings_dir = Path("recordings")
    if not recordings_dir.exists() or not list(recordings_dir.glob("timeline_*.json")):
        print("❌ No recording sessions found!")
        print("\nTo create a recording session:")
        print("1. Run: python run_enhanced_detection.py")
        print("2. Wait for detection to start")
        print("3. Press 'q' to stop recording")
        print("4. Come back and run this demo again")
        return
    
    # List available sessions
    timeline_files = list(recordings_dir.glob("timeline_*.json"))
    print(f"✅ Found {len(timeline_files)} recording session(s)")
    
    for timeline_file in timeline_files:
        session_id = timeline_file.stem.replace("timeline_", "")
        print(f"  - {session_id}")
    
    print("\n🚀 Starting playback server...")
    print("📋 Features included:")
    print("  ✅ Interactive video player with YouTube-like controls")
    print("  ✅ Timeline bar with red hotspot markers") 
    print("  ✅ Click markers to jump to events")
    print("  ✅ Hover over markers to see event details")
    print("  ✅ Event overlay shows when suspicious activity occurs")
    print("  ✅ Events panel lists all detected activities")
    print("  ✅ Keyboard shortcuts (Space=play/pause, arrows=skip)")
    print("  ✅ Fullscreen support")
    
    try:
        # Import and run the Flask app
        from playback_server import app
        
        print(f"\n🌐 Server starting at: http://localhost:5000")
        print("🎯 Available endpoints:")
        print("  - / (Session list)")
        print("  - /playback/<session_id> (Video playback)")
        print("  - /api/sessions (JSON API)")
        
        # Auto-open browser after a delay
        def open_browser():
            time.sleep(2)
            webbrowser.open("http://localhost:5000")
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        print("\n🔥 Starting server... (Press Ctrl+C to stop)")
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Thanks for using CheatGPT Playback!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("Make sure Flask is installed: pip install flask")

if __name__ == "__main__":
    main()
