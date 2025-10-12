#!/usr/bin/env python3
"""
Test Script for Uploaded Video Integration
Tests the complete workflow: upload -> process -> database storage -> analytics
"""

import os
import sys
import json
import logging
import requests
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "cheatgpt"))

from cheatgpt.db.db_manager import DBManager

class UploadedVideoIntegrationTest:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.db = DBManager()
        
    def test_database_integration(self):
        """Test database methods for uploaded videos"""
        logger.info("Testing database integration for uploaded videos...")
        
        # Test session creation
        session_id = f"test_upload_{int(time.time())}"
        video_title = "Test Uploaded Video"
        
        # Create uploaded video session
        success = self.db.create_uploaded_video_session(
            session_id=session_id,
            original_filename=f"{video_title}.mp4",
            video_path="/test/original.mp4",
            video_metadata={'title': video_title, 'duration': 120.5}
        )
        
        if success:
            logger.info(f"✅ Created uploaded video session: {session_id}")
        else:
            logger.error("❌ Failed to create uploaded video session")
            return False
        
        # Test updating processed results
        processed_results = {
            'detection_stats': {
                'total_frames': 3600,
                'phone_detections': 45,
                'gesture_detections': 12,
                'head_turn_detections': 23
            },
            'processing_time': 45.6,
            'model_version': '1.0.0',
            'processed_video_path': "/test/processed.mp4"
        }
        
        success = self.db.update_processed_video_results(
            session_id=session_id,
            result=processed_results
        )
        
        if success:
            logger.info("✅ Updated processed video results")
        else:
            logger.error("❌ Failed to update processed video results")
            return False
        
        # Test storing events
        test_events = [
            {
                'event_type': 'phone_detection',
                'confidence': 0.89,
                'timestamp_seconds': 45.2,
                'bbox': [100, 150, 200, 250],
                'severity': 'red'
            },
            {
                'event_type': 'head_turning',
                'confidence': 0.76,
                'timestamp_seconds': 78.5,
                'angle': 45.0,
                'severity': 'orange'
            }
        ]
        
        success = self.db.store_uploaded_video_events(session_id, test_events)
        
        if success:
            logger.info(f"✅ Stored {len(test_events)} events")
        else:
            logger.error("❌ Failed to store events")
            return False
        
        # Test retrieving session
        session = self.db.get_session_info(session_id)
        if session:
            logger.info(f"✅ Retrieved session: {session['session_id']}")
            logger.info(f"   Filename: {session.get('cam_id')}")
            logger.info(f"   Status: {session.get('status')}")
        else:
            logger.error("❌ Failed to retrieve session")
            return False
        
        # Test retrieving events
        events = self.db.get_session_events(session_id)
        if events:
            logger.info(f"✅ Retrieved {len(events)} events")
            for event in events:
                timestamp = event.get('timestamp', 0)
                event_type = event.get('event_type', 'unknown')
                logger.info(f"   Event: {event_type} at {timestamp}s")
        else:
            logger.error("❌ Failed to retrieve events")
            return False
        
        # Test getting sessions list (should include uploaded videos)
        sessions = self.db.get_all_sessions()
        uploaded_sessions = [s for s in sessions if s.get('session_type') == 'uploaded']
        logger.info(f"✅ Found {len(uploaded_sessions)} uploaded sessions in database")
        
        # Debug: Show what statuses we actually have
        all_statuses = set([s.get('status') for s in sessions])
        logger.info(f"   Available statuses: {all_statuses}")
        
        # Cleanup test session
        try:
            with self.db._lock:
                cursor = self.db.conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM hotspots WHERE session_id = ?", (session_id,))
                self.db.conn.commit()
            logger.info("✅ Cleaned up test session")
        except Exception as e:
            logger.warning(f"Failed to cleanup test session: {e}")
        
        return True
    
    def test_api_endpoints(self):
        """Test API endpoints for uploaded videos"""
        logger.info("Testing API endpoints...")
        
        try:
            # Test sessions list endpoint
            response = requests.get(f"{self.server_url}/api/sessions/list", timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Sessions list API working - found {data.get('total_count', 0)} sessions")
            else:
                logger.error(f"❌ Sessions list API failed: {response.status_code}")
                return False
            
            # Test uploaded videos endpoint
            response = requests.get(f"{self.server_url}/api/sessions/uploaded", timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Uploaded videos API working - found {data.get('total_count', 0)} videos")
            else:
                logger.error(f"❌ Uploaded videos API failed: {response.status_code}")
                return False
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API test failed - server not running? {e}")
            return False
    
    def test_file_structure(self):
        """Test that required directories and files exist"""
        logger.info("Testing file structure...")
        
        required_files = [
            "web_app/app.py",
            "web_app/db_manager.py",
            "cheatgpt/video_processor.py",
            "cheatgpt/db/db_manager.py"
        ]
        
        for file_path in required_files:
            full_path = project_root / file_path
            if full_path.exists():
                logger.info(f"✅ {file_path} exists")
            else:
                logger.error(f"❌ {file_path} missing")
                return False
        
        # Check if results directory exists
        results_dir = project_root / "results"
        if not results_dir.exists():
            results_dir.mkdir(exist_ok=True)
            logger.info("✅ Created results directory")
        else:
            logger.info("✅ Results directory exists")
        
        return True
    
    def run_complete_test(self):
        """Run all integration tests"""
        logger.info("Starting Uploaded Video Integration Test...")
        logger.info("=" * 60)
        
        tests = [
            ("File Structure", self.test_file_structure),
            ("Database Integration", self.test_database_integration),
            ("API Endpoints", self.test_api_endpoints)
        ]
        
        results = {}
        for test_name, test_func in tests:
            logger.info(f"\n🧪 Running {test_name} Test...")
            try:
                results[test_name] = test_func()
            except Exception as e:
                logger.error(f"❌ {test_name} test failed with exception: {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS SUMMARY:")
        logger.info("=" * 60)
        
        all_passed = True
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"{test_name:20} : {status}")
            if not passed:
                all_passed = False
        
        logger.info("=" * 60)
        if all_passed:
            logger.info("🎉 ALL TESTS PASSED! Uploaded video integration is working correctly.")
        else:
            logger.error("⚠️  SOME TESTS FAILED! Please check the issues above.")
        
        return all_passed

if __name__ == "__main__":
    # Run the integration test
    tester = UploadedVideoIntegrationTest()
    success = tester.run_complete_test()
    
    if success:
        print("\n✅ Integration test completed successfully!")
        print("📋 Uploaded videos should now work with:")
        print("   • Analytics pages")
        print("   • Video playback with hotspots")
        print("   • Report generation")
        print("   • Event tracking")
        print("   • Database storage")
    else:
        print("\n❌ Integration test failed!")
        print("🔧 Please fix the issues and run the test again.")
    
    sys.exit(0 if success else 1)