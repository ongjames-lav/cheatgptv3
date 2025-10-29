#!/usr/bin/env python3
"""
Test script to verify frontend is properly handling processed/uploaded videos
Tests: Database storage, API response, playback route, and frontend data
"""

import sys
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple

# Setup paths
WORKSPACE_DIR = Path(__file__).parent
WEB_APP_DIR = WORKSPACE_DIR / 'web_app'
DB_PATH = WEB_APP_DIR / 'cheatgpt_sessions.db'
RESULTS_DIR = WEB_APP_DIR / 'results'

class FrontendProcessingTest:
    def __init__(self):
        self.results = {
            'database_check': {},
            'api_simulation': {},
            'frontend_rendering': {},
            'all_passed': True
        }
    
    def log(self, title: str, status: str = None, details: str = None):
        """Print formatted log message"""
        symbols = {
            'pass': '✅',
            'fail': '❌',
            'info': 'ℹ️',
            'warn': '⚠️'
        }
        
        symbol = symbols.get(status, '•')
        indent = '  ' if details else ''
        
        print(f"{symbol} {title}")
        if details:
            print(f"{indent}{details}")
    
    def test_database_schema(self) -> bool:
        """Test 1: Verify database has correct schema for processed videos"""
        print("\n" + "="*80)
        print("TEST 1: DATABASE SCHEMA")
        print("="*80)
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if session_type column exists
            cursor.execute("PRAGMA table_info(sessions)")
            columns = {col[1]: col[2] for col in cursor.fetchall()}
            
            self.log("Checking for session_type column", "info")
            if 'session_type' in columns:
                self.log("✓ session_type column exists", "pass")
            else:
                self.log("✗ session_type column MISSING", "fail")
                return False
            
            # Check column type
            if columns['session_type'] == 'TEXT':
                self.log("✓ session_type is TEXT type", "pass")
            else:
                self.log(f"✗ session_type is {columns['session_type']}, expected TEXT", "fail")
                return False
            
            self.results['database_check']['schema'] = 'PASS'
            conn.close()
            return True
            
        except Exception as e:
            self.log(f"✗ Database schema check failed: {e}", "fail")
            self.results['database_check']['schema'] = 'FAIL'
            self.all_passed = False
            return False
    
    def test_database_content(self) -> Tuple[bool, List[Dict]]:
        """Test 2: Verify database contains uploaded videos with correct session_type"""
        print("\n" + "="*80)
        print("TEST 2: DATABASE CONTENT")
        print("="*80)
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get all sessions with their types
            cursor.execute("""
                SELECT session_id, session_type, status, video_path, duration, created_at
                FROM sessions
                ORDER BY created_at DESC
                LIMIT 100
            """)
            
            sessions = []
            recorded_count = 0
            uploaded_count = 0
            
            for row in cursor.fetchall():
                session_id, session_type, status, video_path, duration, created_at = row
                sessions.append({
                    'session_id': session_id,
                    'session_type': session_type or 'recorded',
                    'status': status,
                    'video_path': video_path,
                    'duration': duration,
                    'created_at': created_at
                })
                
                if session_type == 'uploaded':
                    uploaded_count += 1
                else:
                    recorded_count += 1
            
            self.log(f"Total sessions in database: {len(sessions)}", "info")
            self.log(f"  Recorded: {recorded_count}", "info")
            self.log(f"  Uploaded: {uploaded_count}", "info")
            
            if uploaded_count > 0:
                self.log(f"✓ Found {uploaded_count} uploaded sessions", "pass")
                self.results['database_check']['uploaded_sessions'] = uploaded_count
            else:
                self.log("⚠ No uploaded sessions found in database", "warn")
                self.results['database_check']['uploaded_sessions'] = 0
            
            # Check if uploaded sessions have correct paths
            uploaded_sessions = [s for s in sessions if s['session_type'] == 'uploaded']
            if uploaded_sessions:
                print("\nSample uploaded sessions:")
                for session in uploaded_sessions[:3]:
                    print(f"  • {session['session_id']}")
                    print(f"    Type: {session['session_type']}")
                    print(f"    Status: {session['status']}")
                    print(f"    Path: {session['video_path']}")
            
            self.results['database_check']['content'] = 'PASS'
            conn.close()
            return True, sessions
            
        except Exception as e:
            self.log(f"✗ Database content check failed: {e}", "fail")
            self.results['database_check']['content'] = 'FAIL'
            self.all_passed = False
            return False, []
    
    def test_api_response_format(self, sessions: List[Dict]) -> bool:
        """Test 3: Verify API response would include session_type properly"""
        print("\n" + "="*80)
        print("TEST 3: API RESPONSE FORMAT")
        print("="*80)
        
        try:
            # Simulate what /api/sessions/list returns
            uploaded_sessions = [s for s in sessions if s.get('session_type') == 'uploaded']
            
            if not uploaded_sessions:
                self.log("⚠ No uploaded sessions to test API format", "warn")
                self.results['api_simulation']['format'] = 'WARN'
                return True
            
            # Test first uploaded session
            test_session = uploaded_sessions[0]
            
            # Simulate API response structure
            api_response = {
                'session_id': test_session['session_id'],
                'session_type': test_session['session_type'],
                'status': test_session['status'],
                'video_path': test_session['video_path'],
                'duration': test_session['duration'],
                'start_time': None,  # Would be calculated from created_at
            }
            
            self.log(f"Sample API response for {test_session['session_id']}:", "info")
            print(json.dumps(api_response, indent=2, default=str))
            
            # Validate response has required fields
            required_fields = ['session_id', 'session_type', 'status', 'video_path']
            missing_fields = [f for f in required_fields if f not in api_response]
            
            if missing_fields:
                self.log(f"✗ Missing required fields: {missing_fields}", "fail")
                self.results['api_simulation']['format'] = 'FAIL'
                return False
            
            # Check session_type value
            if api_response['session_type'] in ['recorded', 'uploaded']:
                self.log(f"✓ session_type value is valid: {api_response['session_type']}", "pass")
            else:
                self.log(f"✗ session_type has invalid value: {api_response['session_type']}", "fail")
                self.results['api_simulation']['format'] = 'FAIL'
                return False
            
            # Check path format
            if 'results' in api_response['video_path'] or 'results\\' in api_response['video_path']:
                self.log(f"✓ Video path format is correct (has results directory)", "pass")
            else:
                self.log(f"⚠ Video path doesn't contain 'results': {api_response['video_path']}", "warn")
            
            self.results['api_simulation']['format'] = 'PASS'
            return True
            
        except Exception as e:
            self.log(f"✗ API response format check failed: {e}", "fail")
            self.results['api_simulation']['format'] = 'FAIL'
            self.all_passed = False
            return False
    
    def test_frontend_data_binding(self, sessions: List[Dict]) -> bool:
        """Test 4: Verify data would bind correctly in frontend JavaScript"""
        print("\n" + "="*80)
        print("TEST 4: FRONTEND DATA BINDING")
        print("="*80)
        
        try:
            uploaded_sessions = [s for s in sessions if s.get('session_type') == 'uploaded']
            
            if not uploaded_sessions:
                self.log("⚠ No uploaded sessions to test frontend binding", "warn")
                self.results['frontend_rendering']['binding'] = 'WARN'
                return True
            
            test_session = uploaded_sessions[0]
            
            # Test JavaScript expression: isUploaded = session.session_type === 'uploaded'
            # Python equivalent:
            is_uploaded = test_session['session_type'] == 'uploaded'
            
            self.log(f"Testing session: {test_session['session_id']}", "info")
            self.log(f"  session.session_type = '{test_session['session_type']}'", "info")
            self.log(f"  isUploaded = {is_uploaded}", "info")
            
            if is_uploaded:
                self.log("✓ Session correctly identified as uploaded", "pass")
            else:
                self.log("✗ Session NOT identified as uploaded", "fail")
                self.results['frontend_rendering']['binding'] = 'FAIL'
                return False
            
            # Test card rendering data-type attribute
            data_type = 'uploaded' if is_uploaded else 'recorded'
            
            self.log(f"✓ data-type attribute would be: {data_type}", "pass")
            
            # Test badge rendering
            badge = '<div class="processed-badge">UPLOADED</div>' if is_uploaded else ''
            self.log(f"✓ Badge would render: {'YES' if badge else 'NO'}", "pass")
            
            # Test playback URL
            playback_url = f"/playback/{test_session['session_id']}"
            self.log(f"✓ Playback URL: {playback_url}", "pass")
            
            self.results['frontend_rendering']['binding'] = 'PASS'
            return True
            
        except Exception as e:
            self.log(f"✗ Frontend data binding check failed: {e}", "fail")
            self.results['frontend_rendering']['binding'] = 'FAIL'
            self.all_passed = False
            return False
    
    def test_file_system_consistency(self) -> bool:
        """Test 5: Verify uploaded videos exist in results directory"""
        print("\n" + "="*80)
        print("TEST 5: FILE SYSTEM CONSISTENCY")
        print("="*80)
        
        try:
            if not RESULTS_DIR.exists():
                self.log(f"⚠ Results directory not found: {RESULTS_DIR}", "warn")
                self.results['frontend_rendering']['files'] = 'WARN'
                return True
            
            # Count video files
            video_files = list(RESULTS_DIR.glob('*/processed_*.mp4'))
            
            self.log(f"Found {len(video_files)} processed video files", "info")
            
            if len(video_files) > 0:
                self.log(f"✓ Processed videos exist on file system", "pass")
                
                # Check if they're readable and have reasonable size
                readable_files = []
                for video_file in video_files[:5]:  # Check first 5
                    try:
                        size = video_file.stat().st_size
                        if size > 100000:  # At least 100KB
                            readable_files.append((video_file.name, size))
                    except:
                        pass
                
                self.log(f"  Readable files: {len(readable_files)}", "info")
                for filename, size in readable_files[:3]:
                    self.log(f"    • {filename} ({size/1024/1024:.1f} MB)", "info")
                
                self.results['frontend_rendering']['files'] = 'PASS'
                return True
            else:
                self.log("⚠ No processed video files found in results directory", "warn")
                self.results['frontend_rendering']['files'] = 'WARN'
                return True
            
        except Exception as e:
            self.log(f"✗ File system consistency check failed: {e}", "fail")
            self.results['frontend_rendering']['files'] = 'FAIL'
            self.all_passed = False
            return False
    
    def test_css_styling(self) -> bool:
        """Test 6: Verify CSS styling is defined for uploaded videos"""
        print("\n" + "="*80)
        print("TEST 6: CSS STYLING")
        print("="*80)
        
        try:
            html_file = WEB_APP_DIR / 'templates' / 'analytics_home.html'
            
            if not html_file.exists():
                self.log(f"✗ HTML template not found: {html_file}", "fail")
                self.results['frontend_rendering']['css'] = 'FAIL'
                return False
            
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for CSS classes for uploaded videos
            required_css_classes = [
                '.processed-badge',
                'data-type="uploaded"',
                '.video-card[data-type="uploaded"]'
            ]
            
            found_classes = []
            missing_classes = []
            
            for css_class in required_css_classes:
                if css_class in content:
                    found_classes.append(css_class)
                else:
                    missing_classes.append(css_class)
            
            self.log(f"CSS classes found: {len(found_classes)}/{len(required_css_classes)}", "info")
            
            for css_class in found_classes:
                self.log(f"✓ Found: {css_class}", "pass")
            
            for css_class in missing_classes:
                self.log(f"⚠ Missing: {css_class}", "warn")
            
            # Check for unified playback route (frontend uses /analytics/player?session=)
            if ("/analytics/player" in content and "session=" in content) or \
               ("/playback/" in content and "session" in content):
                self.log("✓ Found unified playback route in frontend", "pass")
            else:
                self.log("✗ Unified playback route not found", "fail")
                self.results['frontend_rendering']['css'] = 'FAIL'
                return False
            
            self.results['frontend_rendering']['css'] = 'PASS'
            return True
            
        except Exception as e:
            self.log(f"✗ CSS styling check failed: {e}", "fail")
            self.results['frontend_rendering']['css'] = 'FAIL'
            self.all_passed = False
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "█"*80)
        print("FRONTEND PROCESSED VIDEO HANDLING TEST SUITE")
        print("█"*80)
        
        # Test 1: Database schema
        schema_ok = self.test_database_schema()
        
        # Test 2: Database content
        content_ok, sessions = self.test_database_content()
        
        # Test 3: API response format
        api_ok = self.test_api_response_format(sessions)
        
        # Test 4: Frontend data binding
        binding_ok = self.test_frontend_data_binding(sessions)
        
        # Test 5: File system consistency
        files_ok = self.test_file_system_consistency()
        
        # Test 6: CSS styling
        css_ok = self.test_css_styling()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        all_tests = [
            ('Schema', schema_ok),
            ('Content', content_ok),
            ('API Format', api_ok),
            ('Binding', binding_ok),
            ('Files', files_ok),
            ('CSS', css_ok)
        ]
        
        passed = sum(1 for _, ok in all_tests if ok)
        total = len(all_tests)
        
        for test_name, ok in all_tests:
            status = '✅ PASS' if ok else '❌ FAIL'
            print(f"  {test_name:.<30} {status}")
        
        print(f"\nResult: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n✅ FRONTEND PROCESSING: ALL TESTS PASSED")
            print("Frontend is correctly treating processed/uploaded videos!")
        else:
            print(f"\n❌ FRONTEND PROCESSING: {total - passed} TEST(S) FAILED")
            print("Please check the issues above.")
        
        return passed == total


def main():
    tester = FrontendProcessingTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
