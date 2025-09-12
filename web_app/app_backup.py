"""
CheatGPT Web Application
Main Flask app with SocketIO for real-time monitoring and session management
"""

import os
import sys
import time
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import threading
import queue

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_app.db_manager import db
from web_app.reports.session_report import SessionReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import CheatGPT detection engine
try:
    import sys
    import os
    
    # Add the parent directory to the path for cheatgpt imports
    cheatgpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if cheatgpt_path not in sys.path:
        sys.path.insert(0, cheatgpt_path)
    
    from cheatgpt.engine import Engine
    logger.info("✅ CheatGPT detection engine imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import CheatGPT detection engine: {e}")
    Engine = None

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'cheatgpt_web_secret_key_2024'

# Initialize SocketIO with CORS enabled and polling-only transport
socketio = SocketIO(app, cors_allowed_origins="*", transports=['polling'])

# Global state variables
detection_engine = None
current_session_id = None
session_start_time = None
camera_thread = None
recording_active = False
video_feed_active = False
frame_queue = queue.Queue(maxsize=30)

# Shared frame buffer for video streaming
latest_frame = None
frame_lock = threading.Lock()

# Additional global variables for enhanced UI
camera_active = False
recent_events = []  # Store recent suspicious events
session_stats = {
    'frame_count': 0,
    'fps': 0,
    'hotspot_count': 0,
    'elapsed_time': 0
}

# Initialize detection engine
def initialize_detection_engine():
    """Initialize the CheatGPT detection engine"""
    global detection_engine
    
    if Engine is None:
        logger.error("❌ CheatGPT Engine not available")
        return False
    
    try:
        detection_engine = Engine()
        logger.info("✅ Detection engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize detection engine: {e}")
        return False

# Initialize the engine at startup
initialize_detection_engine()

# Directories
BASE_DIR = Path(__file__).parent.parent
VIDEOS_DIR = BASE_DIR / "videos"
RECORDINGS_DIR = BASE_DIR / "recordings"
STATIC_DIR = Path(__file__).parent / "static"

# Ensure directories exist
VIDEOS_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)

@app.route('/')
def index():
    """Main dashboard - live monitoring page"""
    return render_template('live_enhanced.html')

@app.route('/recordings')
def recordings():
    """Recordings management page"""
    return render_template('recordings.html')

@app.route('/session/<session_id>')
def session_detail(session_id):
    """Session detail page"""
    return render_template('session_detail.html', session_id=session_id)

@app.route('/help')
def help_page():
    """Help and documentation page"""
    return render_template('help.html')

# API Routes

@app.route('/api/status')
def api_status():
    """Get current system status"""
    global camera_active, recording_active, current_session_id, session_start_time, session_stats
    
    # Calculate elapsed time if session is active
    elapsed_time = 0
    if session_start_time:
        elapsed_time = time.time() - session_start_time
        session_stats['elapsed_time'] = elapsed_time
    
    return jsonify({
        'camera_active': camera_active,
        'recording_active': recording_active,
        'current_session_id': current_session_id,
        'session_start_time': session_start_time,
        'elapsed_time': elapsed_time,
        'detection_engine_available': detection_engine is not None,
        'frame_count': session_stats['frame_count'],
        'fps': session_stats['fps'],
        'hotspot_count': session_stats['hotspot_count'],
        'recent_events': recent_events[-5:] if recent_events else []  # Last 5 events
    })

@app.route('/api/sessions')
def api_sessions():
    """Get list of recorded sessions"""
    try:
        sessions = db.get_sessions()
        return jsonify({
            'success': True,
            'sessions': sessions
        })
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>')
def api_session_detail(session_id):
    """Get detailed session information"""
    try:
        session = db.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify({
            'success': True,
            'session': session
        })
    except Exception as e:
        logger.error(f"Error fetching session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>/events')
def api_session_events(session_id):
    """Get events for a specific session"""
    try:
        events = db.get_session_events(session_id)
        return jsonify({
            'success': True,
            'events': events
        })
    except Exception as e:
        logger.error(f"Error fetching events for session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>/report')
def api_session_report(session_id):
    """Generate and download session report"""
    try:
        session = db.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Generate report
        report_generator = SessionReportGenerator()
        report_path = report_generator.generate_report(session_id)
        
        if not report_path or not os.path.exists(report_path):
            return jsonify({'error': 'Failed to generate report'}), 500
        
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"session_{session_id}_report.pdf"
        )
    except Exception as e:
        logger.error(f"Error generating report for session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

# Camera Control Routes

@app.route('/camera/start', methods=['POST'])
def start_camera():
    """Start camera and begin recording session"""
    global camera_active, recording_active, current_session_id, session_start_time, camera_thread, session_stats
    
    try:
        if camera_active:
            return jsonify({'error': 'Camera already active'}), 400
        
        # Generate session ID
        current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        session_start_time = time.time()
        
        # Initialize session stats
        session_stats = {
            'frame_count': 0,
            'fps': 0,
            'hotspot_count': 0,
            'elapsed_time': 0
        }
        
        # Note: Detection engine will be used via process_frame in camera worker
        # No need to start a separate session here to avoid camera conflicts
        
        # Start camera worker thread for video feed and frame processing
        camera_active = True
        recording_active = True
        camera_thread = threading.Thread(target=camera_worker, daemon=True)
        camera_thread.start()
        
        # Create database session
        db.create_session(current_session_id, session_start_time, {
            'user_agent': request.headers.get('User-Agent', ''),
            'ip_address': request.remote_addr,
            'started_via': 'web_camera_control'
        })
        
        # Emit to all connected clients
        socketio.emit('camera_started', {
            'session_id': current_session_id,
            'start_time': session_start_time
        })
        
        logger.info(f"🎥 Camera started - Session: {current_session_id}")
        
        return jsonify({
            'success': True,
            'session_id': current_session_id,
            'start_time': session_start_time,
            'message': 'Camera started successfully'
        })
        
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera and end recording session"""
    global camera_active, recording_active, current_session_id, session_start_time, session_stats
    
    try:
        # Check if any recording is active
        if not camera_active and not recording_active:
            return jsonify({'error': 'Camera not active'}), 400
        
        # Stop all processes
        logger.info("🛑 Stopping camera...")
        camera_active = False
        recording_active = False
        video_feed_active = False  # Stop video feed too
        
        # Wait for camera worker to stop
        import time
        time.sleep(1.0)
        logger.info("🛑 Camera stopped")
        
        # Calculate final stats
        end_time = time.time()
        duration = end_time - session_start_time if session_start_time else 0
        
        # Update database
        if current_session_id:
            db.end_session(current_session_id, end_time, session_stats['frame_count'])
        
        # Emit to clients
        socketio.emit('camera_stopped', {
            'session_id': current_session_id,
            'duration': duration,
            'frame_count': session_stats['frame_count']
        })
        
        logger.info(f"🛑 Camera stopped - Session: {current_session_id}, Duration: {duration:.1f}s")
        
        # Reset session variables
        session_id = current_session_id
        current_session_id = None
        session_start_time = None
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'duration': duration,
            'frame_count': session_stats['frame_count'],
            'message': 'Camera stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"Error stopping camera: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/camera/screenshot', methods=['POST'])
def take_screenshot():
    """Take screenshot with overlay"""
    global latest_frame
    
    try:
        if not camera_active:
            return jsonify({'error': 'Camera not active'}), 400
        
        with frame_lock:
            if latest_frame is None:
                return jsonify({'error': 'No frame available'}), 400
            
            # Generate screenshot filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_name = f"screenshot_{timestamp}.jpg"
            screenshot_path = RECORDINGS_DIR / screenshot_name
            
            # Save screenshot
            cv2.imwrite(str(screenshot_path), latest_frame)
            
            logger.info(f"📸 Screenshot saved: {screenshot_name}")
            
            return jsonify({
                'success': True,
                'filename': screenshot_name,
                'path': str(screenshot_path),
                'message': 'Screenshot saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error taking screenshot: {e}")
        return jsonify({'error': str(e)}), 500

# Video streaming route
@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            try:
                if not camera_active:
                    break
                    
                # Get frame from queue
                if not frame_queue.empty():
                    frame = frame_queue.get()
                    
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    time.sleep(0.033)  # ~30 FPS
                    
            except Exception as e:
                logger.error(f"Video feed error: {e}")
                break
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def camera_worker():
    """Background worker for camera feed and frame processing"""
    global camera_active, session_stats, recent_events
    
    cap = None
    try:
        logger.info("🎥 Camera worker started")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frame_count = 0
        last_fps_time = time.time()
        
        while camera_active:
            # Check if we should stop (additional safety check)
            if not camera_active:
                logger.info("🛑 Camera worker detected camera_active = False, breaking loop")
                break
                
            current_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                time.sleep(0.1)
                continue
                
            frame_count += 1
            session_stats['frame_count'] = frame_count
            
            # Process frame through detection engine if available
            display_frame = frame.copy()
            if detection_engine and hasattr(detection_engine, 'process_frame'):
                try:
                    overlay_frame, events = detection_engine.process_frame(frame, "webcam", current_time)
                    if overlay_frame is not None:
                        display_frame = overlay_frame
                        # Update recent events if any detections occurred
                        if events:
                            recent_events.extend(events)
                            # Keep only last 10 events
                            recent_events = recent_events[-10:]
                except Exception as e:
                    logger.debug(f"Error processing frame through engine: {e}")
            
            # Store latest frame for video streaming
            with frame_lock:
                global latest_frame
                latest_frame = display_frame.copy()
            
            # Add frame to queue for video feed
            if not frame_queue.full():
                frame_queue.put(display_frame.copy())
            
            # Calculate FPS
            if current_time - last_fps_time >= 1.0:
                session_stats['fps'] = frame_count / (current_time - (session_start_time or current_time))
                last_fps_time = current_time
            
            # Control frame rate
            time.sleep(0.033)  # ~30 FPS monitoring rate
            
    except Exception as e:
        logger.error(f"Camera worker error: {e}")
    finally:
        if cap is not None:
            cap.release()
        camera_active = False
        logger.info("🎥 Camera worker stopped")

# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('status_update', {
        'camera_active': camera_active,
        'recording_active': recording_active,
        'current_session_id': current_session_id
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('request_status')
def handle_status_request():
    """Handle status request from client"""
    emit('status_update', {
        'camera_active': camera_active,
        'recording_active': recording_active,
        'current_session_id': current_session_id,
        'session_stats': session_stats,
        'recent_events': recent_events[-5:] if recent_events else []
    })

if __name__ == '__main__':
    logger.info("🚀 Starting CheatGPT Web Application...")
    logger.info(f"📁 Videos directory: {VIDEOS_DIR}")
    logger.info(f"📁 Recordings directory: {RECORDINGS_DIR}")
    
    # Initialize database
    db.init_db()
    
    # Run the app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
