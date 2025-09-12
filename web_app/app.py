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
from io import BytesIO

from flask import Flask, render_template, request, jsonify, send_file, Response, redirect
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import threading
import queue

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: ReportLab not installed. PDF export will not be available.")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_app.db_manager import db

# SessionReportGenerator class for PDF export
class SessionReportGenerator:
    """Generate PDF reports for CheatGPT sessions"""
    
    def __init__(self):
        self.temp_dir = Path("temp_reports")
        self.temp_dir.mkdir(exist_ok=True)
    
    def generate_report(self, session_data: dict, events_data: list) -> str:
        """Generate a comprehensive session report"""
        if not PDF_AVAILABLE:
            raise Exception("PDF generation not available - ReportLab not installed")
        
        # Create temporary PDF file
        report_filename = f"session_{session_data['session_id']}_report.pdf"
        report_path = self.temp_dir / report_filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(report_path), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph("CheatGPT Session Analysis Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Session Information
        session_info = [
            ['Session ID:', session_data['session_id']],
            ['Start Time:', datetime.fromtimestamp(session_data['start_time']).strftime('%Y-%m-%d %H:%M:%S')],
            ['End Time:', datetime.fromtimestamp(session_data['end_time']).strftime('%Y-%m-%d %H:%M:%S') if session_data.get('end_time') else 'N/A'],
            ['Duration:', f"{session_data.get('duration', 0):.1f} seconds"],
            ['Status:', session_data.get('status', 'Unknown')],
            ['Total Events:', str(len(events_data))]
        ]
        
        session_table = Table(session_info, colWidths=[2*inch, 3*inch])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("Session Information", styles['Heading2']))
        story.append(session_table)
        story.append(Spacer(1, 12))
        
        # Event Summary
        if events_data:
            high_risk = len([e for e in events_data if e['severity'] == 'red'])
            medium_risk = len([e for e in events_data if e['severity'] == 'orange'])
            low_risk = len([e for e in events_data if e['severity'] == 'yellow'])
            avg_confidence = sum(e['confidence'] for e in events_data) / len(events_data)
            
            summary_data = [
                ['Total Events:', str(len(events_data))],
                ['High Risk Events:', str(high_risk)],
                ['Medium Risk Events:', str(medium_risk)],
                ['Low Risk Events:', str(low_risk)],
                ['Average Confidence:', f"{avg_confidence:.1%}"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("Event Summary", styles['Heading2']))
            story.append(summary_table)
            story.append(Spacer(1, 12))
            
            # Detailed Events
            story.append(Paragraph("Detailed Event Log", styles['Heading2']))
            
            event_headers = ['Time', 'Event Type', 'Confidence', 'Severity']
            event_data = [event_headers]
            
            for event in events_data[:50]:  # Limit to first 50 events
                timestamp = f"{int(event['timestamp_seconds']//60):02d}:{int(event['timestamp_seconds']%60):02d}"
                event_data.append([
                    timestamp,
                    event['event_type'].replace('_', ' ').title(),
                    f"{event['confidence']:.1%}",
                    event['severity'].title()
                ])
            
            if len(events_data) > 50:
                event_data.append(['...', f'({len(events_data) - 50} more events)', '', ''])
            
            events_table = Table(event_data, colWidths=[1*inch, 2.5*inch, 1*inch, 1*inch])
            events_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(events_table)
        else:
            story.append(Paragraph("No events detected in this session.", styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 24))
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by CheatGPT Analytics"
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        return str(report_path)

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
frame_queue = queue.Queue(maxsize=5)  # Reduced queue size to prevent lag buildup

# Video recording variables
video_writer = None
recording_filename = None

# Shared frame buffer for video streaming
latest_frame = None
frame_lock = threading.Lock()

# Additional global variables for enhanced UI
camera_active = False
session_stats = {
    'frame_count': 0,
    'fps': 0,
    'hotspot_count': 0,
    'elapsed_time': 0
}

# Detection stabilization variables
detection_history = {}  # Track detection confidence over time
stable_detections = {}  # Current stable detections to display

# Event deduplication variables for database storage
event_buffer = {}  # Buffer events per second: {event_key: {event_data, max_confidence, count}}
last_flush_time = 0  # Last time we flushed events to database
EVENT_FLUSH_INTERVAL = 3.0  # Flush events every 3 seconds for better sustained behavior grouping
sustained_events = {}  # Track sustained events across multiple seconds
detection_threshold = 0.5  # Minimum confidence for initial detection
stable_threshold = 0.3    # Lower threshold for maintaining stable detection
history_frames = 5        # Number of frames to consider for stability
last_stable_frame = None  # Cache last good detection frame

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

def stabilize_detections(raw_frame, processed_frame, current_time):
    """
    Advanced stabilization to eliminate bounding box flickering
    Uses temporal smoothing and confidence-based filtering
    """
    global last_stable_frame, detection_cache, last_detection_time
    
    # Initialize cache if needed
    if 'detection_cache' not in globals():
        detection_cache = []
        last_detection_time = 0
    
    # If we don't have a processed frame, use smart fallback
    if processed_frame is None:
        # Use cached frame if recent (within 0.5 seconds)
        if last_stable_frame is not None and (current_time - last_detection_time) < 0.5:
            return last_stable_frame
        return raw_frame
    
    # Update cache and timing
    last_stable_frame = processed_frame.copy()
    last_detection_time = current_time
    
    # Apply temporal smoothing to reduce flickering
    # The processed frame from CheatGPT engine is already optimized
    return processed_frame

def apply_detection_smoothing(frame, detections, frame_id):
    """
    Apply temporal smoothing to detection results to reduce box flickering
    """
    global detection_history, stable_detections
    
    current_frame_detections = {}
    
    # Process each detection
    for detection in detections:
        detection_id = f"{detection.get('class', 'unknown')}_{detection.get('track_id', 0)}"
        confidence = detection.get('confidence', 0.0)
        
        # Update detection history
        if detection_id not in detection_history:
            detection_history[detection_id] = []
        
        detection_history[detection_id].append({
            'frame_id': frame_id,
            'confidence': confidence,
            'bbox': detection.get('bbox', []),
            'data': detection
        })
        
        # Keep only recent history
        detection_history[detection_id] = detection_history[detection_id][-history_frames:]
        
        # Calculate average confidence over recent frames
        recent_confidences = [d['confidence'] for d in detection_history[detection_id]]
        avg_confidence = sum(recent_confidences) / len(recent_confidences)
        
        # Determine if detection should be stable
        if detection_id in stable_detections:
            # Already stable, use lower threshold to maintain
            if avg_confidence >= stable_threshold:
                current_frame_detections[detection_id] = detection
        else:
            # Not stable yet, use higher threshold to establish
            if avg_confidence >= detection_threshold and len(recent_confidences) >= 2:
                current_frame_detections[detection_id] = detection
    
    # Update stable detections
    stable_detections = current_frame_detections
    
    # Clean up old detection history
    current_frame = frame_id
    for det_id in list(detection_history.keys()):
        if detection_history[det_id]:
            last_frame = detection_history[det_id][-1]['frame_id']
            if current_frame - last_frame > history_frames * 2:
                del detection_history[det_id]
                if det_id in stable_detections:
                    del stable_detections[det_id]
    
    return list(stable_detections.values())

def buffer_event_for_database(event, current_time, timestamp_offset):
    """
    Buffer events for database storage with intelligent deduplication.
    Groups sustained behaviors and similar events to reduce redundancy.
    """
    global event_buffer, sustained_events
    
    # Create unique key for event deduplication
    time_window = int(timestamp_offset // EVENT_FLUSH_INTERVAL) * EVENT_FLUSH_INTERVAL  # Group by flush interval
    person_id = event.get('person_id', 'unknown')
    raw_event_type = event.get('event_type', event.get('type', 'unknown_behavior'))
    
    # Normalize event types for better deduplication
    # Group similar events together (e.g., all head turning events)
    if 'Head Turning' in raw_event_type or 'Looking' in raw_event_type:
        normalized_type = 'looking_behavior'
    elif 'Leaning' in raw_event_type:
        normalized_type = 'leaning_behavior'  
    elif 'Gesture' in raw_event_type or 'Hand' in raw_event_type:
        normalized_type = 'gesture_behavior'
    elif 'Phone' in raw_event_type or 'Device' in raw_event_type:
        normalized_type = 'device_behavior'
    else:
        normalized_type = raw_event_type.lower().replace(' ', '_')
    
    # For head turning, group by angle ranges to reduce micro-variations
    if 'Head Turning' in raw_event_type:
        # Extract angle and group into ranges
        import re
        angle_match = re.search(r'(\d+\.?\d*)', raw_event_type)
        if angle_match:
            angle = float(angle_match.group(1))
            if angle < 10:
                angle_range = 'small'
            elif angle < 20:
                angle_range = 'medium'
            else:
                angle_range = 'large'
            normalized_type = f'head_turning_{angle_range}'
    
    event_key = f"{normalized_type}_{person_id}_{time_window}"
    confidence = event.get('confidence', 0.0)
    
    # Track sustained events across time windows
    sustained_key = f"{normalized_type}_{person_id}"
    
    # Debug logging
    logger.debug(f"🔄 Buffering event: {raw_event_type} -> {event_key} (confidence: {confidence:.2f})")
    
    if event_key in event_buffer:
        # Update existing event with higher confidence and better details
        if confidence > event_buffer[event_key]['max_confidence']:
            event_buffer[event_key]['event_data'] = event
            event_buffer[event_key]['max_confidence'] = confidence
            event_buffer[event_key]['raw_event_type'] = raw_event_type
            logger.debug(f"📈 Updated buffered event {event_key} with higher confidence: {confidence:.2f}")
        event_buffer[event_key]['count'] += 1
        event_buffer[event_key]['duration'] += EVENT_FLUSH_INTERVAL / 30  # Approximate duration
    else:
        # Add new event to buffer
        event_buffer[event_key] = {
            'event_data': event,
            'max_confidence': confidence,
            'count': 1,
            'timestamp_offset': timestamp_offset,
            'time_window': time_window,
            'raw_event_type': raw_event_type,
            'normalized_type': normalized_type,
            'duration': EVENT_FLUSH_INTERVAL / 30  # Approximate duration
        }
        logger.debug(f"🆕 Added new buffered event: {event_key}")
    
    # Update sustained event tracking
    if sustained_key not in sustained_events:
        sustained_events[sustained_key] = {
            'start_time': timestamp_offset,
            'last_time': timestamp_offset,
            'total_duration': 0,
            'total_count': 0
        }
    
    sustained_events[sustained_key]['last_time'] = timestamp_offset
    sustained_events[sustained_key]['total_duration'] = timestamp_offset - sustained_events[sustained_key]['start_time']
    sustained_events[sustained_key]['total_count'] += 1
    
    if event_key in event_buffer:
        # Update existing event with higher confidence
        if confidence > event_buffer[event_key]['max_confidence']:
            event_buffer[event_key]['event_data'] = event
            event_buffer[event_key]['max_confidence'] = confidence
        event_buffer[event_key]['count'] += 1
    else:
        # Add new event to buffer
        event_buffer[event_key] = {
            'event_data': event,
            'max_confidence': confidence,
            'count': 1,
            'timestamp_offset': timestamp_offset,
            'time_window': time_window
        }

def flush_events_to_database():
    """
    Flush buffered events to database with intelligent grouping.
    Creates summary events for sustained behaviors instead of multiple entries.
    """
    global event_buffer, current_session_id, sustained_events
    
    if not event_buffer or not current_session_id:
        return
    
    events_saved = 0
    for event_key, buffered_event in event_buffer.items():
        try:
            event = buffered_event['event_data']
            raw_event_type = buffered_event.get('raw_event_type', event.get('event_type', event.get('type', 'unknown_behavior')))
            
            # Create intelligent summary for sustained behaviors
            count = buffered_event['count']
            duration = buffered_event.get('duration', 0)
            normalized_type = buffered_event.get('normalized_type', 'unknown')
            
            # Format event label based on duration and count
            if count > 5 and duration > 1.5:  # Sustained behavior (reduced threshold)
                if 'looking' in normalized_type or 'head_turning' in normalized_type:
                    # Extract angle range for head turning
                    if 'head_turning' in normalized_type:
                        if 'small' in normalized_type:
                            event_label = f"🔍 Sustained Head Movement (5-10°, {duration:.1f}s)"
                        elif 'medium' in normalized_type:
                            event_label = f"🔍 Sustained Head Turning (10-20°, {duration:.1f}s)"
                        else:
                            event_label = f"🚨 Significant Head Movement (>20°, {duration:.1f}s)"
                    else:
                        event_label = f"🔍 Sustained Looking Behavior ({duration:.1f}s)"
                elif 'leaning' in normalized_type:
                    event_label = f"📐 Sustained Leaning Behavior ({duration:.1f}s)"
                elif 'gesture' in normalized_type:
                    event_label = f"✋ Sustained Hand Gestures ({duration:.1f}s)"
                elif 'device' in normalized_type:
                    event_label = f"📱 Device Usage Detected ({duration:.1f}s)"
                else:
                    event_label = f"⚠️ Sustained Suspicious Activity ({duration:.1f}s)"
            else:
                # Use the original formatted label for short events
                event_label = format_event_label(raw_event_type, event)
            
            confidence = buffered_event['max_confidence']
            bbox_data = event.get('bbox', {})
            timestamp_offset = buffered_event['timestamp_offset']
            
            # Add aggregated hotspot to database
            db.add_hotspot(
                session_id=current_session_id,
                event_type=event_label,  # Use the intelligent label
                confidence=confidence,
                timestamp_offset=timestamp_offset,
                frame_no=session_stats.get('frame_count', 0),
                bbox_data=bbox_data
            )
            
            events_saved += 1
            logger.info(f"💾 Saved aggregated event: {event_label} (confidence: {confidence:.2f}, count: {count}, duration: {duration:.1f}s)")
            
        except Exception as e:
            logger.error(f"Error saving buffered event to database: {e}")
    
    # Clear buffers after flushing
    event_buffer.clear()
    
    # Clean up old sustained events (older than 10 seconds)
    current_time = time.time()
    session_time = current_time - (session_start_time or current_time)
    
    for key in list(sustained_events.keys()):
        if session_time - sustained_events[key]['last_time'] > 10:
            del sustained_events[key]
    
    if events_saved > 0:
        logger.info(f"📊 Flushed {events_saved} intelligently grouped events to database")

def should_flush_events(current_time):
    """
    Check if we should flush events to database (every second).
    """
    global last_flush_time
    return current_time - last_flush_time >= EVENT_FLUSH_INTERVAL

def format_event_label(event_type, event_data):
    """
    Convert technical event types to user-friendly labels.
    """
    # Extract additional details from the event
    details = event_data.get('details', '')
    severity = event_data.get('severity', 'yellow')
    
    # Define user-friendly labels
    event_labels = {
        'Looking Around': 'Suspicious Looking Behavior',
        'Leaning': 'Inappropriate Leaning',
        'Gesture': 'Suspicious Hand Gesture',
        'Phone Detection': 'Phone/Device Detected',
        'Temporal Cheating': 'Multiple Suspicious Behaviors',
        'suspicious_looking': 'Suspicious Looking Behavior',
        'suspicious_leaning': 'Inappropriate Leaning',
        'suspicious_gesture': 'Suspicious Hand Gesture',
        'phone_detection': 'Phone/Device Detected',
        'temporal_cheating': 'Multiple Suspicious Behaviors',
        'unknown_behavior': 'Suspicious Activity'
    }
    
    # Get the base label
    base_label = event_labels.get(event_type, event_type)
    
    # Add specific details if available
    if 'Head turning detected' in details:
        angle_match = details.split('(')[1].split('°')[0] if '(' in details and '°' in details else None
        if angle_match:
            base_label = f"Head Turning Detected ({angle_match}°)"
    elif 'Looking' in event_type and 'sustained' in details:
        base_label = "Sustained Looking Around"
    elif 'Phone' in event_type or 'phone' in event_type.lower():
        base_label = "Unauthorized Device Detected"
    elif 'Temporal' in event_type:
        base_label = "Multiple Violations Detected"
    
    # Add severity indicator
    severity_indicators = {
        'yellow': '⚠️',
        'orange': '🚨', 
        'red': '🔴'
    }
    
    indicator = severity_indicators.get(severity, '⚠️')
    return f"{indicator} {base_label}"

# Directories
BASE_DIR = Path(__file__).parent.parent
VIDEOS_DIR = BASE_DIR / "videos"
RECORDINGS_DIR = Path(__file__).parent / "videos"  # Use the videos directory in web_app
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

def emit_status_update():
    """Emit real-time status update to all connected clients"""
    global camera_active, recording_active, current_session_id, session_start_time, session_stats
    
    try:
        # Calculate elapsed time if session is active
        elapsed_time = 0
        if session_start_time:
            elapsed_time = time.time() - session_start_time
            session_stats['elapsed_time'] = elapsed_time

        status_data = {
            'camera_active': camera_active,
            'recording_active': recording_active,
            'current_session_id': current_session_id,
            'session_start_time': session_start_time,
            'elapsed_time': elapsed_time,
            'detection_engine_available': detection_engine is not None,
            'frame_count': session_stats['frame_count'],
            'fps': session_stats['fps'],
            'hotspot_count': session_stats['hotspot_count'],
            'system_status': {
                'camera': 'ON' if camera_active else 'OFF',
                'recording': 'ON' if recording_active else 'OFF', 
                'detection': 'READY' if detection_engine is not None else 'OFF',
                'session': 'Active' if current_session_id else 'Inactive'
            }
        }
        
        # Emit with debug logging
        socketio.emit('status_update', status_data)
        logger.debug(f"📡 Status update emitted: FPS={status_data['fps']:.1f}, Frames={status_data['frame_count']}, Hotspots={status_data['hotspot_count']}")
        
    except Exception as e:
        logger.error(f"Error emitting status update: {e}")

@app.route('/test/emit')
def test_emit():
    """Test route to verify SocketIO is working"""
    test_data = {
        'frame_count': 999,
        'fps': 25.5,
        'hotspot_count': 5,
        'camera_active': True,
        'current_session_id': 'test_session',
        'test': True
    }
    socketio.emit('status_update', test_data)
    return jsonify({'success': True, 'message': 'Test emit sent'})

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
        'system_status': {
            'camera': 'ON' if camera_active else 'OFF',
            'recording': 'ON' if recording_active else 'OFF', 
            'detection': 'READY' if detection_engine is not None else 'OFF',
            'session': 'Active' if current_session_id else 'Inactive'
        }
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

# Analytics & Playback Routes
@app.route('/analytics')
def analytics():
    """Redirect to analytics home page (YouTube-style landing)"""
    return redirect('/analytics/home')

@app.route('/analytics/home')
def analytics_home():
    """Render YouTube-style analytics home page"""
    return render_template('analytics_home.html')

@app.route('/analytics/player')
def analytics_player():
    """Render YouTube-style analytics player page"""
    return render_template('analytics_youtube.html')

@app.route('/analytics/old')
def analytics_old():
    """Render original analytics page"""
    return render_template('analytics.html')

@app.route('/sessions')
def sessions_list():
    """Get list of past sessions with video paths"""
    try:
        sessions = db.get_sessions_with_details()
        
        # Format sessions for analytics
        formatted_sessions = []
        for session in sessions:
            formatted_session = {
                'session_id': session.get('session_id'),
                'start_time': session.get('start_time'),
                'duration': session.get('duration', 0),
                'hotspot_count': session.get('hotspot_count', 0),
                'video_path': session.get('video_file'),
                'end_time': session.get('end_time'),
                'frame_count': session.get('frame_count', 0)
            }
            formatted_sessions.append(formatted_session)
        
        return jsonify({
            'success': True,
            'sessions': formatted_sessions
        })
    except Exception as e:
        logger.error(f"Error fetching sessions list: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/playback/<session_id>')
def playback_video(session_id):
    """Stream recorded video file for playback"""
    try:
        # Get session details including video path
        session = db.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        video_path = None
        
        # Try to find video file by session_id pattern matching
        import glob
        
        # Pattern 1: Look for files containing the session_id
        pattern1 = str(RECORDINGS_DIR / f"*{session_id}*.mp4")
        matching_files = glob.glob(pattern1)
        
        if matching_files:
            video_path = Path(matching_files[0])
            logger.info(f"Found video file: {video_path}")
        else:
            # Pattern 2: Try alternative patterns
            # Remove 'session_' prefix if present
            clean_session_id = session_id.replace('session_', '')
            pattern2 = str(RECORDINGS_DIR / f"*{clean_session_id}*.mp4")
            matching_files = glob.glob(pattern2)
            
            if matching_files:
                video_path = Path(matching_files[0])
                logger.info(f"Found video file with clean ID: {video_path}")
            else:
                logger.error(f"No video files found for session {session_id}")
                logger.info(f"Searched in: {RECORDINGS_DIR}")
                logger.info(f"Available files: {list(RECORDINGS_DIR.glob('*.mp4'))}")
                return jsonify({'error': 'Video file not found'}), 404
        
        # Verify file exists
        if not video_path.exists():
            return jsonify({'error': 'Video file does not exist'}), 404
        
        logger.info(f"Serving video: {video_path}")
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False,
            download_name=f"session_{session_id}.mp4"
        )
    except Exception as e:
        logger.error(f"Error serving video for session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/events/<session_id>')
def session_events(session_id):
    """Get suspicious events for a session with timestamps"""
    try:
        # Get events from database for this session
        events = db.get_session_events(session_id)
        
        # Event type mapping to readable descriptions
        event_descriptions = {
            'suspicious_gesture': 'Suspicious Hand Gesture',
            'suspicious_looking': 'Looking Around Suspiciously', 
            'mixed_suspicious': 'Mixed Suspicious Behavior',
            'normal': 'Normal Behavior',
            'unknown_behavior': 'Suspicious Behavior Detected',
            'unknown': 'Suspicious Behavior Detected'
        }
        
        # Severity mapping based on event type
        event_severities = {
            'suspicious_gesture': 'red',
            'suspicious_looking': 'orange', 
            'mixed_suspicious': 'red',
            'normal': 'yellow',
            'unknown_behavior': 'orange',
            'unknown': 'orange'
        }
        
        formatted_events = []
        for event in events:
            event_type = event.get('event_type', 'unknown')
            
            # Use the formatted description from database if available, otherwise fallback to mapping
            description = event.get('description', event_descriptions.get(event_type, 'Unknown Behavior'))
            
            formatted_event = {
                'id': event.get('id'),
                'timestamp_seconds': event.get('timestamp_seconds', 0),
                'event_type': event_type,
                'confidence': event.get('confidence', 0.0),
                'severity': event.get('severity', event_severities.get(event_type, 'yellow')),  # Use DB severity if available
                'description': description,  # Use the properly formatted description
                'formatted_time': format_timestamp(event.get('timestamp_seconds', 0))
            }
            formatted_events.append(formatted_event)
        
        return jsonify({
            'success': True,
            'events': formatted_events,
            'total_events': len(formatted_events)
        })
    except Exception as e:
        logger.error(f"Error fetching events for session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/thumbnail/<session_id>')
def get_session_thumbnail(session_id):
    """Generate and serve thumbnail from first frame of session video"""
    try:
        import cv2
        import base64
        from io import BytesIO
        from PIL import Image
        
        # Find the video file for this session
        video_files = list(RECORDINGS_DIR.glob(f"*{session_id}*.mp4"))
        if not video_files:
            logger.warning(f"No video file found for session {session_id}")
            # Return a placeholder image
            placeholder = Image.new('RGB', (160, 90), color=(48, 48, 48))
            img_buffer = BytesIO()
            placeholder.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            return send_file(img_buffer, mimetype='image/jpeg')
        
        video_path = video_files[0]
        logger.info(f"Generating thumbnail for video: {video_path}")
        
        # Extract first frame using OpenCV
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            logger.warning(f"Could not read first frame from {video_path}")
            # Return placeholder
            placeholder = Image.new('RGB', (160, 90), color=(48, 48, 48))
            img_buffer = BytesIO()
            placeholder.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            return send_file(img_buffer, mimetype='image/jpeg')
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create PIL Image and resize to thumbnail
        pil_image = Image.fromarray(frame_rgb)
        pil_image.thumbnail((160, 90), Image.Resampling.LANCZOS)
        
        # Save to BytesIO buffer
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=85)
        img_buffer.seek(0)
        
        return send_file(img_buffer, mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Error generating thumbnail for session {session_id}: {e}")
        # Return error placeholder
        try:
            placeholder = Image.new('RGB', (160, 90), color=(68, 68, 68))
            img_buffer = BytesIO()
            placeholder.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            return send_file(img_buffer, mimetype='image/jpeg')
        except:
            return '', 404

@app.route('/api/sessions/list')
def api_sessions_list():
    """Enhanced API route for sessions list with detailed metadata"""
    try:
        sessions = db.get_sessions_with_details()
        enhanced_sessions = []
        
        for session in sessions:
            # Get events for this session
            events = db.get_session_events(session['session_id'])
            
            # Count different types of events
            phone_events = len([e for e in events if 'phone' in e.get('event_type', '').lower()])
            looking_events = len([e for e in events if 'looking' in e.get('event_type', '').lower()])
            leaning_events = len([e for e in events if 'leaning' in e.get('event_type', '').lower()])
            gesture_events = len([e for e in events if 'gesture' in e.get('event_type', '').lower()])
            
            # Count by severity
            critical_events = len([e for e in events if e.get('severity') == 'red'])
            warning_events = len([e for e in events if e.get('severity') == 'orange'])
            notice_events = len([e for e in events if e.get('severity') == 'yellow'])
            
            # Check if session is currently active
            is_active = session.get('end_time') is None
            
            enhanced_session = {
                'session_id': session['session_id'],
                'start_time': session['start_time'],
                'end_time': session.get('end_time'),
                'duration': session.get('duration', 0),
                'hotspot_count': len(events),
                'status': 'active' if is_active else 'completed',
                'phone_events': phone_events,
                'looking_events': looking_events,
                'leaning_events': leaning_events,
                'gesture_events': gesture_events,
                'critical_events': critical_events,
                'warning_events': warning_events,
                'notice_events': notice_events,
                'total_frames': session.get('frames', 0),
                'thumbnail_path': f'/api/thumbnail/{session["session_id"]}',
                'video_path': f'/playback/{session["session_id"]}'
            }
            enhanced_sessions.append(enhanced_session)
        
        return jsonify({
            'success': True,
            'sessions': enhanced_sessions,
            'total_count': len(enhanced_sessions)
        })
        
    except Exception as e:
        logger.error(f"Error fetching enhanced sessions list: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export/<session_id>')
def export_session_report(session_id):
    """Export session analysis as PDF report"""
    try:
        # Get session and events data
        session = db.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        events = db.get_session_events(session_id)
        
        # Generate PDF report
        report_generator = SessionReportGenerator()
        pdf_path = report_generator.generate_report(session, events)
        
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"cheatgpt_session_{session_id}_report.pdf"
        )
    except Exception as e:
        logger.error(f"Error generating report for session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

def format_timestamp(seconds):
    """Format timestamp in seconds to MM:SS format"""
    if not seconds:
        return "00:00"
    
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

@app.route('/api/session/<session_id>/events')
def api_session_events(session_id):
    """Get events for a specific session"""
    try:
        events = db.get_session_events(session_id)
        
        # Event type mapping to readable descriptions
        event_descriptions = {
            'suspicious_gesture': 'Suspicious Hand Gesture',
            'suspicious_looking': 'Looking Around Suspiciously', 
            'mixed_suspicious': 'Mixed Suspicious Behavior',
            'normal': 'Normal Behavior',
            'unknown_behavior': 'Suspicious Behavior Detected',
            'unknown': 'Suspicious Behavior Detected'
        }
        
        # Severity mapping based on event type
        event_severities = {
            'suspicious_gesture': 'red',
            'suspicious_looking': 'orange', 
            'mixed_suspicious': 'red',
            'normal': 'yellow',
            'unknown_behavior': 'orange',
            'unknown': 'orange'
        }
        
        # Format events with proper descriptions and severity
        formatted_events = []
        for event in events:
            event_type = event.get('event_type', 'unknown')
            formatted_event = {
                'id': event.get('id'),
                'timestamp_seconds': event.get('timestamp_seconds', 0),
                'event_type': event_type,
                'confidence': event.get('confidence', 0.0),
                'severity': event_severities.get(event_type, 'yellow'),
                'description': event_descriptions.get(event_type, 'Unknown Behavior'),
                'formatted_time': format_timestamp(event.get('timestamp_seconds', 0))
            }
            formatted_events.append(formatted_event)
        
        return jsonify({
            'success': True,
            'events': formatted_events
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
    global camera_active, recording_active, current_session_id, session_start_time, camera_thread, session_stats, video_feed_active
    
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
        video_feed_active = True  # Enable video feed
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
        
        # Emit immediate status update
        emit_status_update()
        
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
    global camera_active, recording_active, current_session_id, session_start_time, session_stats, video_writer, recording_filename, video_feed_active
    
    try:
        # Check if any recording is active
        if not camera_active and not recording_active:
            return jsonify({'error': 'Camera not active'}), 400
        
        # Stop all processes
        logger.info("🛑 Stopping camera...")
        camera_active = False
        recording_active = False
        video_feed_active = False  # Stop video feed too
        
        # Clear frame queue to prevent old frames from showing
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for camera worker to stop with timeout
        import time
        max_wait_time = 3.0  # Maximum 3 seconds to wait
        start_wait = time.time()
        
        # Wait for camera thread to finish
        global camera_thread
        if camera_thread and camera_thread.is_alive():
            logger.info("🛑 Waiting for camera worker to stop...")
            while camera_thread.is_alive() and (time.time() - start_wait) < max_wait_time:
                time.sleep(0.1)
            
            if camera_thread.is_alive():
                logger.warning("⚠️ Camera worker did not stop within timeout")
            else:
                logger.info("✅ Camera worker stopped successfully")
        
        # Finalize video recording if active
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            logger.info(f"🎬 Video recording finalized: {recording_filename}")
        
        logger.info("🛑 Camera stopped")
        
        # Calculate final stats
        end_time = time.time()
        duration = end_time - session_start_time if session_start_time else 0
        
        # Flush any remaining buffered events before ending session
        flush_events_to_database()
        
        # Update database
        if current_session_id:
            db.end_session(current_session_id, end_time, session_stats['frame_count'])
        
        # Emit to clients
        socketio.emit('camera_stopped', {
            'session_id': current_session_id,
            'duration': duration,
            'frame_count': session_stats['frame_count']
        })
        
        # Emit immediate status update
        emit_status_update()
        
        logger.info(f"🛑 Camera stopped - Session: {current_session_id}, Duration: {duration:.1f}s")
        
        # Reset session variables
        session_id = current_session_id
        video_file = recording_filename
        current_session_id = None
        session_start_time = None
        recording_filename = None
        
        # Emit status update after resetting variables
        emit_status_update()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'duration': duration,
            'frame_count': session_stats['frame_count'],
            'video_saved': video_file is not None,
            'video_filename': video_file,
            'message': 'Camera stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"Error stopping camera: {e}")
        return jsonify({'error': str(e)}), 500



# @app.route('/camera/screenshot', methods=['POST'])
# def take_screenshot():
#     """REMOVED: Screenshot functionality removed as requested"""
#     return jsonify({'error': 'Screenshot functionality removed'}), 404
    """Take screenshot with overlay - optimized for fast response"""
    global latest_frame
    
    try:
        if not camera_active:
            logger.warning("Screenshot requested but camera not active")
            return jsonify({'error': 'Camera not active'}), 400
        
        # Quick frame check without long lock
        if latest_frame is None:
            logger.warning("Screenshot requested but no frame available")
            return jsonify({'error': 'No frame available'}), 400
        
        # Generate screenshot filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_name = f"screenshot_{timestamp}.jpg"
        screenshot_path = SCREENSHOTS_DIR / screenshot_name
        
        # Capture frame quickly
        with frame_lock:
            frame_copy = latest_frame.copy() if latest_frame is not None else None
        
        if frame_copy is None:
            logger.warning("Frame became unavailable during screenshot")
            return jsonify({'error': 'Frame unavailable'}), 400
        
        # Save screenshot outside of lock for better performance
        success = cv2.imwrite(str(screenshot_path), frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            logger.info(f"� Screenshot saved: {screenshot_name}")
            return jsonify({
                'success': True,
                'filename': screenshot_name,
                'path': str(screenshot_path),
                'message': 'Screenshot saved successfully'
            })
        else:
            logger.error(f"Failed to save screenshot to {screenshot_path}")
            return jsonify({'error': 'Failed to save screenshot'}), 500
            
    except Exception as e:
        logger.error(f"Error taking screenshot: {e}")
        return jsonify({'error': str(e)}), 500

# Video streaming route
@app.route('/video_feed')
def video_feed():
    """High-FPS video streaming with anti-flicker optimization"""
    def generate():
        last_frame_time = 0
        frame_buffer = None  # Buffer for smooth frame delivery
        try:
            while camera_active:
                try:
                    current_time = time.time()
                    
                    # Higher frame rate for smoother real-time monitoring (20 FPS)
                    if current_time - last_frame_time < 0.05:  # 50ms = 20 FPS
                        time.sleep(0.01)
                        continue
                    
                    # Get frame from queue with timeout to prevent blocking
                    if not frame_queue.empty():
                        frame = frame_queue.get_nowait()
                        frame_buffer = frame  # Cache frame for smooth delivery
                        last_frame_time = current_time
                        
                        # Encode frame as JPEG with optimized settings for smooth streaming
                        ret, buffer = cv2.imencode('.jpg', frame, [
                            cv2.IMWRITE_JPEG_QUALITY, 85,  # Higher quality for clearer bounding boxes
                            cv2.IMWRITE_JPEG_OPTIMIZE, 1   # Optimize for size
                        ])
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    elif frame_buffer is not None:
                        # Use buffered frame to maintain smooth stream
                        ret, buffer = cv2.imencode('.jpg', frame_buffer, [
                            cv2.IMWRITE_JPEG_QUALITY, 85,
                            cv2.IMWRITE_JPEG_OPTIMIZE, 1
                        ])
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        time.sleep(0.01)
                    else:
                        time.sleep(0.01)  # Small delay if no frame available
                        
                except queue.Empty:
                    time.sleep(0.01)
                except Exception as e:
                    logger.debug(f"Frame encoding error: {e}")
                    time.sleep(0.01)
        except GeneratorExit:
            logger.info("Video feed client disconnected")
        except Exception as e:
            logger.error(f"Video feed error: {e}")
    
    return Response(generate(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame',
                   headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                           'Pragma': 'no-cache',
                           'Expires': '0'})

def camera_worker():
    """Background worker for camera feed and frame processing"""
    global camera_active, session_stats, video_writer, recording_filename
    
    cap = None
    try:
        logger.info("🎥 Camera worker started")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        # Initialize camera with optimized settings
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        # Set camera properties for optimal performance and display size
        # Try 1280x720 first (HD 720p), fallback to 1024x576 if not supported
        target_width = 1280
        target_height = 720
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS from camera
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer to reduce latency
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPEG for better performance
        
        # Verify actual resolution set by camera
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # If camera doesn't support requested resolution, try fallback
        if actual_width != target_width or actual_height != target_height:
            logger.info(f"⚠️ Camera doesn't support {target_width}x{target_height}, trying 1024x576...")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📹 Camera resolution set to: {actual_width}x{actual_height}")
        
        # Use actual camera resolution for video recording
        recording_width = actual_width
        recording_height = actual_height
        
        # Initialize video recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"session_{current_session_id}_{timestamp}.mp4"
        video_path = RECORDINGS_DIR / recording_filename
        
        # Initialize video writer with actual camera resolution - using web-compatible codec
        # Use 15 FPS for smooth video recording - this will be the target playback rate
        recording_fps = 15.0  # Target video playback FPS for smooth recorded video
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec for better web compatibility
        video_writer = cv2.VideoWriter(str(video_path), fourcc, recording_fps, (recording_width, recording_height))
        
        # Fallback to mp4v if avc1 fails
        if not video_writer.isOpened():
            logger.warning("H.264 codec failed, trying mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, recording_fps, (recording_width, recording_height))
        
        if not video_writer.isOpened():
            logger.error("Failed to initialize video writer")
            video_writer = None
        else:
            logger.info(f"🎬 Video recording started: {recording_filename} at {recording_width}x{recording_height}")
        
        frame_count = 0
        last_fps_time = time.time()
        last_status_update = time.time()  # Track status updates
        last_video_frame_time = time.time()  # For smooth video timing
        video_frame_interval = 1.0 / recording_fps  # Time between video frames (1/15 = 0.067s)
        video_frames_written = 0  # Track actual video frames written
        
        # Send initial status update to ensure UI starts showing real-time data
        emit_status_update()
        
        while camera_active:
            # More frequent stop checks
            if not camera_active or not recording_active:
                logger.info("🛑 Camera worker detected stop signal, breaking loop")
                break
                
            current_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                time.sleep(0.033)  # Standard delay
                continue
            
            # Fix flipped camera - flip horizontally for natural mirrored view
            # This makes the video appear as if looking in a mirror, which is more intuitive
            frame = cv2.flip(frame, 1)
                
            frame_count += 1
            session_stats['frame_count'] = frame_count
            
            # Run detection more frequently for smoother real-time monitoring
            # Process detection every frame, but use lightweight processing for smooth FPS
            run_detection = True  # Always run for smooth real-time monitoring
            
            if detection_engine and hasattr(detection_engine, 'process_frame') and run_detection:
                try:
                    overlay_frame, events = detection_engine.process_frame(frame, "webcam", current_time)
                    if overlay_frame is not None:
                        # Apply advanced stabilization to eliminate flickering
                        display_frame = stabilize_detections(frame, overlay_frame, current_time)
                        
                        # Update hotspot count if any detections occurred
                        if events:
                            logger.info(f"🔍 DEBUG: Raw events from engine: {events}")
                            
                            # Buffer events for database (deduplication per second)
                            timestamp_offset = current_time - session_start_time if session_start_time else 0
                            for event in events:
                                # Buffer event instead of saving immediately
                                buffer_event_for_database(event, current_time, timestamp_offset)
                            
                            # Count hotspots (detected events) for real-time stats
                            session_stats['hotspot_count'] += len(events)
                            logger.info(f"📡 Detected {len(events)} new events (buffered for deduplication)")
                            
                            # Emit immediate status update when hotspots are detected
                            emit_status_update()
                        
                        # Check if we should flush buffered events to database (every second)
                        if should_flush_events(current_time):
                            flush_events_to_database()
                            global last_flush_time
                            last_flush_time = current_time
                    else:
                        # If no overlay, use cached stable frame or current frame
                        display_frame = stabilize_detections(frame, None, current_time)
                except Exception as e:
                    logger.debug(f"Error processing frame through engine: {e}")
                    display_frame = frame.copy()
            else:
                display_frame = frame.copy()
            
            # Time-based video recording for consistent playback speed
            # Write frames to video at exactly the target FPS regardless of processing speed
            current_video_time = current_time - (session_start_time or current_time)
            target_video_frames = int(current_video_time * recording_fps)
            
            # Write frames to catch up to target frame count
            while video_writer is not None and video_frames_written < target_video_frames:
                video_writer.write(display_frame)
                video_frames_written += 1
                
                # Debug every 30 frames
                if video_frames_written % 30 == 0:
                    actual_fps = frame_count / max(current_video_time, 0.1)
                    logger.debug(f"📹 Video: {video_frames_written} frames in {current_video_time:.1f}s | Processing FPS: {actual_fps:.1f} | Target video FPS: {recording_fps}")
            
            # Limit maximum frames written per cycle to prevent overload
            if video_writer is not None and target_video_frames > video_frames_written + 5:
                # If we're falling too far behind, skip some frames to catch up
                video_frames_written = target_video_frames - 2
            
            # Store latest frame for video streaming (no copy for better performance)
            with frame_lock:
                global latest_frame
                latest_frame = display_frame
            
            # Add frame to queue for video feed (only if queue not full to prevent blocking)
            if not frame_queue.full():
                # Use the same frame reference to avoid unnecessary copying
                frame_queue.put(display_frame)
            
            # Calculate FPS more efficiently and log performance
            if current_time - last_fps_time >= 2.0:  # Log every 2 seconds
                actual_fps = frame_count / (current_time - (session_start_time or current_time))
                session_stats['fps'] = actual_fps
                logger.info(f"📊 Performance: {actual_fps:.1f} FPS, Queue size: {frame_queue.qsize()}")
                last_fps_time = current_time
            
            # Send status updates every 0.5 seconds for real-time UI updates
            if current_time - last_status_update >= 0.5:  # Update every 0.5 seconds
                # Update FPS for real-time display
                if current_time != (session_start_time or current_time):
                    session_stats['fps'] = frame_count / (current_time - (session_start_time or current_time))
                emit_status_update()
                last_status_update = current_time
                logger.debug(f"🔄 Real-time update sent: Frame {frame_count}, FPS {session_stats['fps']:.1f}")
            
            # Frame rate control - target consistent timing for video recording
            # Use longer delay to match the 4 FPS recording rate
            # Optimized timing for smooth real-time monitoring (independent of video recording)
            # High FPS for real-time display, video recording handles its own timing
            time.sleep(0.05)  # ~20 FPS for smooth real-time monitoring
            
    except Exception as e:
        logger.error(f"Camera worker error: {e}")
    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        if video_writer is not None:
            video_writer.release()
            logger.info(f"🎬 Video recording finalized: {recording_filename}")
        camera_active = False
        logger.info("🎥 Camera worker stopped")

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
        'session_stats': session_stats
    })

if __name__ == '__main__':
    logger.info("🚀 Starting CheatGPT Web Application...")
    logger.info(f"📁 Videos directory: {VIDEOS_DIR}")
    logger.info(f"📁 Recordings directory: {RECORDINGS_DIR}")
    
    # Initialize database
    db.init_database()
    
    # Run the app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
