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

from flask import Flask, render_template, request, jsonify, send_file, Response, redirect, flash, url_for
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import threading
import queue
import zipfile

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

from web_app.db_manager import DatabaseManager as WebAppDB
from cheatgpt.db.db_manager import DBManager as MainDB

# Initialize both database managers
db = WebAppDB()  # For web app specific data (recorded sessions)
main_db = MainDB()  # For uploaded videos and main processing

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

class ProcessingTask:
    """Class to track video processing tasks"""
    def __init__(self, session_id: str, file_path: str):
        self.session_id = session_id
        self.file_path = file_path
        self.status = "initializing"
        self.progress = 0
        self.message = "Initializing..."
        self.result = {}
        self.error = None
        self.start_time = time.time()

def process_video_async(task: ProcessingTask):
    """Process video in background thread"""
    try:
        task.status = "processing"
        task.message = "Processing video..."
        task.progress = 10
        
        # Create database session for uploaded video
        main_db.create_uploaded_video_session(
            session_id=task.session_id,
            original_filename=os.path.basename(task.file_path),
            video_path=task.file_path
        )
        
        # Define progress callback to update task progress
        def progress_callback(progress, message):
            # Check if processing was stopped
            if task.status == 'stopped':
                logger.info(f"🛑 Processing stopped for session {task.session_id}")
                return False  # Signal to stop processing
                
            task.progress = min(progress, 90)  # Cap at 90% until completion
            task.message = message
            return True  # Continue processing
        
        # Check if stopped before starting
        if task.status == 'stopped':
            logger.info(f"🛑 Processing stopped before video processing for session {task.session_id}")
            return
        
        # Process the video with progress callback
        result = video_processor.process_video(
            input_path=task.file_path,
            session_id=task.session_id,
            progress_callback=progress_callback
        )
        
        # Check if stopped after processing
        if task.status == 'stopped':
            logger.info(f"🛑 Processing stopped after video processing for session {task.session_id}")
            return
        
        task.progress = 90
        task.message = "Saving to database..."
        
        # Store processing results in database
        main_db.update_processed_video_results(task.session_id, result)
        
        # Store events in database if they exist
        if 'events' in result:
            main_db.store_uploaded_video_events(task.session_id, result['events'])
        
        # Store hotspots if they exist
        if 'hotspots' in result:
            main_db.store_uploaded_video_hotspots(task.session_id, result['hotspots'])
        
        # Final check before completion
        if task.status == 'stopped':
            logger.info(f"🛑 Processing stopped before completion for session {task.session_id}")
            return
        
        # Update final results
        task.result = result
        task.status = "completed"
        task.message = "Processing completed successfully"
        task.progress = 100
        
        logger.info(f"✅ Video processing completed for session {task.session_id}")
        logger.info(f"   - Total events: {result.get('total_events', 0)}")
        logger.info(f"   - Results saved to database")
        
    except Exception as e:
        # Check if this is due to stopping
        if task.status == 'stopped':
            logger.info(f"🛑 Processing stopped with exception for session {task.session_id}: {e}")
            return
            
        # Update session status to error
        task.status = "error"
        task.error = str(e)
        task.message = f"Error: {str(e)}"
        task.progress = 0
        logger.error(f"❌ Video processing failed for session {task.session_id}: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce Werkzeug logging verbosity
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Reduce SocketIO logging
logging.getLogger('socketio').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING)

# Completely disable Flask development server access logs
import sys
if 'werkzeug' in sys.modules:
    import werkzeug
    werkzeug._internal._log = lambda *args: None

# Also disable werkzeug's access logger completely
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.disabled = True

# Import CheatGPT video processing components
try:
    from cheatgpt.video_processor import VideoProcessor
    from cheatgpt.upload_handler import UploadHandler
    from cheatgpt.report_generator import ReportGenerator
    VIDEO_PROCESSING_AVAILABLE = True
    logger.info("✅ CheatGPT video processing components imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import CheatGPT video processing components: {e}")
    VIDEO_PROCESSING_AVAILABLE = False

# Import CheatGPT detection engine
try:
    import sys
    import os
    
    # Add the parent directory to the path for cheatgpt imports
    cheatgpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if cheatgpt_path not in sys.path:
        sys.path.insert(0, cheatgpt_path)
    
    from cheatgpt.engines.engine_hybrid import EngineHybrid
    logger.info("✅ CheatGPT Hybrid detection engine imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import CheatGPT Hybrid detection engine: {e}")
    EngineHybrid = None

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'cheatgpt_web_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize SocketIO with CORS enabled and polling-only transport
socketio = SocketIO(app, cors_allowed_origins="*", transports=['polling'])

# Global state variables for live monitoring
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

# Video processing global variables
processing_status = {}
upload_handler = None
video_processor = None
report_generator = None

# Initialize video processing components
if VIDEO_PROCESSING_AVAILABLE:
    upload_handler = UploadHandler()
    video_processor = VideoProcessor()
    report_generator = ReportGenerator()

# Configuration directories
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

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
    """Initialize the CheatGPT Hybrid detection engine"""
    global detection_engine
    
    if EngineHybrid is None:
        logger.error("❌ CheatGPT EngineHybrid not available")
        return False
    
    try:
        detection_engine = EngineHybrid()
        logger.info("✅ Hybrid detection engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize hybrid detection engine: {e}")
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
            logger.debug(f"💾 Saved aggregated event: {event_label} (confidence: {confidence:.2f}, count: {count}, duration: {duration:.1f}s)")
            
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
        logger.info(f"� Saved {events_saved} events to database")

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

@app.route('/upload')
def video_upload():
    """Video upload page"""
    return render_template('video_upload.html')

@app.route('/guide')
def user_guide():
    """User guide and documentation page"""
    return render_template('user_guide.html')

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

@app.route('/analytics/reports')
def analytics_reports():
    """Render analytics reports page"""
    return render_template('analytics_reports.html')

@app.route('/analytics/reports/<session_id>')
def analytics_session_report(session_id):
    """Render individual session report page"""
    try:
        # Get session details by session_id string
        session = db.get_session(session_id)
        if not session:
            flash('Session not found', 'error')
            return redirect(url_for('analytics_home'))
        
        return render_template('analytics_session_report.html', session=session, session_id=session_id)
    except Exception as e:
        logger.error(f"Error loading session report {session_id}: {e}")
        flash('Error loading session report', 'error')
        return redirect(url_for('analytics_home'))

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

@app.route('/playback/processed/<session_id>/<filename>')
def playback_processed_video(session_id, filename):
    """Stream processed video file with bounding boxes for playback"""
    try:
        # Construct path to processed video in results directory
        video_path = os.path.join("results", session_id, filename)
        
        # Verify file exists and is within the results directory for security
        if not os.path.exists(video_path):
            logger.error(f"Processed video file not found: {video_path}")
            return jsonify({'error': 'Video file not found'}), 404
        
        # Security check: ensure the file is within the results directory
        results_abs_path = os.path.abspath("results")
        video_abs_path = os.path.abspath(video_path)
        if not video_abs_path.startswith(results_abs_path):
            logger.error(f"Security violation: attempted access to file outside results directory")
            return jsonify({'error': 'Access denied'}), 403
        
        # Get file info
        file_size = os.path.getsize(video_path)
        file_modified = os.path.getmtime(video_path)
        
        logger.info(f"Serving processed video: {video_path}")
        logger.info(f"File size: {file_size} bytes")
        logger.info(f"File modified: {file_modified}")
        
        # Check if file is too small (might be corrupted)
        if file_size < 1000:  # Less than 1KB
            logger.warning(f"Video file appears to be corrupted (size: {file_size} bytes)")
            return jsonify({'error': 'Video file appears to be corrupted or incomplete'}), 500
        
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False,
            download_name=filename,
            conditional=True,  # Enable HTTP range requests for better streaming
            max_age=300  # Cache for 5 minutes
        )
    except Exception as e:
        logger.error(f"Error serving processed video {session_id}/{filename}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/playback/<session_id>')
def playback_video(session_id):
    """Stream recorded or processed video file for playback"""
    try:
        # Get session details including video path - check both databases
        session = db.get_session(session_id)
        if not session:
            # Try the main database for uploaded videos
            session = main_db.get_session_info(session_id)
            logger.info(f"Checking main database for session: {session_id}")
        
        if not session:
            logger.error(f"Session not found in any database: {session_id}")
            return jsonify({'error': 'Session not found'}), 404
        
        video_path = None
        
        # Check for uploaded/processed video path first
        if session.get('processed_video_path'):
            video_path = Path(session['processed_video_path'])
            logger.info(f"Using processed video path from database: {video_path}")
        # Check if this is a processed video (has results directory path)
        elif session.get('video_path') and 'results/' in session.get('video_path', ''):
            # This is a processed video - use the path from database
            video_path = Path(session['video_path'])
            logger.info(f"Using processed video path from database: {video_path}")
        else:
            # Try to find recorded video file by session_id pattern matching
            import glob
            
            # Pattern 1: Look for files containing the session_id
            pattern1 = str(RECORDINGS_DIR / f"*{session_id}*.mp4")
            matching_files = glob.glob(pattern1)
            
            logger.info(f"Searching for recorded video files with session_id: {session_id}")
            logger.info(f"Pattern 1: {pattern1}")
            logger.info(f"Matching files: {matching_files}")
            
            if matching_files:
                video_path = Path(matching_files[0])
                logger.info(f"Found recorded video file: {video_path}")
            else:
                # Pattern 2: Try alternative patterns
                # Remove 'session_' prefix if present
                clean_session_id = session_id.replace('session_', '')
                pattern2 = str(RECORDINGS_DIR / f"*{clean_session_id}*.mp4")
                matching_files = glob.glob(pattern2)
                
                logger.info(f"Pattern 2 (clean ID): {pattern2}")
                logger.info(f"Clean session ID: {clean_session_id}")
                logger.info(f"Matching files with clean ID: {matching_files}")
                
                if matching_files:
                    video_path = Path(matching_files[0])
                    logger.info(f"Found recorded video file with clean ID: {video_path}")
                else:
                    logger.error(f"No video files found for session {session_id}")
                    logger.info(f"Searched in: {RECORDINGS_DIR}")
                    all_mp4_files = list(RECORDINGS_DIR.glob('*.mp4'))
                    logger.info(f"Available MP4 files: {all_mp4_files}")
                    
                    # Try to find by timestamp pattern if session_id contains timestamp
                    if '_' in session_id:
                        parts = session_id.split('_')
                        if len(parts) >= 3:  # session_YYYYMMDD_HHMMSS_hash format
                            timestamp_part = f"{parts[1]}_{parts[2]}"
                        pattern3 = str(RECORDINGS_DIR / f"*{timestamp_part}*.mp4")
                        timestamp_files = glob.glob(pattern3)
                        logger.info(f"Pattern 3 (timestamp): {pattern3}")
                        logger.info(f"Timestamp matching files: {timestamp_files}")
                        
                        if timestamp_files:
                            video_path = Path(timestamp_files[0])
                            logger.info(f"Found video file by timestamp: {video_path}")
                
                if not video_path:
                    return jsonify({'error': 'Video file not found'}), 404
        
        # Verify file exists and get file info
        if not video_path.exists():
            logger.error(f"Video file does not exist at path: {video_path}")
            return jsonify({'error': 'Video file does not exist'}), 404
        
        # Get file size and modification time for debugging
        file_size = video_path.stat().st_size
        file_mtime = video_path.stat().st_mtime
        import datetime
        file_modified = datetime.datetime.fromtimestamp(file_mtime)
        
        logger.info(f"Serving video: {video_path}")
        logger.info(f"File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        logger.info(f"File modified: {file_modified}")
        
        # Check if file is too small (might be corrupted)
        if file_size < 1000:  # Less than 1KB
            logger.warning(f"Video file appears to be corrupted (size: {file_size} bytes)")
            return jsonify({'error': 'Video file appears to be corrupted or incomplete'}), 500
        
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False,
            download_name=f"session_{session_id}.mp4",
            conditional=True,  # Enable HTTP range requests for better streaming
            max_age=300  # Cache for 5 minutes
        )
    except Exception as e:
        logger.error(f"Error serving video for session {session_id}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/events/<session_id>')
def session_events(session_id):
    """Get suspicious events for a session with timestamps"""
    try:
        # Get events from database for this session - check both databases
        events = db.get_session_events(session_id)
        if not events:
            # Try main database for uploaded videos
            events = main_db.get_session_events(session_id)
        
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
        
        # Find the video file for this session - check both databases
        session = db.get_session(session_id)
        if not session:
            # Try main database for uploaded videos
            session = main_db.get_session_info(session_id)
        
        video_path = None
        
        if session:
            # Check for uploaded/processed video path first
            if session.get('processed_video_path'):
                video_path = Path(session['processed_video_path'])
            elif session.get('video_path'):
                video_path = Path(session['video_path'])
        
        # Fallback to file system search for recorded videos
        if not video_path or not video_path.exists():
            video_files = list(RECORDINGS_DIR.glob(f"*{session_id}*.mp4"))
            if video_files:
                video_path = video_files[0]
        
        if not video_path or not video_path.exists():
            logger.warning(f"No video file found for session {session_id}")
            # Return a placeholder image
            placeholder = Image.new('RGB', (160, 90), color=(48, 48, 48))
            img_buffer = BytesIO()
            placeholder.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            return send_file(img_buffer, mimetype='image/jpeg')
        
        video_path = video_path
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

@app.route('/api/sessions/uploaded')
def api_uploaded_videos():
    """Get list of uploaded video sessions from database"""
    try:
        # Import the uploaded video database manager
        from cheatgpt.db.db_manager import DBManager
        upload_db = DBManager()
        
        # Get uploaded video sessions
        sessions = upload_db.get_uploaded_video_sessions(limit=100)
        
        # Format sessions for frontend compatibility
        formatted_sessions = []
        for session in sessions:
            formatted_session = {
                'session_id': session['session_id'],
                'video_title': session.get('original_filename', session['session_id']),
                'filename': session.get('original_filename', ''),
                'processing_time': session.get('created_at', session.get('start_timestamp', 0)),
                'start_time': session.get('start_timestamp', 0),
                'end_time': session.get('end_timestamp', 0),
                'duration': session.get('end_timestamp', 0) - session.get('start_timestamp', 0) if session.get('end_timestamp') and session.get('start_timestamp') else 0,
                'status': session.get('status', 'unknown'),
                'hotspot_count': session.get('total_events', 0),
                'event_count': session.get('total_events', 0),
                'video_path': session.get('processed_video_path', session.get('video_path', '')),
                'session_type': 'uploaded',
                'frame_count': session.get('frame_count', 0),
                'video_metadata': session.get('video_metadata', '{}')
            }
            formatted_sessions.append(formatted_session)
        
        return jsonify({
            'success': True,
            'videos': formatted_sessions,
            'total_count': len(formatted_sessions)
        })
        
    except Exception as e:
        logger.error(f"Error fetching uploaded videos: {e}")
        return jsonify({'error': str(e)}), 500

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
            
            # Determine session type and status
            session_type = session.get('session_type', 'recorded')
            if session_type == 'uploaded':
                status = 'uploaded'
                video_title = session.get('video_title', 'Uploaded Video')
            else:
                is_active = session.get('end_time') is None
                status = 'active' if is_active else 'completed'
                video_title = session.get('video_title', f"Session {session['session_id']}")
            
            enhanced_session = {
                'session_id': session['session_id'],
                'video_title': video_title,
                'start_time': session.get('start_time') or session.get('start_ts'),
                'end_time': session.get('end_time') or session.get('end_ts'),
                'duration': session.get('duration', 0),
                'hotspot_count': len(events),
                'status': status,
                'session_type': session_type,
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

@app.route('/api/session/<session_id>/delete', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session and its associated video file"""
    try:
        # Get session details before deletion
        session = db.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Find and delete video file(s)
        import glob
        video_files_deleted = []
        
        # Search for video files matching this session
        patterns = [
            str(RECORDINGS_DIR / f"*{session_id}*.mp4"),
            str(VIDEOS_DIR / f"*{session_id}*.mp4"),
            str(Path(__file__).parent / "videos" / f"*{session_id}*.mp4")
        ]
        
        for pattern in patterns:
            matching_files = glob.glob(pattern)
            for video_file in matching_files:
                try:
                    video_path = Path(video_file)
                    if video_path.exists():
                        video_path.unlink()  # Delete the file
                        video_files_deleted.append(str(video_path))
                        logger.info(f"🗑️ Deleted video file: {video_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete video file {video_file}: {e}")
        
        # Delete session from database (this will cascade delete events)
        success = db.delete_session(session_id)
        
        if success:
            logger.info(f"🗑️ Deleted session {session_id} and {len(video_files_deleted)} video files")
            return jsonify({
                'success': True,
                'message': f'Session deleted successfully',
                'session_id': session_id,
                'video_files_deleted': len(video_files_deleted),
                'files_deleted': video_files_deleted
            })
        else:
            return jsonify({'error': 'Failed to delete session from database'}), 500
            
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>/rename', methods=['POST'])
def rename_session(session_id):
    """Rename a session's video title"""
    try:
        # Get session details first
        session = db.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Get new title from request
        data = request.get_json()
        if not data or 'new_title' not in data:
            return jsonify({'error': 'new_title is required in request body'}), 400
        
        new_title = data['new_title'].strip()
        if not new_title:
            return jsonify({'error': 'Title cannot be empty'}), 400
        
        # Validate title length
        if len(new_title) > 200:
            return jsonify({'error': 'Title must be 200 characters or less'}), 400
        
        # Update the session title in database
        success = db.update_session_title(session_id, new_title)
        
        if success:
            logger.info(f"📝 Renamed session {session_id} to: {new_title}")
            return jsonify({
                'success': True,
                'message': 'Session renamed successfully',
                'session_id': session_id,
                'new_title': new_title
            })
        else:
            return jsonify({'error': 'Failed to update session title in database'}), 500
            
    except Exception as e:
        logger.error(f"Error renaming session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

# Reports API Endpoints
@app.route('/api/reports/overview')
def api_reports_overview():
    """Get comprehensive analytics overview for reports page"""
    try:
        # Get date range filter from query params
        date_range = request.args.get('range', '30')  # Default to 30 days
        
        # Calculate date filter
        import time
        from datetime import datetime, timedelta
        
        now = time.time()
        if date_range == 'all':
            start_time = 0
        else:
            days = int(date_range)
            start_time = now - (days * 24 * 60 * 60)
        
        # Get all sessions in date range
        all_sessions = db.get_sessions_with_details(limit=1000)
        filtered_sessions = [s for s in all_sessions if s['start_time'] >= start_time]
        
        # Calculate summary statistics
        total_sessions = len(filtered_sessions)
        total_hotspots = sum(s.get('hotspot_count', 0) for s in filtered_sessions)
        total_duration = sum(s.get('duration', 0) for s in filtered_sessions)
        avg_duration = total_duration / total_sessions if total_sessions > 0 else 0
        avg_events_per_session = total_hotspots / total_sessions if total_sessions > 0 else 0
        
        # Get event type breakdown
        event_types = {}
        session_risk_levels = {'high': 0, 'medium': 0, 'low': 0}
        
        for session in filtered_sessions:
            events = db.get_session_events(session['session_id'])
            
            # Risk level calculation (matching the frontend logic)
            event_count = len(events)
            if event_count > 50:
                session_risk_levels['high'] += 1
            elif event_count > 25:
                session_risk_levels['medium'] += 1
            else:
                session_risk_levels['low'] += 1
            
            # Count event types
            for event in events:
                event_type = event.get('event_type', 'unknown')
                # Normalize event types for better categorization
                if 'phone' in event_type.lower() or 'device' in event_type.lower():
                    category = 'Phone Detection'
                elif 'head' in event_type.lower() or 'looking' in event_type.lower():
                    category = 'Suspicious Looking'
                elif 'leaning' in event_type.lower():
                    category = 'Inappropriate Leaning'
                elif 'gesture' in event_type.lower() or 'hand' in event_type.lower():
                    category = 'Suspicious Gesture'
                else:
                    category = 'Other'
                
                event_types[category] = event_types.get(category, 0) + 1
        
        # Generate timeline data (daily aggregates)
        timeline_data = []
        duration_distribution = {'0-15': 0, '15-30': 0, '30-45': 0, '45-60': 0, '60+': 0}
        
        # Group sessions by day for timeline
        daily_sessions = {}
        daily_events = {}
        daily_risk = {}
        
        for session in filtered_sessions:
            session_date = datetime.fromtimestamp(session['start_time']).strftime('%Y-%m-%d')
            
            # Daily session count
            daily_sessions[session_date] = daily_sessions.get(session_date, 0) + 1
            
            # Daily event count
            daily_events[session_date] = daily_events.get(session_date, 0) + session.get('hotspot_count', 0)
            
            # Daily risk levels
            if session_date not in daily_risk:
                daily_risk[session_date] = {'high': 0, 'medium': 0, 'low': 0}
            
            event_count = session.get('hotspot_count', 0)
            if event_count > 50:
                daily_risk[session_date]['high'] += 1
            elif event_count > 25:
                daily_risk[session_date]['medium'] += 1
            else:
                daily_risk[session_date]['low'] += 1
            
            # Duration distribution
            duration_minutes = session.get('duration', 0) / 60
            if duration_minutes <= 15:
                duration_distribution['0-15'] += 1
            elif duration_minutes <= 30:
                duration_distribution['15-30'] += 1
            elif duration_minutes <= 45:
                duration_distribution['30-45'] += 1
            elif duration_minutes <= 60:
                duration_distribution['45-60'] += 1
            else:
                duration_distribution['60+'] += 1
        
        # Convert daily data to timeline format
        dates = sorted(daily_sessions.keys())[-30:]  # Last 30 days max
        for date in dates:
            timeline_data.append({
                'date': date,
                'sessions': daily_sessions.get(date, 0),
                'events': daily_events.get(date, 0),
                'high_risk': daily_risk.get(date, {}).get('high', 0),
                'medium_risk': daily_risk.get(date, {}).get('medium', 0),
                'low_risk': daily_risk.get(date, {}).get('low', 0)
            })
        
        # Most recent sessions for sidebar
        recent_sessions = filtered_sessions[:20]  # Top 20 recent sessions
        
        return jsonify({
            'success': True,
            'date_range': date_range,
            'summary': {
                'total_sessions': total_sessions,
                'total_hotspots': total_hotspots,
                'avg_duration': avg_duration,
                'avg_events_per_session': avg_events_per_session,
                'total_duration': total_duration
            },
            'event_types': event_types,
            'session_risk_levels': session_risk_levels,
            'timeline_data': timeline_data,
            'duration_distribution': duration_distribution,
            'recent_sessions': recent_sessions
        })
        
    except Exception as e:
        logger.error(f"Error getting reports overview: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/export/pdf', methods=['GET', 'POST'])
def export_reports_pdf():
    """Export comprehensive analytics report as PDF"""
    try:
        if not PDF_AVAILABLE:
            return jsonify({'error': 'PDF export not available - ReportLab not installed'}), 500
        
        # Get date range from request (GET or POST)
        if request.method == 'POST':
            data = request.get_json() or {}
            date_range = data.get('range', '30')
        else:
            date_range = request.args.get('period', '30')
        
        # Get analytics data
        overview_response = api_reports_overview()
        if overview_response.status_code != 200:
            return jsonify({'error': 'Failed to fetch analytics data'}), 500
        
        analytics_data = overview_response.get_json()
        
        # Create PDF report
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        # Create temporary PDF file
        report_filename = f"cheatgpt_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = Path("temp_reports") / report_filename
        report_path.parent.mkdir(exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(str(report_path), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph("CheatGPT Analytics Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Report metadata
        range_text = {
            '7': 'Last 7 days',
            '30': 'Last 30 days', 
            '90': 'Last 90 days',
            '365': 'Last year',
            'all': 'All time'
        }.get(date_range, f'Last {date_range} days')
        
        meta_info = [
            ['Report Period:', range_text],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Total Sessions:', str(analytics_data['summary']['total_sessions'])],
            ['Total Events:', str(analytics_data['summary']['total_hotspots'])],
        ]
        
        meta_table = Table(meta_info, colWidths=[2*inch, 3*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("Report Summary", styles['Heading2']))
        story.append(meta_table)
        story.append(Spacer(1, 12))
        
        # Summary statistics
        summary = analytics_data['summary']
        summary_data = [
            ['Metric', 'Value'],
            ['Total Sessions', str(summary['total_sessions'])],
            ['Total Events Detected', str(summary['total_hotspots'])],
            ['Average Session Duration', f"{summary['avg_duration']/60:.1f} minutes"],
            ['Average Events per Session', f"{summary['avg_events_per_session']:.1f}"],
            ['Total Monitoring Time', f"{summary['total_duration']/3600:.1f} hours"]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("Analytics Summary", styles['Heading2']))
        story.append(summary_table)
        story.append(Spacer(1, 12))
        
        # Event types breakdown
        event_types = analytics_data['event_types']
        if event_types:
            event_data = [['Event Type', 'Count', 'Percentage']]
            total_events = sum(event_types.values())
            
            for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_events * 100) if total_events > 0 else 0
                event_data.append([event_type, str(count), f"{percentage:.1f}%"])
            
            event_table = Table(event_data, colWidths=[2.5*inch, 1*inch, 1*inch])
            event_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("Event Type Breakdown", styles['Heading2']))
            story.append(event_table)
            story.append(Spacer(1, 12))
        
        # Risk level distribution
        risk_levels = analytics_data['session_risk_levels']
        total_risk_sessions = sum(risk_levels.values())
        
        if total_risk_sessions > 0:
            risk_data = [
                ['Risk Level', 'Sessions', 'Percentage'],
                ['High Risk (>50 events)', str(risk_levels['high']), f"{(risk_levels['high']/total_risk_sessions*100):.1f}%"],
                ['Medium Risk (26-50 events)', str(risk_levels['medium']), f"{(risk_levels['medium']/total_risk_sessions*100):.1f}%"],
                ['Low Risk (≤25 events)', str(risk_levels['low']), f"{(risk_levels['low']/total_risk_sessions*100):.1f}%"],
            ]
            
            risk_table = Table(risk_data, colWidths=[2.5*inch, 1*inch, 1*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("Risk Level Distribution", styles['Heading2']))
            story.append(risk_table)
            story.append(Spacer(1, 12))
        
        # Recent sessions
        recent_sessions = analytics_data['recent_sessions'][:10]  # Top 10
        if recent_sessions:
            session_data = [['Session ID', 'Date', 'Duration', 'Events', 'Risk Level']]
            
            for session in recent_sessions:
                date = datetime.fromtimestamp(session['start_time']).strftime('%Y-%m-%d %H:%M')
                duration = f"{session.get('duration', 0)/60:.1f}m"
                events = str(session.get('hotspot_count', 0))
                
                # Determine risk level
                event_count = session.get('hotspot_count', 0)
                if event_count > 50:
                    risk = 'High'
                elif event_count > 25:
                    risk = 'Medium'
                else:
                    risk = 'Low'
                
                session_data.append([
                    session['session_id'][:16] + '...' if len(session['session_id']) > 16 else session['session_id'],
                    date,
                    duration,
                    events,
                    risk
                ])
            
            session_table = Table(session_data, colWidths=[2*inch, 1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch])
            session_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("Recent Sessions", styles['Heading2']))
            story.append(session_table)
        
        # Footer
        story.append(Spacer(1, 24))
        footer_text = f"Report generated by CheatGPT Analytics v3.0 on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        # Send file
        return send_file(
            str(report_path),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"cheatgpt_analytics_{date_range}days_{datetime.now().strftime('%Y%m%d')}.pdf"
        )
        
    except Exception as e:
        logger.error(f"Error exporting PDF report: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/export/csv', methods=['GET', 'POST'])
def export_reports_csv():
    """Export analytics data as CSV"""
    try:
        import csv
        from io import StringIO
        
        # Get date range from request (GET or POST)
        if request.method == 'POST':
            data = request.get_json() or {}
            date_range = data.get('range', '30')
        else:
            date_range = request.args.get('period', '30')
        
        # Get analytics data
        overview_response = api_reports_overview()
        if overview_response.status_code != 200:
            return jsonify({'error': 'Failed to fetch analytics data'}), 500
        
        analytics_data = overview_response.get_json()
        
        # Create CSV content
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Session ID', 'Start Time', 'Duration (minutes)', 'Event Count', 'Risk Level', 'Status'])
        
        # Write session data
        for session in analytics_data['recent_sessions']:
            start_time = datetime.fromtimestamp(session['start_time']).strftime('%Y-%m-%d %H:%M:%S')
            duration_minutes = round(session.get('duration', 0) / 60, 1)
            event_count = session.get('hotspot_count', 0)
            
            # Determine risk level
            if event_count > 50:
                risk_level = 'High'
            elif event_count > 25:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            writer.writerow([
                session['session_id'],
                start_time,
                duration_minutes,
                event_count,
                risk_level,
                session.get('status', 'completed')
            ])
        
        # Create file response
        output.seek(0)
        
        response = Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=cheatgpt_sessions_{date_range}days_{datetime.now().strftime("%Y%m%d")}.csv'
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error exporting CSV report: {e}")
        return jsonify({'error': str(e)}), 500

# Session-specific Reports API Endpoints
@app.route('/api/reports/session/<session_id>')
def api_session_analytics(session_id):
    """Get analytics data for a specific session"""
    try:
        # Get session by session_id string first
        session = db.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Get session analytics using the session_id string
        analytics_data = db.get_session_analytics_by_session_id(session_id)
        if not analytics_data:
            return jsonify({'error': 'No analytics data found for session'}), 404
        
        return jsonify({
            'success': True,
            **analytics_data
        })
        
    except Exception as e:
        logger.error(f"Error getting session report for {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/export/session_pdf')
def export_session_pdf():
    """Export individual session report as PDF"""
    try:
        if not PDF_AVAILABLE:
            return jsonify({'error': 'PDF export not available - ReportLab not installed'}), 500
        
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Get session analytics
        analytics_data = db.get_session_analytics_by_session_id(session_id)
        if not analytics_data:
            return jsonify({'error': 'Session not found'}), 404
        
        # Create PDF report
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        # Create temporary PDF file
        report_filename = f"cheatgpt_session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = Path("temp_reports") / report_filename
        report_path.parent.mkdir(exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(str(report_path), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        session_info = analytics_data['session_info']
        title = Paragraph(f"Session Report: {session_info.get('video_title', session_info['session_id'])}", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Session metadata
        # Handle the date field properly - use created_at and parse as string
        created_at = session_info.get('created_at', session_info.get('started_at', 'Unknown'))
        if created_at and created_at != 'Unknown':
            try:
                # Parse the string date format: "2025-09-14 05:52:45"
                start_time = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, AttributeError):
                start_time = str(created_at)
        else:
            start_time = 'Unknown'
        
        meta_info = [
            ['Session ID:', session_info['session_id']],
            ['Start Time:', start_time],
            ['Duration:', f"{analytics_data['summary']['duration_minutes']} minutes"],
            ['Total Events:', str(analytics_data['summary']['total_events'])],
            ['Events per Minute:', str(analytics_data['summary']['events_per_minute'])],
            ['Status:', session_info['status'].title()]
        ]
        
        meta_table = Table(meta_info, colWidths=[2*inch, 3*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.lightgrey),
            ('TEXTCOLOR',(0,0),(-1,-1),colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 24))
        
        # Event Types Summary
        story.append(Paragraph("Event Types Distribution", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        event_data = [['Event Type', 'Count']]
        for event_type, count in analytics_data['event_types'].items():
            event_data.append([event_type, str(count)])
        
        if len(event_data) > 1:
            event_table = Table(event_data, colWidths=[3*inch, 1*inch])
            event_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 12),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID',(0,0),(-1,-1),1,colors.black)
            ]))
            story.append(event_table)
        else:
            story.append(Paragraph("No events detected in this session.", styles['Normal']))
        
        story.append(Spacer(1, 24))
        
        # Confidence Distribution
        story.append(Paragraph("Confidence Distribution", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        conf_dist = analytics_data['confidence_distribution']
        conf_data = [
            ['Confidence Level', 'Count'],
            ['High (90%+)', str(conf_dist.get('high', 0))],
            ['Medium (70-89%)', str(conf_dist.get('medium', 0))],
            ['Low (<70%)', str(conf_dist.get('low', 0))]
        ]
        
        conf_table = Table(conf_data, colWidths=[3*inch, 1*inch])
        conf_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID',(0,0),(-1,-1),1,colors.black)
        ]))
        story.append(conf_table)
        story.append(Spacer(1, 24))
        
        # Timeline Summary
        story.append(Paragraph("Timeline Analysis", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        timeline_data = analytics_data.get('timeline', [])
        if timeline_data:
            timeline_table_data = [['Time (Minutes)', 'Events']]
            for item in timeline_data:
                timeline_table_data.append([f"{item['minute']}m", str(item['events'])])
            
            timeline_table = Table(timeline_table_data, colWidths=[2*inch, 1*inch])
            timeline_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 12),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID',(0,0),(-1,-1),1,colors.black)
            ]))
            story.append(timeline_table)
        else:
            story.append(Paragraph("No timeline data available.", styles['Normal']))
        
        story.append(Spacer(1, 24))
        
        # Detailed Events List
        story.append(Paragraph("Detailed Events", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        try:
            # Get detailed events
            events_response = db.get_session_events(session_id)
            if events_response and len(events_response) > 0:
                events_table_data = [['Time', 'Event Type', 'Confidence']]
                for event in events_response[:20]:  # Limit to first 20 events
                    time_str = event.get('formatted_time', f"{event.get('timestamp_seconds', 0):.1f}s")
                    event_type = event.get('event_type', 'Unknown')
                    confidence = f"{(event.get('confidence', 0) * 100):.1f}%"
                    events_table_data.append([time_str, event_type, confidence])
                
                if len(events_response) > 20:
                    events_table_data.append(['...', f'and {len(events_response) - 20} more events', ''])
                
                events_table = Table(events_table_data, colWidths=[1*inch, 3*inch, 1*inch])
                events_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 12),
                    ('FONTSIZE', (0,1), (-1,-1), 10),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                    ('GRID',(0,0),(-1,-1),1,colors.black)
                ]))
                story.append(events_table)
            else:
                story.append(Paragraph("No detailed events available.", styles['Normal']))
        except Exception as events_error:
            story.append(Paragraph(f"Error loading detailed events: {str(events_error)}", styles['Normal']))
        
        story.append(Spacer(1, 24))
        
        # Summary Statistics
        story.append(Paragraph("Summary Statistics", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        summary = analytics_data.get('summary', {})
        summary_data = [
            ['Metric', 'Value'],
            ['Total Events', str(summary.get('total_events', 0))],
            ['Session Duration', f"{summary.get('duration_minutes', 0):.1f} minutes"],
            ['Events per Minute', f"{summary.get('events_per_minute', 0):.2f}"],
            ['Session Date', summary.get('session_date', 'Unknown')]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID',(0,0),(-1,-1),1,colors.black)
        ]))
        story.append(summary_table)
        
        # Footer
        story.append(Spacer(1, 24))
        story.append(Paragraph(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph("CheatGPT Analytics - Session Analysis Report", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        return send_file(
            str(report_path),
            as_attachment=True,
            download_name=f"cheatgpt_session_{session_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
        )
        
    except Exception as e:
        logger.error(f"Error exporting session PDF report: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/export/session_csv')
def export_session_csv():
    """Export individual session data as CSV"""
    try:
        import csv
        from io import StringIO
        
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Get session events
        session = db.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        events = db.get_session_events(session_id)
        
        # Create CSV content
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Timestamp', 'Event Type', 'Confidence', 'Severity', 'Frame Number'])
        
        # Write event data
        for event in events:
            writer.writerow([
                f"{event['timestamp_seconds']:.2f}s",
                event['event_type'],
                f"{event['confidence']:.2%}",
                event['severity'],
                event.get('frame_no', '')
            ])
        
        # Create file response
        output.seek(0)
        
        response = Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=cheatgpt_session_{session_id}_{datetime.now().strftime("%Y%m%d")}.csv'
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error exporting session CSV: {e}")
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
        
        # Start detection engine session with frame size (matching demo pattern)
        if detection_engine and hasattr(detection_engine, 'start_session'):
            try:
                # Use actual camera resolution like in demo
                engine_session_id = detection_engine.start_session("web_camera", (640, 480))
                logger.info(f"🎯 Started EngineHybrid session: {engine_session_id}")
            except Exception as e:
                logger.warning(f"Failed to start engine session: {e}")
        
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
        
        # Stop detection engine session and get statistics (matching demo)
        if detection_engine and hasattr(detection_engine, 'stop_session'):
            try:
                engine_stats = detection_engine.stop_session()
                logger.info(f"🎯 Stopped EngineHybrid session: {engine_stats}")
                
                # Get detailed statistics like in demo
                if hasattr(detection_engine, 'get_statistics'):
                    detailed_stats = detection_engine.get_statistics()
                    logger.info(f"📈 Engine Performance: {detailed_stats['performance']['avg_fps']:.1f} FPS, "
                              f"Detection: {detailed_stats['performance']['avg_detection_time_ms']:.1f}ms, "
                              f"Active persons: {detailed_stats['rule_engine']['active_persons']}")
            except Exception as e:
                logger.warning(f"Failed to stop engine session: {e}")
        
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
        
        # Initialize camera with optimized settings (matching demo)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        # Set camera properties for enhanced HD resolution
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Enhanced from 640 to 1280 (HD)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Enhanced from 480 to 720 (HD)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer to reduce latency
        
        # Verify actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"📹 Camera initialized: {actual_width}x{actual_height} @ 30 FPS")
        
        # Use enhanced HD resolution for video recording
        recording_width = 1280  # Enhanced from 640 to 1280
        recording_height = 720   # Enhanced from 480 to 720
        
        # Initialize video recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"session_{current_session_id}_{timestamp}.mp4"
        video_path = RECORDINGS_DIR / recording_filename
        
        # Initialize video writer with robust codec fallback system
        recording_fps = 15.0  # Target video playback FPS for smooth recorded video
        
        # Codec preference order: prioritize H.264 AVC1 for best web compatibility
        codec_options = [
            ('avc1', '.mp4', 'H.264 AVC1'),  # Best web compatibility, modern browsers prefer this
            ('H264', '.mp4', 'H.264'),       # Alternative H.264 fourcc code
            ('mp4v', '.mp4', 'MPEG-4'),      # Fallback for older systems
            ('XVID', '.avi', 'Xvid'),        # Very reliable fallback
            ('MJPG', '.avi', 'Motion JPEG'), # Always works but larger files
        ]
        
        video_writer = None
        final_video_path = None
        
        for codec, ext, name in codec_options:
            try:
                # Update path extension based on codec
                test_path = str(video_path).replace('.mp4', ext)
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(test_path, fourcc, recording_fps, (recording_width, recording_height))
                
                if video_writer.isOpened():
                    final_video_path = Path(test_path)
                    recording_filename = final_video_path.name  # Update global filename
                    logger.info(f"✅ Video recording initialized with {name} codec: {final_video_path.name}")
                    break
                else:
                    video_writer.release()
                    video_writer = None
                    
            except Exception as e:
                logger.warning(f"❌ {name} codec failed: {e}")
                if video_writer:
                    video_writer.release()
                    video_writer = None
                continue
        
        if not video_writer or not video_writer.isOpened():
            logger.error("Failed to initialize video writer with any codec")
            return jsonify({"success": False, "error": "Video recording initialization failed"})
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
            
            # Process frame with hybrid engine (same as demo)
            if detection_engine and hasattr(detection_engine, 'process_frame'):
                try:
                    # Direct processing like in demo - engine handles timing internally
                    overlay_frame, events = detection_engine.process_frame(frame, "webcam", current_time)
                    
                    if overlay_frame is not None:
                        display_frame = overlay_frame  # Use engine's optimized overlay directly
                        
                        # Handle events (same format as demo)
                        if events:
                            logger.info(f"🔍 Detection: {len(events)} events found")
                            
                            # Log each event with severity emoji (matching demo style)
                            for event in events:
                                severity_emoji = {
                                    'red': '🚨',
                                    'orange': '⚠️', 
                                    'yellow': '💛'
                                }.get(event.get('severity', 'yellow'), '📊')
                                
                                event_type = event.get('event_type', 'Unknown')
                                person_id = event.get('person_id', 'Unknown')
                                confidence = event.get('confidence', 0) * 100
                                details = event.get('details', '')
                                
                                logger.info(f"   {severity_emoji} {person_id}: {event_type} (confidence: {confidence:.1f}%)")
                                if details:
                                    logger.info(f"      Details: {details}")
                            
                            # Buffer events for database storage
                            timestamp_offset = current_time - session_start_time if session_start_time else 0
                            for event in events:
                                buffer_event_for_database(event, current_time, timestamp_offset)
                            
                            # Update session stats
                            session_stats['hotspot_count'] += len(events)
                            emit_status_update()
                    else:
                        display_frame = frame  # Fallback to original frame
                        
                except Exception as e:
                    logger.error(f"Error processing frame through hybrid engine: {e}")
                    display_frame = frame
            else:
                display_frame = frame
            
            # Check if we should flush buffered events to database
            if should_flush_events(current_time):
                flush_events_to_database()
                global last_flush_time
                last_flush_time = current_time
            
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
            if current_time - last_fps_time >= 10.0:  # Log every 10 seconds (reduced frequency)
                actual_fps = frame_count / (current_time - (session_start_time or current_time))
                session_stats['fps'] = actual_fps
                logger.debug(f"📊 Performance: {actual_fps:.1f} FPS, Queue size: {frame_queue.qsize()}")
                last_fps_time = current_time
            
            # Send status updates every 0.5 seconds for real-time UI updates
            if current_time - last_status_update >= 0.5:  # Update every 0.5 seconds
                # Update FPS for real-time display
                if current_time != (session_start_time or current_time):
                    session_stats['fps'] = frame_count / (current_time - (session_start_time or current_time))
                emit_status_update()
                last_status_update = current_time
                # Remove the frequent debug log for cleaner output
            
            # Frame rate control - optimized for 30 FPS streaming (matching hybrid engine)
            # The hybrid engine separates 30 FPS streaming from 10 FPS detection internally
            time.sleep(0.033)  # ~30 FPS for smooth real-time streaming
            
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

# Video Processing API Routes
@app.route('/api/upload', methods=['POST'])
def api_upload_video():
    """Upload video file for processing"""
    try:
        logger.info("📤 Video upload request received")
        logger.info(f"VIDEO_PROCESSING_AVAILABLE: {VIDEO_PROCESSING_AVAILABLE}")
        logger.info(f"upload_handler: {upload_handler}")
        
        if not VIDEO_PROCESSING_AVAILABLE:
            logger.error("❌ Video processing not available")
            return jsonify({'error': 'Video processing not available'}), 500
        
        if 'video' not in request.files:
            logger.error("❌ No video file in request")
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        logger.info(f"📁 File received: {file.filename}")
        
        if file.filename == '':
            logger.error("❌ Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate session ID
        session_id = f"single_{int(time.time())}"
        logger.info(f"🆔 Generated session ID: {session_id}")
        
        # Upload file
        logger.info("📤 Starting file upload...")
        result = upload_handler.upload_single_video(file, session_id)
        logger.info(f"📤 Upload result: {result}")
        
        if 'error' in result:
            logger.error(f"❌ Upload error: {result}")
            return jsonify(result), 400
        
        # Create database session for uploaded video
        main_db.create_uploaded_video_session(
            session_id=session_id,
            original_filename=file.filename,
            video_path=result['file_path']
        )
        
        # Start video processing automatically
        logger.info("🎬 Starting automatic video processing...")
        try:
            # Create processing task
            task = ProcessingTask(
                session_id=session_id,
                file_path=result['file_path']
            )
            
            # Add to processing status dictionary
            processing_status[session_id] = task
            
            # Start processing in background
            import threading
            thread = threading.Thread(target=process_video_async, args=(task,))
            thread.daemon = True
            thread.start()
            
            logger.info(f"✅ Upload successful and processing started for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start processing: {e}")
            # Still return success for upload, but note processing failed to start
            return jsonify({
                'success': True,
                'session_id': session_id,
                'path': result['file_path'],
                'file_path': result['file_path'],
                'message': 'Upload successful, but processing failed to start automatically',
                'processing_error': str(e)
            })
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'path': result['file_path'],
            'file_path': result['file_path'],
            'message': 'Upload successful, processing started'
        })
        
    except Exception as e:
        logger.error(f"❌ Error uploading video: {e}")
        logger.exception("Full error details:")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def api_start_processing():
    """Start video processing"""
    try:
        logger.info("🔄 Video processing request received")
        
        if not VIDEO_PROCESSING_AVAILABLE:
            logger.error("❌ Video processing not available")
            return jsonify({'success': False, 'error': 'Video processing not available'}), 500
        
        data = request.get_json()
        logger.info(f"📋 Processing request data: {data}")
        
        # Handle both frontend (video_path) and backend (file_path, session_id) formats
        video_path = data.get('video_path') or data.get('file_path')
        session_id = data.get('session_id')
        
        # If no session_id provided, extract from video_path or generate one
        if not session_id and video_path:
            # Extract session_id from path like: uploads/single_123456/filename.mp4
            import os
            path_parts = os.path.normpath(video_path).split(os.sep)
            if len(path_parts) >= 2 and path_parts[-2].startswith('single_'):
                session_id = path_parts[-2]
            else:
                session_id = f"single_{int(time.time())}"
        
        if not video_path:
            logger.error("❌ Missing video_path")
            return jsonify({'success': False, 'error': 'Missing video_path'}), 400
        
        logger.info(f"🎬 Processing video: {video_path} for session: {session_id}")
        
        # Create processing task
        task = ProcessingTask(session_id, video_path)
        processing_status[session_id] = task
        
        # Start processing in background
        thread = threading.Thread(target=process_video_async, args=(task,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"✅ Processing started for session: {session_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'status': 'started',
            'message': 'Processing started'
        })
        
    except Exception as e:
        logger.error(f"❌ Error starting video processing: {e}")
        logger.exception("Full error details:")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status/<session_id>', methods=['GET'])
def api_processing_status(session_id):
    """Get processing status"""
    try:
        logger.info(f"📊 Status request for session: {session_id}")
        
        if session_id not in processing_status:
            logger.error(f"❌ Session not found: {session_id}")
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        task = processing_status[session_id]
        
        # Determine if processing is completed
        completed = task.status in ['completed', 'error', 'stopped']
        
        # Prepare result object for completed tasks
        result = None
        if completed:
            if task.status == 'completed':
                result = {'success': True, 'data': task.result}
            else:
                result = {'success': False, 'error': task.error}
        
        status_response = {
            'success': True,
            'session_id': session_id,
            'status': {
                'progress': task.progress,
                'message': task.message,
                'completed': completed,
                'result': result,
                'processing_time': time.time() - task.start_time
            }
        }
        
        logger.info(f"📊 Status response: {status_response}")
        return jsonify(status_response)
        
    except Exception as e:
        logger.error(f"❌ Error getting processing status: {e}")
        logger.exception("Full error details:")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop/<session_id>', methods=['POST'])
def api_stop_processing(session_id):
    """Stop video processing for a given session"""
    try:
        logger.info(f"🛑 Stop processing request for session: {session_id}")
        
        if session_id not in processing_status:
            logger.error(f"❌ Session not found: {session_id}")
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        task = processing_status[session_id]
        
        # Check if already completed or stopped
        if task.status in ['completed', 'error', 'stopped']:
            logger.warning(f"⚠️ Session {session_id} already finished with status: {task.status}")
            return jsonify({
                'success': True,
                'message': f'Session already finished with status: {task.status}'
            })
        
        # Mark the task as stopped
        task.status = 'stopped'
        task.message = 'Processing stopped by user'
        task.error = 'Processing cancelled by user request'
        
        logger.info(f"✅ Processing stopped for session: {session_id}")
        
        return jsonify({
            'success': True,
            'message': 'Processing stopped successfully',
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"❌ Error stopping processing for session {session_id}: {e}")
        logger.exception("Full error details:")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download/<file_type>/<session_id>', methods=['GET'])
def api_download_file(file_type, session_id):
    """Download processed files"""
    try:
        if session_id not in processing_status:
            return jsonify({'error': 'Session not found'}), 404
        
        task = processing_status[session_id]
        
        if task.status != 'completed':
            return jsonify({'error': 'Processing not completed'}), 400
        
        result = task.result
        
        if file_type == 'video':
            if 'output_paths' in result and 'processed_video' in result['output_paths']:
                return send_file(result['output_paths']['processed_video'], as_attachment=True)
        elif file_type == 'report':
            if 'output_paths' in result and 'json_report' in result['output_paths']:
                return send_file(result['output_paths']['json_report'], as_attachment=True)
        elif file_type == 'visualization':
            if 'output_paths' in result and 'visualizations' in result['output_paths']:
                # Create a zip file with all visualizations
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    viz_paths = result['output_paths']['visualizations']
                    for viz_name, viz_path in viz_paths.items():
                        if os.path.exists(viz_path):
                            zip_file.write(viz_path, f"{viz_name}.png")
                
                zip_buffer.seek(0)
                
                return send_file(
                    BytesIO(zip_buffer.read()),
                    as_attachment=True,
                    download_name=f"visualizations_{session_id}.zip",
                    mimetype='application/zip'
                )
        
        return jsonify({'error': f'{file_type} not available'}), 404
        
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return jsonify({'error': str(e)}), 500

# SocketIO Events
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
    
    # Set up clean logging before starting the server
    import os
    os.environ['WERKZEUG_RUN_MAIN'] = 'true'  # Suppress werkzeug startup message
    
    # Disable all werkzeug logging
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.handlers.clear()
    werkzeug_logger.propagate = False
    werkzeug_logger.disabled = True
    
    # Run the app with minimal logging
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, use_reloader=False)
