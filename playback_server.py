#!/usr/bin/env python3
"""
Simple Flask server for CheatGPT session playback.
"""

import os
import json
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
import logging

# Paths
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "cheatgpt" / "templates"
STATIC_DIR = BASE_DIR / "cheatgpt" / "static"
RECORDINGS_DIR = BASE_DIR / "recordings"
VIDEOS_DIR = BASE_DIR / "videos"
EVENTS_DB = BASE_DIR / "data" / "events.db"

app = Flask(__name__, 
           template_folder=str(TEMPLATE_DIR),
           static_folder=str(STATIC_DIR))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Jinja2 filters
@app.template_filter('datetime_format')
def datetime_format(timestamp):
    """Format Unix timestamp to readable datetime"""
    try:
        dt = datetime.fromtimestamp(float(timestamp))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(timestamp)

@app.template_filter('tojsonfilter')
def to_json_filter(obj):
    """Convert object to JSON string"""
    return json.dumps(obj)

@app.route('/')
def index():
    """List all available sessions"""
    sessions = []
    
    # Get timeline files
    timeline_files = list(RECORDINGS_DIR.glob("timeline_*.json"))
    
    for timeline_file in timeline_files:
        try:
            with open(timeline_file, 'r') as f:
                timeline_data = json.load(f)
            
            session_name = timeline_file.stem.replace("timeline_", "")
            
            # Look for corresponding video file
            video_file = None
            for video_path in VIDEOS_DIR.glob(f"*{session_name}*.mp4"):
                video_file = video_path.name
                break
            
            if not video_file:
                # Try recordings directory
                for video_path in RECORDINGS_DIR.glob(f"{session_name}.mp4"):
                    video_file = f"recordings/{video_path.name}"
                    break
            
            sessions.append({
                'session_id': session_name,
                'timeline_file': timeline_file.name,
                'video_file': video_file,
                'duration': timeline_data.get('session_info', {}).get('duration', 0),
                'total_events': timeline_data.get('session_info', {}).get('total_events', 0),
                'start_time': timeline_data.get('session_info', {}).get('start_time', 0)
            })
            
        except Exception as e:
            logger.error(f"Error processing {timeline_file}: {e}")
    
    return render_template('session_list.html', sessions=sessions)

@app.route('/playback/<session_id>')
def playback(session_id):
    """Session playback page"""
    try:
        # Load timeline data
        timeline_file = RECORDINGS_DIR / f"timeline_{session_id}.json"
        with open(timeline_file, 'r') as f:
            timeline_data = json.load(f)
        
        # Find video file
        video_path = None
        for video_file in VIDEOS_DIR.glob(f"*{session_id}*.mp4"):
            video_path = f"/videos/{video_file.name}"
            break
        
        if not video_path:
            for video_file in RECORDINGS_DIR.glob(f"{session_id}.mp4"):
                video_path = f"/recordings/{video_file.name}"
                break
        
        if not video_path:
            return "Video file not found", 404
        
        session_info = timeline_data.get('session_info', {})
        events = timeline_data.get('events', [])
        
        # Format duration
        duration = session_info.get('duration', 0)
        duration_formatted = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
        
        return render_template('session_playback.html',
            session_id=session_id,
            video_path=video_path,
            duration=duration,
            duration_formatted=duration_formatted,
            start_time=session_info.get('start_time', 0),
            total_events=session_info.get('total_events', 0),
            events=events
        )
        
    except FileNotFoundError:
        return f"Session {session_id} not found", 404
    except Exception as e:
        logger.error(f"Error loading session {session_id}: {e}")
        return "Internal server error", 500

@app.route('/api/sessions')
def api_sessions():
    """API endpoint for session list"""
    sessions = []
    timeline_files = list(RECORDINGS_DIR.glob("timeline_*.json"))
    
    for timeline_file in timeline_files:
        try:
            with open(timeline_file, 'r') as f:
                timeline_data = json.load(f)
            
            session_name = timeline_file.stem.replace("timeline_", "")
            sessions.append({
                'session_id': session_name,
                'duration': timeline_data.get('session_info', {}).get('duration', 0),
                'total_events': timeline_data.get('session_info', {}).get('total_events', 0),
                'start_time': timeline_data.get('session_info', {}).get('start_time', 0)
            })
        except Exception as e:
            logger.error(f"Error processing {timeline_file}: {e}")
    
    return jsonify(sessions)

@app.route('/api/session/<session_id>')
def api_session(session_id):
    """API endpoint for specific session data"""
    try:
        timeline_file = RECORDINGS_DIR / f"timeline_{session_id}.json"
        with open(timeline_file, 'r') as f:
            timeline_data = json.load(f)
        return jsonify(timeline_data)
    except FileNotFoundError:
        return jsonify({'error': 'Session not found'}), 404

# Static file serving
@app.route('/recordings/<filename>')
def serve_recording(filename):
    """Serve recording files"""
    return send_from_directory(RECORDINGS_DIR, filename)

@app.route('/videos/<filename>')
def serve_video(filename):
    """Serve video files"""
    return send_from_directory(VIDEOS_DIR, filename)

@app.route('/timeline/<filename>')
def serve_timeline(filename):
    """Serve timeline files"""
    return send_from_directory(RECORDINGS_DIR, filename)

if __name__ == '__main__':
    # Create directories if they don't exist
    RECORDINGS_DIR.mkdir(exist_ok=True)
    VIDEOS_DIR.mkdir(exist_ok=True)
    (BASE_DIR / "data").mkdir(exist_ok=True)
    
    logger.info("🚀 Starting CheatGPT Playback Server...")
    logger.info(f"📁 Recordings directory: {RECORDINGS_DIR}")
    logger.info(f"📁 Videos directory: {VIDEOS_DIR}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
