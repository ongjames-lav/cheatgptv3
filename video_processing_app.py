"""
Simplified CheatGPT3 Video Processing Web Application
Single video upload and processing only (batch processing removed)
"""

import os
import time
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string
from werkzeug.utils import secure_filename
import uuid
import threading
from typing import Dict, Optional, List

# Import CheatGPT components
from cheatgpt.video_processor import VideoProcessor
from cheatgpt.upload_handler import UploadHandler
from cheatgpt.report_generator import ReportGenerator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Global state management
processing_status = {}
upload_handler = UploadHandler()
video_processor = VideoProcessor()
report_generator = ReportGenerator()

# Configuration
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

class ProcessingTask:
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
        
        # Process the video
        result = video_processor.process_video(
            input_path=task.file_path,
            session_id=task.session_id
        )
        
        task.progress = 90
        task.message = "Generating reports..."
        
        # Update final results
        task.result = result
        task.status = "completed"
        task.message = "Processing completed successfully"
        task.progress = 100
        
        print(f"✅ Video processing completed for session {task.session_id}")
        
    except Exception as e:
        task.status = "error"
        task.error = str(e)
        task.message = f"Error: {str(e)}"
        task.progress = 0
        print(f"❌ Video processing failed for session {task.session_id}: {e}")

@app.route('/')
def index():
    """Main page with drag-drop upload"""
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CheatGPT3 Video Processing</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: #0f0f0f;
            color: #ffffff;
            font-family: 'Roboto', 'Arial', sans-serif;
            overflow-x: hidden;
        }

        /* Header */
        .header {
            background: #212121;
            padding: 12px 0;
            border-bottom: 1px solid #303030;
            position: sticky;
            top: 0;
            z-index: 1000;
        }

        .header-content {
            max-width: 1900px;
            width: 100%;
            margin: 0 auto;
            padding: 0 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .logo {
            font-size: 20px;
            font-weight: 500;
            color: #ff0000;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .logo img {
            height: 50px;
            width: auto;
        }

        .search-container {
            flex: 1;
            max-width: 600px;
            margin: 0 40px;
            position: relative;
        }

        .search-bar {
            width: 100%;
            background: #121212;
            border: 1px solid #303030;
            color: #ffffff;
            padding: 10px 44px 10px 16px;
            border-radius: 20px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.2s;
        }

        .search-bar:focus {
            border-color: #065fd4;
            box-shadow: 0 0 0 1px #065fd4;
        }

        .search-bar::placeholder {
            color: #aaaaaa;
        }

        .search-btn {
            position: absolute;
            right: 4px;
            top: 50%;
            transform: translateY(-50%);
            background: #313131;
            border: none;
            color: #ffffff;
            padding: 6px 8px;
            border-radius: 14px;
            cursor: pointer;
            transition: background 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
        }

        .search-btn:hover {
            background: #404040;
        }

        .user-menu {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .nav-item {
            color: #ffffff;
            text-decoration: none;
            padding: 12px;
            border-radius: 8px;
            transition: background 0.2s;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            width: 40px;
            height: 40px;
        }

        .nav-item:hover {
            background: #303030;
        }

        .nav-item.active {
            background: #ff0000;
        }

        .nav-item i {
            font-size: 16px;
        }

        /* Tooltip styles */
        .nav-item .tooltip {
            visibility: hidden;
            opacity: 0;
            background-color: #1a1a1a;
            color: #ffffff;
            text-align: center;
            border-radius: 6px;
            padding: 8px 12px;
            position: absolute;
            z-index: 1001;
            top: 130%;
            left: 50%;
            transform: translateX(-50%);
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s ease;
            white-space: nowrap;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            border: 1px solid #404040;
            pointer-events: none;
        }

        .nav-item .tooltip::after {
            content: "";
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            border-width: 6px;
            border-style: solid;
            border-color: transparent transparent #1a1a1a transparent;
        }

        .nav-item .tooltip::before {
            content: "";
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            border-width: 7px;
            border-style: solid;
            border-color: transparent transparent #404040 transparent;
            z-index: -1;
        }

        .nav-item:hover .tooltip {
            visibility: visible;
            opacity: 1;
            transform: translateX(-50%) translateY(2px);
        }

        /* Main Layout */
        .main-layout {
            display: block;
            max-width: 1900px;
            width: 100%;
            margin: 0 auto;
            min-height: calc(100vh - 57px);
        }

        /* Content Area */
        .content {
            width: 100%;
            padding: 24px 32px;
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Upload Section */
        .upload-section {
            margin-bottom: 32px;
        }

        .section-title {
            font-size: 24px;
            font-weight: 500;
            color: #ffffff;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .upload-card {
            background: #181818;
            border-radius: 12px;
            padding: 32px;
            border: 1px solid #303030;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .upload-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }

        /* Upload Area */
        .upload-area {
            border: 2px dashed #404040;
            border-radius: 12px;
            padding: 48px 32px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #121212;
        }

        .upload-area:hover, .upload-area.drag-over {
            border-color: #ff0000;
            background: #1a1111;
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 64px;
            color: #ff0000;
            margin-bottom: 24px;
        }

        .upload-title {
            font-size: 24px;
            font-weight: 500;
            color: #ffffff;
            margin-bottom: 12px;
        }

        .upload-subtitle {
            font-size: 16px;
            color: #aaaaaa;
            margin-bottom: 24px;
        }

        .file-input {
            display: none;
        }

        .upload-browse-btn {
            background: #ff0000;
            color: #ffffff;
            border: none;
            padding: 12px 24px;
            border-radius: 20px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 24px;
        }

        .upload-browse-btn:hover {
            background: #cc0000;
        }

        .upload-info {
            color: #aaaaaa;
            font-size: 14px;
        }

        /* Progress Section */
        .progress-section {
            text-align: center;
            padding: 32px;
        }

        .progress-title {
            font-size: 20px;
            font-weight: 500;
            color: #ffffff;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: #303030;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 16px;
        }

        .progress-fill {
            height: 100%;
            background: #ff0000;
            border-radius: 4px;
            width: 0%;
            transition: width 0.3s ease;
        }

        .progress-animated {
            background: linear-gradient(90deg, #ff0000, #ff4444, #ff0000);
            background-size: 200% 100%;
            animation: progressShine 2s linear infinite;
        }

        @keyframes progressShine {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        .progress-text {
            color: #aaaaaa;
            font-size: 16px;
            margin: 0;
        }

        /* Results Section */
        .results-section {
            text-align: center;
            padding: 32px;
        }

        .results-header {
            margin-bottom: 32px;
        }

        .results-title {
            font-size: 20px;
            font-weight: 500;
            color: #10b981;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }

        .results-actions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
            max-width: 800px;
            margin: 0 auto;
        }

        .result-btn {
            padding: 12px 20px;
            border: none;
            border-radius: 20px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            text-decoration: none;
        }

        .result-btn.primary {
            background: #ff0000;
            color: #ffffff;
        }

        .result-btn.primary:hover {
            background: #cc0000;
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(255, 0, 0, 0.3);
        }

        .result-btn.secondary {
            background: #313131;
            color: #ffffff;
        }

        .result-btn.secondary:hover {
            background: #404040;
            transform: translateY(-2px);
        }

        .result-btn.tertiary {
            background: #065fd4;
            color: #ffffff;
        }

        .result-btn.tertiary:hover {
            background: #0c4ba6;
            transform: translateY(-2px);
        }

        /* Error Section */
        .error-section {
            text-align: center;
            padding: 32px;
        }

        .error-header {
            margin-bottom: 16px;
        }

        .error-title {
            font-size: 20px;
            font-weight: 500;
            color: #ff4444;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }

        .error-text {
            color: #ff8888;
            font-size: 16px;
            margin: 0;
            line-height: 1.5;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .content {
                padding: 16px;
            }
            
            .upload-card {
                padding: 24px;
            }
            
            .upload-area {
                padding: 32px 16px;
            }
            
            .upload-icon {
                font-size: 48px;
            }
            
            .upload-title {
                font-size: 20px;
            }
            
            .results-actions {
                grid-template-columns: 1fr;
                gap: 12px;
            }
            
            .header-content {
                padding: 0 16px;
            }
            
            .search-container {
                margin: 0 12px;
            }
            
            .user-menu {
                gap: 8px;
            }
            
            .nav-item {
                padding: 6px 8px;
                font-size: 12px;
            }
        }

        @media (max-width: 480px) {
            .section-title {
                font-size: 20px;
            }
            
            .upload-title {
                font-size: 18px;
            }
            
            .upload-subtitle {
                font-size: 14px;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="header-content">
            <a href="/analytics/home" class="logo">
                <i class="fas fa-video" style="color: #ff0000;"></i>
                CheatGPT3
            </a>
            
            <div class="search-container">
                <input type="text" class="search-bar" id="searchInput" 
                       placeholder="Search videos">
                <button class="search-btn" id="searchBtn">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
                    </svg>
                </button>
            </div>
            
            <nav class="user-menu">
                <a href="/analytics/home" class="nav-item">
                    <i class="fas fa-home"></i>
                    <span class="tooltip">Home</span>
                </a>
                <a href="/" class="nav-item">
                    <i class="fas fa-video"></i>
                    <span class="tooltip">Live</span>
                </a>
                <a href="/analytics/player" class="nav-item">
                    <i class="fas fa-play"></i>
                    <span class="tooltip">Player</span>
                </a>
                <a href="/analytics/reports" class="nav-item">
                    <i class="fas fa-chart-bar"></i>
                    <span class="tooltip">Reports</span>
                </a>
                <a href="#" class="nav-item active">
                    <i class="fas fa-upload"></i>
                    <span class="tooltip">Upload</span>
                </a>
            </nav>
        </div>
    </header>

    <!-- Main Layout -->
    <div class="main-layout">
        <!-- Content -->
        <main class="content">
            <!-- Upload Section -->
            <section class="upload-section">
                <div class="section-title">
                    <i class="fas fa-cloud-upload-alt"></i> Upload Video for Processing
                </div>
                
                <div class="upload-card">
                    <div id="dragDropArea" class="upload-area">
                        <div class="upload-icon">
                            <i class="fas fa-cloud-upload-alt"></i>
                        </div>
                        <h3 class="upload-title">Drop video file here or click to browse</h3>
                        <p class="upload-subtitle">Supports MP4, AVI, MOV, MKV files (max 500MB)</p>
                        <input type="file" id="fileInput" class="file-input" accept="video/*">
                        <button class="upload-browse-btn" id="browseBtn">
                            <i class="fas fa-folder-open"></i> Browse Files
                        </button>
                        <div class="upload-info">
                            <p>Advanced video processing with AI-powered detection</p>
                        </div>
                    </div>
                    
                    <div id="fileInfo" class="file-info" style="display: none;">
                        <h6><i class="fas fa-file-video"></i> Selected File:</h6>
                        <p id="fileName" style="margin-bottom: 4px; color: #ffffff;"></p>
                        <p id="fileSize" style="margin: 0; color: #aaaaaa;"></p>
                    </div>
                    
                    <div style="text-align: center; margin-top: 24px;">
                        <button id="processBtn" class="upload-browse-btn" disabled style="background: #666666;">
                            <i class="fas fa-play"></i> Process Video
                        </button>
                    </div>
                </div>
            </section>
            
            <!-- Progress Section -->
            <section id="progressSection" class="upload-section" style="display: none;">
                <div class="upload-card">
                    <div class="progress-section">
                        <h4 class="progress-title" id="progressTitle">
                            <i class="fas fa-cogs"></i> Processing Status
                        </h4>
                        <div class="progress-bar">
                            <div class="progress-fill progress-animated" id="progressBar"></div>
                        </div>
                        <p class="progress-text" id="statusMessage">Waiting...</p>
                    </div>
                </div>
            </section>
            
            <!-- Results Section -->
            <section id="resultsSection" class="upload-section" style="display: none;">
                <div class="upload-card">
                    <div class="results-section">
                        <div class="results-header">
                            <h4 class="results-title">
                                <i class="fas fa-check-circle"></i> Processing Complete!
                            </h4>
                        </div>
                        <div class="results-actions" id="downloadButtons">
                            <!-- Download buttons will be added here -->
                        </div>
                    </div>
                </div>
            </section>
        </main>
    </div>

    <script>
        let currentFile = null;
        let currentSessionId = null;
        let statusCheckInterval = null;

        // DOM elements
        const dragDropArea = document.getElementById('dragDropArea');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const processBtn = document.getElementById('processBtn');
        const browseBtn = document.getElementById('browseBtn');

        // Event listeners
        dragDropArea.addEventListener('click', () => fileInput.click());
        browseBtn.addEventListener('click', () => fileInput.click());
        dragDropArea.addEventListener('dragover', handleDragOver);
        dragDropArea.addEventListener('dragleave', handleDragLeave);
        dragDropArea.addEventListener('drop', handleDrop);
        fileInput.addEventListener('change', handleFileSelect);
        processBtn.addEventListener('click', processVideo);

        function handleDragOver(e) {
            e.preventDefault();
            dragDropArea.classList.add('drag-over');
        }

        function handleDragLeave(e) {
            e.preventDefault();
            dragDropArea.classList.remove('drag-over');
        }

        function handleDrop(e) {
            e.preventDefault();
            dragDropArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        }

        function handleFileSelect(e) {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        }

        function handleFile(file) {
            if (file && file.type.startsWith('video/')) {
                currentFile = file;
                document.getElementById('fileName').textContent = file.name;
                document.getElementById('fileSize').textContent = formatFileSize(file.size);
                fileInfo.style.display = 'block';
                processBtn.disabled = false;
                processBtn.style.background = '#ff0000';
                processBtn.style.cursor = 'pointer';
            } else {
                alert('Please select a valid video file.');
            }
        }

        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        async function processVideo() {
            if (!currentFile) return;

            try {
                processBtn.disabled = true;
                processBtn.style.background = '#666666';
                
                // Upload file
                const formData = new FormData();
                formData.append('video', currentFile);
                
                const uploadResponse = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (!uploadResponse.ok) {
                    throw new Error('Upload failed');
                }
                
                const uploadResult = await uploadResponse.json();
                
                // Start processing
                const processResponse = await fetch('/api/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        file_path: uploadResult.file_path,
                        session_id: uploadResult.session_id
                    })
                });
                
                if (!processResponse.ok) {
                    throw new Error('Processing failed to start');
                }
                
                const processResult = await processResponse.json();
                currentSessionId = processResult.session_id;
                
                // Show progress section
                document.getElementById('progressSection').style.display = 'block';
                
                // Start status checking
                startStatusCheck(currentSessionId);
                
            } catch (error) {
                alert('Error: ' + error.message);
                processBtn.disabled = false;
                processBtn.style.background = '#ff0000';
            }
        }

        function startStatusCheck(sessionId) {
            statusCheckInterval = setInterval(async () => {
                try {
                    const response = await fetch(`/api/status/${sessionId}`);
                    if (!response.ok) return;
                    
                    const status = await response.json();
                    updateProgress(status);
                    
                    if (status.status === 'completed') {
                        clearInterval(statusCheckInterval);
                        showResults(status);
                        processBtn.disabled = false;
                        processBtn.style.background = '#ff0000';
                    } else if (status.status === 'error') {
                        clearInterval(statusCheckInterval);
                        alert('Processing failed: ' + status.error);
                        processBtn.disabled = false;
                        processBtn.style.background = '#ff0000';
                    }
                } catch (error) {
                    console.error('Status check failed:', error);
                }
            }, 2000);
        }

        function updateProgress(status) {
            const progressBar = document.getElementById('progressBar');
            const statusMessage = document.getElementById('statusMessage');
            
            progressBar.style.width = status.progress + '%';
            statusMessage.textContent = status.message;
        }

        function showResults(status) {
            const resultsSection = document.getElementById('resultsSection');
            const downloadButtons = document.getElementById('downloadButtons');
            
            // Clear existing buttons
            downloadButtons.innerHTML = '';
            
            // Add download buttons
            const buttons = [
                { label: 'Processed Video', endpoint: 'video', icon: 'fas fa-video', class: 'primary' },
                { label: 'Detection Report', endpoint: 'report', icon: 'fas fa-file-alt', class: 'secondary' },
                { label: 'Visualizations', endpoint: 'visualization', icon: 'fas fa-chart-bar', class: 'tertiary' }
            ];
            
            buttons.forEach(btn => {
                const button = document.createElement('a');
                button.href = `/api/download/${btn.endpoint}/${currentSessionId}`;
                button.className = `result-btn ${btn.class}`;
                button.innerHTML = `<i class="${btn.icon}"></i> ${btn.label}`;
                button.download = true;
                downloadButtons.appendChild(button);
            });
            
            resultsSection.style.display = 'block';
        }
    </script>
</body>
</html>
    ''')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video file"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate session ID
        session_id = f"single_{int(time.time())}"
        
        # Upload file
        result = upload_handler.upload_single_video(file, session_id)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'session_id': session_id,
            'file_path': result['file_path'],
            'message': 'Upload successful'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def start_processing():
    """Start video processing"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        session_id = data.get('session_id')
        
        if not file_path or not session_id:
            return jsonify({'error': 'Missing file_path or session_id'}), 400
        
        # Create processing task
        task = ProcessingTask(session_id, file_path)
        processing_status[session_id] = task
        
        # Start processing in background
        thread = threading.Thread(target=process_video_async, args=(task,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'session_id': session_id,
            'status': 'started',
            'message': 'Processing started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<session_id>', methods=['GET'])
def get_status(session_id):
    """Get processing status"""
    try:
        if session_id not in processing_status:
            return jsonify({'error': 'Session not found'}), 404
        
        task = processing_status[session_id]
        
        return jsonify({
            'session_id': session_id,
            'status': task.status,
            'progress': task.progress,
            'message': task.message,
            'error': task.error,
            'processing_time': time.time() - task.start_time
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<file_type>/<session_id>', methods=['GET'])
def download_file(file_type, session_id):
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
                import zipfile
                import io
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    viz_paths = result['output_paths']['visualizations']
                    for viz_name, viz_path in viz_paths.items():
                        if os.path.exists(viz_path):
                            zip_file.write(viz_path, f"{viz_name}.png")
                
                zip_buffer.seek(0)
                
                return send_file(
                    io.BytesIO(zip_buffer.read()),
                    as_attachment=True,
                    download_name=f"visualizations_{session_id}.zip",
                    mimetype='application/zip'
                )
        
        return jsonify({'error': f'{file_type} not available'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting CheatGPT3 Video Processing Web Application...")
    print("Access the app at: http://127.0.0.1:5001")
    print("Network access at: http://0.0.0.0:5001")
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )
