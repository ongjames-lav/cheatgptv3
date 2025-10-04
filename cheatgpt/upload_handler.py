"""
Upload Handler for CheatGPT3 Video Processing
Handles file uploads and validation
"""

import os
import time
import mimetypes
from pathlib import Path
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from typing import Dict, Any, List

class UploadHandler:
    """Handle video file uploads and validation"""
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        self.allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'm4v', 'flv', 'webm'}
        self.max_file_size = 500 * 1024 * 1024  # 500MB
        
        # Create upload directory
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Check for python-magic
        try:
            import magic
            self.magic = magic
            print("Using python-magic for file type detection")
        except ImportError:
            self.magic = None
            print("Warning: python-magic not available, using mimetypes for file detection")
    
    def upload_single_video(self, file: FileStorage, session_id: str) -> Dict[str, Any]:
        """
        Upload a single video file
        
        Args:
            file: Uploaded file object
            session_id: Session identifier
            
        Returns:
            Dict with upload result
        """
        try:
            # Validate file
            validation_result = self._validate_file(file)
            if validation_result['valid']:
                
                # Create session directory
                session_dir = os.path.join(self.upload_dir, session_id)
                os.makedirs(session_dir, exist_ok=True)
                
                # Generate secure filename
                filename = secure_filename(file.filename)
                if not filename:
                    filename = f"upload_{int(time.time())}.mp4"
                
                # Save file
                file_path = os.path.join(session_dir, filename)
                file.save(file_path)
                
                # Verify file was saved
                if not os.path.exists(file_path):
                    return {'error': 'Failed to save file'}
                
                file_size = os.path.getsize(file_path)
                
                return {
                    'session_id': session_id,
                    'file_path': file_path,
                    'filename': filename,
                    'file_size': file_size,
                    'upload_time': time.time(),
                    'status': 'uploaded'
                }
            else:
                return {'error': validation_result['error']}
                
        except Exception as e:
            return {'error': f'Upload failed: {str(e)}'}
    
    def _validate_file(self, file: FileStorage) -> Dict[str, Any]:
        """Validate uploaded file"""
        try:
            # Check if file exists
            if not file or not file.filename:
                return {'valid': False, 'error': 'No file provided'}
            
            # Check file extension
            file_ext = Path(file.filename).suffix.lower().lstrip('.')
            if file_ext not in self.allowed_extensions:
                return {
                    'valid': False, 
                    'error': f'File type .{file_ext} not allowed. Supported: {", ".join(self.allowed_extensions)}'
                }
            
            # Check file size (if available)
            if hasattr(file, 'content_length') and file.content_length:
                if file.content_length > self.max_file_size:
                    max_size_mb = self.max_file_size // (1024 * 1024)
                    return {
                        'valid': False,
                        'error': f'File too large. Maximum size: {max_size_mb}MB'
                    }
            
            # Check MIME type
            if not self._is_video_file(file):
                return {
                    'valid': False,
                    'error': 'File is not a valid video format'
                }
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation failed: {str(e)}'}
    
    def _is_video_file(self, file: FileStorage) -> bool:
        """Check if file is a video using MIME type detection"""
        try:
            if self.magic:
                # Use python-magic for more accurate detection
                file_data = file.stream.read(1024)  # Read first 1KB
                file.stream.seek(0)  # Reset stream position
                
                mime_type = self.magic.from_buffer(file_data, mime=True)
                return mime_type.startswith('video/')
            else:
                # Fallback to mimetypes module
                mime_type, _ = mimetypes.guess_type(file.filename)
                return mime_type and mime_type.startswith('video/')
                
        except Exception:
            # If detection fails, rely on file extension
            file_ext = Path(file.filename).suffix.lower().lstrip('.')
            return file_ext in self.allowed_extensions
    
    def cleanup_session(self, session_id: str) -> bool:
        """Clean up uploaded files for a session"""
        try:
            session_dir = os.path.join(self.upload_dir, session_id)
            if os.path.exists(session_dir):
                import shutil
                shutil.rmtree(session_dir)
                return True
            return False
        except Exception as e:
            print(f"Cleanup failed for session {session_id}: {e}")
            return False
    
    def get_upload_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about uploaded files for a session"""
        try:
            session_dir = os.path.join(self.upload_dir, session_id)
            if not os.path.exists(session_dir):
                return {'error': 'Session not found'}
            
            files = []
            for filename in os.listdir(session_dir):
                file_path = os.path.join(session_dir, filename)
                if os.path.isfile(file_path):
                    file_info = {
                        'filename': filename,
                        'file_path': file_path,
                        'file_size': os.path.getsize(file_path),
                        'upload_time': os.path.getctime(file_path)
                    }
                    files.append(file_info)
            
            return {
                'session_id': session_id,
                'files': files,
                'total_files': len(files)
            }
            
        except Exception as e:
            return {'error': f'Failed to get upload info: {str(e)}'}
