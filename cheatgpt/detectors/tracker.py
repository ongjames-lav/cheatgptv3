"""Simple but effective object tracker for person tracking in exam scenarios."""
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Track:
    """Represents a single tracked object (person)."""
    
    def __init__(self, track_id: str, bbox: List[float], confidence: float, timestamp: float):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.last_seen = timestamp
        self.age = 0  # Number of frames since creation
        self.time_since_update = 0  # Frames since last detection
        self.hit_streak = 1  # Consecutive successful detections
        self.history = [bbox]  # History of bounding boxes
        self.created_at = timestamp
        
    def update(self, bbox: List[float], confidence: float, timestamp: float):
        """Update track with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = timestamp
        self.time_since_update = 0
        self.hit_streak += 1
        self.history.append(bbox)
        
        # Keep only recent history (last 10 positions)
        if len(self.history) > 10:
            self.history.pop(0)
    
    def predict(self):
        """Predict next position (simple linear prediction)."""
        if len(self.history) < 2:
            return self.bbox
        
        # Simple linear extrapolation based on last two positions
        prev_bbox = self.history[-2]
        curr_bbox = self.history[-1]
        
        dx = curr_bbox[0] - prev_bbox[0]
        dy = curr_bbox[1] - prev_bbox[1]
        dw = curr_bbox[2] - prev_bbox[2]
        dh = curr_bbox[3] - prev_bbox[3]
        
        predicted_bbox = [
            curr_bbox[0] + dx,
            curr_bbox[1] + dy,
            curr_bbox[2] + dw,
            curr_bbox[3] + dh
        ]
        
        return predicted_bbox
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_area(self) -> float:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

class Tracker:
    """Simple multi-object tracker for person tracking."""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100.0):
        """
        Initialize tracker.
        
        Args:
            max_disappeared: Maximum frames a track can be missing before deletion
            max_distance: Maximum distance for track association
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.tracks: Dict[str, Track] = {}
        self.next_track_id = 0
        self.frame_count = 0
        
        logger.info(f"Tracker initialized with max_disappeared={max_disappeared}, max_distance={max_distance}")
    
    def update(self, detections: List[Dict[str, Any]], timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with 'bbox', 'conf', 'cls_name'
            timestamp: Current timestamp
            
        Returns:
            List of tracked detections with track_id added
        """
        if timestamp is None:
            timestamp = self.frame_count
        
        self.frame_count += 1
        
        # Filter for person detections only
        person_detections = [det for det in detections if det.get('cls_name') == 'person']
        
        if not person_detections:
            # No detections, just age existing tracks
            self._age_tracks()
            return []
        
        # Extract detection bboxes and confidences
        detection_bboxes = [det['bbox'] for det in person_detections]
        detection_confs = [det['conf'] for det in person_detections]
        
        # Predict track positions
        predicted_positions = {}
        for track_id, track in self.tracks.items():
            predicted_positions[track_id] = track.predict()
        
        # Associate detections with existing tracks
        track_assignments, unmatched_detections = self._associate_detections(
            detection_bboxes, predicted_positions, timestamp
        )
        
        # Update matched tracks
        tracked_detections = []
        for track_id, det_idx in track_assignments.items():
            track = self.tracks[track_id]
            detection = person_detections[det_idx]
            
            track.update(detection['bbox'], detection['conf'], timestamp)
            
            # Create tracked detection
            tracked_detection = detection.copy()
            tracked_detection['track_id'] = track_id
            tracked_detections.append(tracked_detection)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = person_detections[det_idx]
            track_id = self._create_new_track(detection['bbox'], detection['conf'], timestamp)
            
            # Create tracked detection
            tracked_detection = detection.copy()
            tracked_detection['track_id'] = track_id
            tracked_detections.append(tracked_detection)
        
        # Age and clean up tracks
        self._age_tracks()
        
        logger.debug(f"Frame {self.frame_count}: {len(tracked_detections)} tracked, {len(self.tracks)} active tracks")
        
        return tracked_detections
    
    def _associate_detections(self, detection_bboxes: List[List[float]], 
                            predicted_positions: Dict[str, List[float]], 
                            timestamp: float) -> Tuple[Dict[str, int], List[int]]:
        """Associate detections with existing tracks using improved distance-based matching."""
        
        if not self.tracks or not detection_bboxes:
            return {}, list(range(len(detection_bboxes)))
        
        # Calculate distance matrix
        track_ids = list(self.tracks.keys())
        distance_matrix = np.full((len(track_ids), len(detection_bboxes)), float('inf'))
        
        for i, track_id in enumerate(track_ids):
            predicted_bbox = predicted_positions[track_id]
            
            for j, detection_bbox in enumerate(detection_bboxes):
                # Use improved distance calculation
                combined_distance = self._calculate_bbox_distance(predicted_bbox, detection_bbox)
                
                # Apply distance threshold (adjusted for new distance metric)
                if combined_distance < self.max_distance * 1.5:  # More lenient threshold
                    distance_matrix[i, j] = combined_distance
        
        # Simple greedy assignment (can be improved with Hungarian algorithm)
        assignments = {}
        used_detections = set()
        
        # Sort by distance and assign greedily
        track_det_pairs = []
        for i in range(len(track_ids)):
            for j in range(len(detection_bboxes)):
                if distance_matrix[i, j] != float('inf'):
                    track_det_pairs.append((distance_matrix[i, j], track_ids[i], j))
        
        track_det_pairs.sort(key=lambda x: x[0])  # Sort by distance
        
        for distance, track_id, det_idx in track_det_pairs:
            if track_id not in assignments and det_idx not in used_detections:
                assignments[track_id] = det_idx
                used_detections.add(det_idx)
        
        # Find unmatched detections
        unmatched_detections = [i for i in range(len(detection_bboxes)) if i not in used_detections]
        
        return assignments, unmatched_detections
    
    def _create_new_track(self, bbox: List[float], confidence: float, timestamp: float) -> str:
        """Create a new track."""
        track_id = f"person_{self.next_track_id:03d}"
        self.next_track_id += 1
        
        track = Track(track_id, bbox, confidence, timestamp)
        self.tracks[track_id] = track
        
        logger.info(f"Created new track: {track_id}")
        return track_id
    
    def _age_tracks(self):
        """Age tracks and remove old ones."""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            track.age += 1
            track.time_since_update += 1
            
            # Remove tracks that haven't been seen for too long
            if track.time_since_update > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            logger.info(f"Removing lost track: {track_id}")
            del self.tracks[track_id]
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_bbox_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate distance between two bounding boxes using both center distance and IoU."""
        # Center distance
        center1 = self._get_bbox_center(bbox1)
        center2 = self._get_bbox_center(bbox2)
        center_dist = self._calculate_distance(center1, center2)
        
        # IoU (higher IoU = lower distance)
        iou = self._calculate_iou(bbox1, bbox2)
        iou_factor = 1.0 - iou  # Convert to distance metric
        
        # Size similarity factor
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        size_factor = 1.0 - size_ratio
        
        # Combined distance (weighted)
        combined_distance = (0.5 * center_dist) + (0.3 * iou_factor * 100) + (0.2 * size_factor * 100)
        
        return combined_distance
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # Calculate union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            if union <= 0:
                return 0.0
            
            return intersection / union
        
        except Exception:
            return 0.0
    
    def get_active_tracks(self) -> Dict[str, Track]:
        """Get all currently active tracks."""
        return self.tracks.copy()
    
    def get_track_info(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific track."""
        if track_id not in self.tracks:
            return None
        
        track = self.tracks[track_id]
        return {
            'track_id': track_id,
            'bbox': track.bbox,
            'confidence': track.confidence,
            'age': track.age,
            'hit_streak': track.hit_streak,
            'last_seen': track.last_seen,
            'created_at': track.created_at
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            'active_tracks': len(self.tracks),
            'total_tracks_created': self.next_track_id,
            'frame_count': self.frame_count,
            'max_disappeared': self.max_disappeared,
            'max_distance': self.max_distance
        }
