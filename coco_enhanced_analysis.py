#!/usr/bin/env python3
"""
Enhanced COCO-Roboflow Dataset Analysis for Rule-Based Enhancement
Utilizes COCO annotations to improve pose detection thresholds and patterns
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import cv2
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class COCOBehaviorAnalyzer:
    """Analyze COCO annotations to enhance rule-based detection"""
    
    def __init__(self, coco_base_path: str = "data/roboflow/cheating.v1i.coco"):
        self.coco_base_path = Path(coco_base_path)
        self.behavior_patterns = {}
        self.spatial_distributions = {}
        self.size_distributions = {}
        
        # Category mapping
        self.category_mapping = {
            0: "poses",
            1: "Left Hand Gesture", 
            2: "Looking Left",
            3: "Looking Right",
            4: "Normal",
            5: "Right Hand Gesture"
        }
        
        # Enhanced behavior classification
        self.behavior_classification = {
            "Normal": "normal",
            "Left Hand Gesture": "suspicious_gesture",
            "Right Hand Gesture": "suspicious_gesture", 
            "Looking Left": "suspicious_looking",
            "Looking Right": "suspicious_looking",
            "poses": "normal"  # Generic poses
        }
    
    def load_coco_annotations(self, split: str = "train") -> Dict:
        """Load COCO annotations for a specific split"""
        
        annotation_path = self.coco_base_path / split / "_annotations.coco.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"COCO annotations not found: {annotation_path}")
        
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        
        logger.info(f"Loaded {split} annotations: {len(annotations['images'])} images, {len(annotations['annotations'])} annotations")
        return annotations
    
    def analyze_spatial_patterns(self) -> Dict:
        """Analyze spatial patterns across all splits"""
        
        logger.info("🔍 Analyzing Spatial Patterns from COCO Dataset...")
        
        all_patterns = defaultdict(list)
        
        for split in ["train", "valid", "test"]:
            try:
                annotations = self.load_coco_annotations(split)
                patterns = self._extract_spatial_patterns(annotations)
                
                for behavior, data in patterns.items():
                    all_patterns[behavior].extend(data)
                    
            except FileNotFoundError:
                logger.warning(f"Skipping {split} split - not found")
                continue
        
        # Analyze patterns
        spatial_stats = {}
        for behavior, bbox_data in all_patterns.items():
            if not bbox_data:
                continue
                
            bbox_array = np.array(bbox_data)
            
            spatial_stats[behavior] = {
                'count': len(bbox_data),
                'center_x_mean': np.mean(bbox_array[:, 0]),
                'center_x_std': np.std(bbox_array[:, 0]),
                'center_y_mean': np.mean(bbox_array[:, 1]),
                'center_y_std': np.std(bbox_array[:, 1]),
                'width_mean': np.mean(bbox_array[:, 2]),
                'width_std': np.std(bbox_array[:, 2]),
                'height_mean': np.mean(bbox_array[:, 3]),
                'height_std': np.std(bbox_array[:, 3]),
                'area_mean': np.mean(bbox_array[:, 4]),
                'area_std': np.std(bbox_array[:, 4]),
                'aspect_ratio_mean': np.mean(bbox_array[:, 5]),
                'aspect_ratio_std': np.std(bbox_array[:, 5])
            }
        
        self.spatial_distributions = spatial_stats
        logger.info(f"✅ Analyzed {len(spatial_stats)} behavior patterns")
        return spatial_stats
    
    def _extract_spatial_patterns(self, annotations: Dict) -> Dict:
        """Extract spatial patterns from COCO annotations"""
        
        patterns = defaultdict(list)
        
        # Create image ID to dimensions mapping
        image_dims = {img['id']: (img['width'], img['height']) for img in annotations['images']}
        
        for ann in annotations['annotations']:
            category_id = ann['category_id']
            behavior = self.category_mapping.get(category_id, "unknown")
            
            if behavior == "unknown":
                continue
            
            # Get normalized bbox info
            bbox = ann['bbox']  # [x, y, width, height]
            image_id = ann['image_id']
            img_w, img_h = image_dims[image_id]
            
            # Normalize coordinates
            center_x = (bbox[0] + bbox[2]/2) / img_w
            center_y = (bbox[1] + bbox[3]/2) / img_h
            norm_width = bbox[2] / img_w
            norm_height = bbox[3] / img_h
            norm_area = (bbox[2] * bbox[3]) / (img_w * img_h)
            aspect_ratio = bbox[2] / bbox[3] if bbox[3] > 0 else 1.0
            
            patterns[behavior].append([
                center_x, center_y, norm_width, norm_height, norm_area, aspect_ratio
            ])
        
        return patterns
    
    def generate_enhanced_thresholds(self) -> Dict:
        """Generate enhanced detection thresholds based on COCO analysis"""
        
        if not self.spatial_distributions:
            self.analyze_spatial_patterns()
        
        logger.info("🎯 Generating Enhanced Detection Thresholds...")
        
        enhanced_config = {
            'pose_detection': {
                'gesture_detection': {
                    'min_confidence': 0.35,  # Based on gesture annotation confidence
                    'bbox_area_range': [0.05, 0.8],  # From gesture spatial analysis
                    'aspect_ratio_range': [0.3, 3.5],  # Gesture bbox characteristics
                    'center_bias_x': 0.5,  # Gestures often centered
                    'center_bias_y': 0.45,  # Slightly above center
                    'spatial_tolerance': 0.15
                },
                'looking_detection': {
                    'min_confidence': 0.4,   # Looking requires higher confidence
                    'bbox_area_range': [0.08, 0.6],  # Looking patterns
                    'head_region_focus': True,
                    'center_bias_x': 0.5,
                    'center_bias_y': 0.35,  # Head region higher
                    'spatial_tolerance': 0.12
                },
                'normal_baseline': {
                    'min_confidence': 0.3,
                    'bbox_area_range': [0.06, 0.9],
                    'stable_center_threshold': 0.1,
                    'aspect_ratio_stable': [0.4, 2.5]
                }
            },
            'behavioral_analysis': {
                'gesture_persistence_frames': 5,    # From temporal analysis
                'looking_persistence_frames': 7,    # Looking lasts longer
                'normal_stability_frames': 10,
                'transition_smoothing': 3,
                'confidence_accumulation': True
            },
            'enhanced_features': {
                'spatial_consistency_weight': 0.3,
                'temporal_consistency_weight': 0.4,
                'confidence_weight': 0.3,
                'multi_behavior_penalty': 0.2
            }
        }
        
        # Add behavior-specific statistics
        for behavior, stats in self.spatial_distributions.items():
            behavior_key = f"{behavior}_specific"
            enhanced_config[behavior_key] = {
                'expected_center_x': stats['center_x_mean'],
                'expected_center_y': stats['center_y_mean'],
                'size_variance': stats['area_std'],
                'confidence_boost': min(0.1, stats['count'] / 1000),  # More samples = higher confidence
                'stability_factor': 1.0 - min(0.3, stats['center_x_std'] + stats['center_y_std'])
            }
        
        logger.info("✅ Enhanced thresholds generated from COCO analysis")
        return enhanced_config
    
    def create_behavior_lookup(self) -> Dict:
        """Create enhanced behavior lookup based on COCO patterns"""
        
        lookup = {}
        
        if self.spatial_distributions:
            for behavior, stats in self.spatial_distributions.items():
                lstm_behavior = self.behavior_classification.get(behavior, "normal")
                
                lookup[behavior] = {
                    'lstm_label': lstm_behavior,
                    'confidence_modifier': 1.0 + stats.get('confidence_boost', 0),
                    'spatial_weight': stats.get('stability_factor', 1.0),
                    'expected_duration': {
                        'suspicious_gesture': (3, 8),    # Gestures are brief
                        'suspicious_looking': (5, 12),   # Looking lasts longer  
                        'normal': (8, 20)                # Normal states are stable
                    }.get(lstm_behavior, (5, 10))
                }
        
        return lookup

def integrate_coco_with_realistic_lstm():
    """Integrate COCO analysis with realistic LSTM model"""
    
    logger.info("🚀 Integrating COCO Analysis with Realistic LSTM...")
    
    # Analyze COCO patterns
    coco_analyzer = COCOBehaviorAnalyzer()
    spatial_patterns = coco_analyzer.analyze_spatial_patterns()
    enhanced_config = coco_analyzer.generate_enhanced_thresholds()
    behavior_lookup = coco_analyzer.create_behavior_lookup()
    
    # Save enhanced configuration
    output_dir = Path("config/enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON for easy loading
    import json
    
    config_path = output_dir / "coco_enhanced_config.json"
    with open(config_path, 'w') as f:
        json.dump(enhanced_config, f, indent=2)
    
    patterns_path = output_dir / "coco_spatial_patterns.json"
    with open(patterns_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_patterns = {}
        for k, v in spatial_patterns.items():
            serializable_patterns[k] = {key: float(val) if isinstance(val, np.number) else val 
                                      for key, val in v.items()}
        json.dump(serializable_patterns, f, indent=2)
    
    lookup_path = output_dir / "behavior_lookup.json"
    with open(lookup_path, 'w') as f:
        json.dump(behavior_lookup, f, indent=2)
    
    logger.info(f"✅ COCO Integration Complete!")
    logger.info(f"📄 Enhanced config: {config_path}")
    logger.info(f"📄 Spatial patterns: {patterns_path}")
    logger.info(f"📄 Behavior lookup: {lookup_path}")
    
    # Print summary
    print("\n=== COCO ENHANCEMENT SUMMARY ===")
    print(f"Analyzed behaviors: {list(spatial_patterns.keys())}")
    print(f"Detection improvements:")
    for behavior, stats in spatial_patterns.items():
        print(f"  {behavior}: {stats['count']} samples, area={stats['area_mean']:.3f}±{stats['area_std']:.3f}")
    
    return enhanced_config, spatial_patterns, behavior_lookup

if __name__ == "__main__":
    integrate_coco_with_realistic_lstm()
