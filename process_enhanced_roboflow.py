#!/usr/bin/env python3
"""
Enhanced Roboflow Dataset Processor for CheatGPT
Processes the 6-category cheating behavior dataset for better LSTM and rule-based integration
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import cv2
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRoboflowProcessor:
    """Enhanced processor for the 6-category cheating behavior dataset"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.category_mapping = {
            'Normal': 'normal',
            'poses': 'normal',  # General poses are considered normal
            'Left Hand Gesture': 'suspicious_gesture',
            'Right Hand Gesture': 'suspicious_gesture', 
            'Looking Left': 'suspicious_looking',
            'Looking Right': 'suspicious_looking'
        }
        
        # Enhanced feature mapping for rule-based system
        self.rule_mapping = {
            'Normal': {'lean_flag': 0, 'look_flag': 0, 'phone_flag': 0, 'gesture_flag': 0},
            'poses': {'lean_flag': 0, 'look_flag': 0, 'phone_flag': 0, 'gesture_flag': 0},
            'Left Hand Gesture': {'lean_flag': 0, 'look_flag': 0, 'phone_flag': 1, 'gesture_flag': 1},
            'Right Hand Gesture': {'lean_flag': 0, 'look_flag': 0, 'phone_flag': 1, 'gesture_flag': 1},
            'Looking Left': {'lean_flag': 0, 'look_flag': 1, 'phone_flag': 0, 'gesture_flag': 0},
            'Looking Right': {'lean_flag': 0, 'look_flag': 1, 'phone_flag': 0, 'gesture_flag': 0}
        }
        
    def load_annotations(self, split: str = 'train') -> Dict:
        """Load COCO annotations for a specific split"""
        annotation_file = self.dataset_path / split / '_annotations.coco.json'
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
            
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            
        logger.info(f"Loaded {len(data['images'])} images with {len(data['annotations'])} annotations from {split}")
        return data
    
    def process_enhanced_features(self, annotation_data: Dict, split: str = 'train') -> pd.DataFrame:
        """Process annotations into enhanced features for both rule-based and LSTM"""
        
        # Create category lookup
        categories = {cat['id']: cat['name'] for cat in annotation_data['categories']}
        
        # Create image lookup
        images = {img['id']: img for img in annotation_data['images']}
        
        processed_data = []
        
        for ann in annotation_data['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id']
            
            if image_id not in images or category_id not in categories:
                continue
                
            image_info = images[image_id]
            category_name = categories[category_id]
            
            # Get bbox information for spatial features
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # Calculate position features (normalized)
            img_width = image_info['width']
            img_height = image_info['height']
            
            center_x = (x + w/2) / img_width
            center_y = (y + h/2) / img_height
            bbox_area = (w * h) / (img_width * img_height)
            
            # Enhanced rule-based features
            rules = self.rule_mapping.get(category_name, {'lean_flag': 0, 'look_flag': 0, 'phone_flag': 0, 'gesture_flag': 0})
            
            # Enhanced angle estimation based on category
            lean_angle = self._estimate_lean_angle(category_name, center_x, center_y)
            head_turn_angle = self._estimate_head_angle(category_name, center_x)
            
            # Enhanced confidence based on bbox size and position
            confidence = self._calculate_confidence(category_name, bbox_area, center_x, center_y)
            
            # Map to LSTM categories
            lstm_label = self.category_mapping.get(category_name, 'normal')
            
            # Create comprehensive feature vector
            feature_row = {
                'image_id': image_id,
                'frame_id': image_id,  # Use image_id as frame_id
                'person_id': f"person_{image_id:04d}",
                'split': split,
                
                # Original category information
                'original_category': category_name,
                'lstm_label': lstm_label,
                'label': lstm_label,  # For training compatibility
                
                # Rule-based features
                'lean_flag': rules['lean_flag'],
                'look_flag': rules['look_flag'], 
                'phone_flag': rules['phone_flag'],
                'gesture_flag': rules['gesture_flag'],
                
                # Enhanced continuous features
                'lean_angle': lean_angle,
                'head_turn_angle': head_turn_angle,
                'confidence': confidence,
                
                # Spatial features
                'center_x': center_x,
                'center_y': center_y,
                'bbox_area': bbox_area,
                'bbox_width': w / img_width,
                'bbox_height': h / img_height,
                
                # Derived features for LSTM
                'combined_suspicious': int(rules['look_flag'] or rules['gesture_flag']),
                'lean_magnitude': abs(lean_angle) / 45.0,
                'head_magnitude': abs(head_turn_angle) / 90.0,
                'spatial_center_offset': abs(center_x - 0.5) + abs(center_y - 0.5),
                
                # Image metadata
                'image_width': img_width,
                'image_height': img_height,
                'image_file': image_info['file_name']
            }
            
            processed_data.append(feature_row)
            
        df = pd.DataFrame(processed_data)
        logger.info(f"Processed {len(df)} feature vectors from {split} split")
        
        # Log category distribution
        logger.info(f"Category distribution in {split}:")
        for cat, count in df['original_category'].value_counts().items():
            lstm_cat = df[df['original_category'] == cat]['lstm_label'].iloc[0]
            logger.info(f"  {cat} -> {lstm_cat}: {count} samples")
            
        return df
    
    def _estimate_lean_angle(self, category: str, center_x: float, center_y: float) -> float:
        """Estimate lean angle based on category and position"""
        if 'Hand Gesture' in category:
            # Hand gestures might indicate leaning
            return np.random.normal(8, 3) if np.random.random() > 0.5 else np.random.normal(0, 2)
        elif 'Looking' in category:
            # Looking behaviors might involve slight leaning
            return np.random.normal(5, 2) if np.random.random() > 0.6 else np.random.normal(0, 1)
        else:
            # Normal poses
            return np.random.normal(0, 2)
    
    def _estimate_head_angle(self, category: str, center_x: float) -> float:
        """Estimate head turn angle based on category and position"""
        if category == 'Looking Left':
            return np.random.normal(-20, 5)  # Negative for left turn
        elif category == 'Looking Right':
            return np.random.normal(20, 5)   # Positive for right turn
        elif 'Hand Gesture' in category:
            # Gestures might involve head movement
            return np.random.normal(0, 8)
        else:
            # Normal poses
            return np.random.normal(0, 3)
    
    def _calculate_confidence(self, category: str, bbox_area: float, center_x: float, center_y: float) -> float:
        """Calculate detection confidence based on various factors"""
        base_confidence = 0.7
        
        # Higher confidence for clear suspicious behaviors
        if category in ['Looking Left', 'Looking Right', 'Left Hand Gesture', 'Right Hand Gesture']:
            base_confidence = 0.85
        elif category == 'Normal':
            base_confidence = 0.9
        
        # Adjust for bbox size (larger = more confident)
        size_factor = min(bbox_area * 10, 0.15)  # Cap the bonus
        
        # Adjust for position (center = more confident)
        center_factor = 1 - (abs(center_x - 0.5) + abs(center_y - 0.5)) * 0.1
        
        final_confidence = base_confidence + size_factor + center_factor * 0.05
        return np.clip(final_confidence, 0.1, 0.99)
    
    def create_enhanced_sequences(self, df: pd.DataFrame, sequence_length: int = 10) -> pd.DataFrame:
        """Create enhanced temporal sequences from the processed data"""
        
        # Group by similar behaviors to create realistic sequences
        sequences = []
        sequence_id = 0
        
        # Group images by behavior type for sequence creation
        behavior_groups = df.groupby('lstm_label')
        
        for behavior, group in behavior_groups:
            # Sort by image_id to maintain some temporal order
            group = group.sort_values('image_id')
            
            # Create overlapping sequences
            for i in range(0, len(group), sequence_length // 2):  # 50% overlap
                sequence_data = group.iloc[i:i+sequence_length].copy()
                
                if len(sequence_data) < sequence_length:
                    # Pad with the last frame if needed
                    last_frame = sequence_data.iloc[-1:].copy()
                    while len(sequence_data) < sequence_length:
                        sequence_data = pd.concat([sequence_data, last_frame], ignore_index=True)
                
                # Update sequence metadata
                sequence_data['sequence_id'] = sequence_id
                sequence_data['frame_in_sequence'] = range(len(sequence_data))
                sequence_data['sequence_behavior'] = behavior
                
                sequences.append(sequence_data)
                sequence_id += 1
        
        # Combine all sequences
        result_df = pd.concat(sequences, ignore_index=True)
        
        logger.info(f"Created {sequence_id} sequences of length {sequence_length}")
        logger.info(f"Sequence behavior distribution:")
        for behavior, count in result_df.groupby('sequence_id')['sequence_behavior'].first().value_counts().items():
            logger.info(f"  {behavior}: {count} sequences")
            
        return result_df
    
    def process_all_splits(self, output_dir: str = "data/roboflow/enhanced_processed"):
        """Process all dataset splits and save enhanced features"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_data = []
        
        for split in ['train', 'valid', 'test']:
            try:
                logger.info(f"Processing {split} split...")
                annotations = self.load_annotations(split)
                df = self.process_enhanced_features(annotations, split)
                
                # Save individual split
                split_file = output_path / f"{split}_features.csv"
                df.to_csv(split_file, index=False)
                logger.info(f"Saved {split} features to {split_file}")
                
                all_data.append(df)
                
            except FileNotFoundError:
                logger.warning(f"Split {split} not found, skipping...")
                continue
        
        if all_data:
            # Combine all splits for sequence creation
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Create enhanced sequences
            sequence_df = self.create_enhanced_sequences(combined_df)
            
            # Save combined features and sequences
            combined_file = output_path / "combined_features.csv"
            sequence_file = output_path / "enhanced_sequences.csv"
            
            combined_df.to_csv(combined_file, index=False)
            sequence_df.to_csv(sequence_file, index=False)
            
            logger.info(f"Saved combined features to {combined_file}")
            logger.info(f"Saved enhanced sequences to {sequence_file}")
            
            # Print final statistics
            logger.info("\n=== ENHANCED DATASET STATISTICS ===")
            logger.info(f"Total samples: {len(combined_df)}")
            logger.info(f"Total sequences: {len(sequence_df) // 10}")
            logger.info("\nFeature distribution:")
            feature_stats = combined_df[['lean_flag', 'look_flag', 'phone_flag', 'gesture_flag']].sum()
            for feature, count in feature_stats.items():
                logger.info(f"  {feature}: {count} ({count/len(combined_df)*100:.1f}%)")
            
            logger.info("\nLSTM label distribution:")
            for label, count in combined_df['lstm_label'].value_counts().items():
                logger.info(f"  {label}: {count} ({count/len(combined_df)*100:.1f}%)")
                
            return combined_df, sequence_df
        
        return None, None

def main():
    """Main processing function"""
    dataset_path = "data/roboflow/cheating.v1i.coco"
    
    if not Path(dataset_path).exists():
        logger.error(f"Dataset path not found: {dataset_path}")
        return
    
    processor = EnhancedRoboflowProcessor(dataset_path)
    combined_df, sequence_df = processor.process_all_splits()
    
    if combined_df is not None:
        logger.info("✅ Enhanced Roboflow dataset processing completed!")
        logger.info("📊 Ready for enhanced LSTM training and rule-based integration")
    else:
        logger.error("❌ Processing failed")

if __name__ == "__main__":
    main()
