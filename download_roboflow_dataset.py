"""
Download and Process Roboflow Cheating Dataset for LSTM Training
================================================================

This script downloads the Roboflow cheating dataset and processes it into
pose sequences for training the LSTM body movement classifier.

Dataset: https://universe.roboflow.com/rp-fcesi/cheating-n3uwx
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from typing import List, Dict, Any, Tuple
import requests
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, using system environment variables only

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RoboflowDatasetDownloader:
    """Download and process Roboflow cheating dataset."""
    
    def __init__(self, api_key: str = None):
        """Initialize the downloader with API key."""
        self.api_key = api_key or os.getenv('ROBOFLOW_API_KEY')
        self.dataset_url = "https://universe.roboflow.com/rp-fcesi/cheating-n3uwx"
        self.workspace = "rp-fcesi"
        self.project = "cheating-n3uwx"
        self.version = 1
        
        # Local paths
        self.data_dir = Path("data/roboflow")
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        for dir_path in [self.data_dir, self.images_dir, self.annotations_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_with_roboflow_pip(self):
        """Download dataset using roboflow pip package."""
        try:
            # Try to import roboflow
            try:
                from roboflow import Roboflow
                logger.info("✓ Roboflow package found")
            except ImportError:
                logger.error("❌ Roboflow package not installed. Installing...")
                os.system("pip install roboflow")
                from roboflow import Roboflow
                logger.info("✓ Roboflow package installed")
            
            if not self.api_key:
                logger.error("❌ ROBOFLOW_API_KEY not found. Please set it in .env file or environment.")
                logger.info("Get your API key from: https://app.roboflow.com/")
                return False
            
            # Initialize Roboflow
            rf = Roboflow(api_key=self.api_key)
            project = rf.workspace(self.workspace).project(self.project)
            
            # Download dataset
            logger.info(f"📥 Downloading dataset from {self.dataset_url}")
            dataset = project.version(self.version).download("coco", location=str(self.data_dir))
            
            logger.info(f"✓ Dataset downloaded to {self.data_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading with roboflow package: {e}")
            return False
    
    def download_sample_dataset(self):
        """Download a sample subset if full dataset is not available."""
        logger.info("📥 Creating sample cheating dataset for demonstration...")
        
        # Create sample images and annotations
        sample_data = self._create_sample_cheating_data()
        
        # Save sample dataset
        sample_file = self.processed_dir / "sample_cheating_sequences.csv"
        sample_data.to_csv(sample_file, index=False)
        
        logger.info(f"✓ Sample dataset created: {sample_file}")
        return True
    
    def _create_sample_cheating_data(self) -> pd.DataFrame:
        """Create realistic sample cheating behavior sequences."""
        np.random.seed(42)
        
        behaviors = ['normal', 'leaning', 'looking_around', 'phone_use', 'cheating']
        behavior_weights = [0.5, 0.15, 0.15, 0.1, 0.1]
        
        data = []
        num_sequences = 50  # Number of behavior sequences
        frames_per_sequence = 30  # Frames per sequence (1 second at 30fps)
        
        for seq_id in range(num_sequences):
            # Choose sequence behavior
            sequence_behavior = np.random.choice(behaviors, p=behavior_weights)
            
            for frame_id in range(frames_per_sequence):
                # Create temporal progression within sequence
                progression = frame_id / frames_per_sequence
                
                # Generate realistic pose features based on behavior
                if sequence_behavior == 'normal':
                    lean_flag = int(np.random.random() < 0.05)  # 5% chance
                    look_flag = int(np.random.random() < 0.1)   # 10% chance
                    phone_flag = 0
                    lean_angle = np.random.normal(0, 2)  # Small natural variation
                    head_angle = np.random.normal(0, 5)  # Small head movements
                    
                elif sequence_behavior == 'leaning':
                    lean_flag = int(np.random.random() < 0.8)  # 80% leaning
                    look_flag = int(np.random.random() < 0.3)  # 30% looking
                    phone_flag = 0
                    lean_angle = np.random.normal(15, 5)  # Consistent lean
                    head_angle = np.random.normal(0, 8)
                    
                elif sequence_behavior == 'looking_around':
                    lean_flag = int(np.random.random() < 0.2)  # 20% leaning
                    look_flag = int(np.random.random() < 0.9)  # 90% looking
                    phone_flag = 0
                    lean_angle = np.random.normal(5, 3)
                    head_angle = np.random.normal(25, 10)  # Significant head turns
                    
                elif sequence_behavior == 'phone_use':
                    lean_flag = int(np.random.random() < 0.6)  # 60% leaning toward phone
                    look_flag = int(np.random.random() < 0.7)  # 70% looking at phone
                    phone_flag = int(np.random.random() < 0.9)  # 90% phone detected
                    lean_angle = np.random.normal(10, 4)
                    head_angle = np.random.normal(-15, 8)  # Looking down at phone
                    
                else:  # cheating
                    lean_flag = int(np.random.random() < 0.9)  # 90% leaning
                    look_flag = int(np.random.random() < 0.9)  # 90% looking
                    phone_flag = int(np.random.random() < 0.7)  # 70% phone
                    lean_angle = np.random.normal(20, 6)  # Strong lean
                    head_angle = np.random.normal(30, 12)  # Strong head turn
                
                # Add temporal smoothing
                if frame_id > 0:
                    # Smooth angles with previous frame
                    prev_lean = data[-1]['lean_angle'] if data else 0
                    prev_head = data[-1]['head_turn_angle'] if data else 0
                    
                    lean_angle = 0.7 * lean_angle + 0.3 * prev_lean
                    head_angle = 0.7 * head_angle + 0.3 * prev_head
                
                # Calculate confidence based on consistency
                confidence = 0.8 + 0.2 * (1 - abs(np.random.normal(0, 0.1)))
                confidence = np.clip(confidence, 0.3, 1.0)
                
                data.append({
                    'sequence_id': seq_id,
                    'frame_id': frame_id,
                    'person_id': f"person_{seq_id:03d}",
                    'lean_flag': lean_flag,
                    'look_flag': look_flag,
                    'phone_flag': phone_flag,
                    'lean_angle': round(lean_angle, 2),
                    'head_turn_angle': round(head_angle, 2),
                    'confidence': round(confidence, 3),
                    'behavior': sequence_behavior,
                    'label': sequence_behavior,
                    'timestamp': frame_id * 0.033  # 30fps timing
                })
        
        return pd.DataFrame(data)
    
    def process_annotations_to_sequences(self):
        """Process COCO annotations into behavior sequences."""
        logger.info("🔄 Processing annotations into behavior sequences...")
        
        # Look for annotation files
        annotation_files = list(self.data_dir.glob("**/*.json"))
        
        if not annotation_files:
            logger.warning("⚠️ No annotation files found. Using sample data.")
            return self.download_sample_dataset()
        
        all_sequences = []
        
        for ann_file in annotation_files:
            logger.info(f"Processing {ann_file.name}...")
            
            try:
                with open(ann_file, 'r') as f:
                    coco_data = json.load(f)
                
                # Extract sequences from COCO annotations
                sequences = self._extract_sequences_from_coco(coco_data)
                all_sequences.extend(sequences)
                
            except Exception as e:
                logger.error(f"Error processing {ann_file}: {e}")
        
        if all_sequences:
            # Save processed sequences
            df = pd.DataFrame(all_sequences)
            
            # Force better balance by creating additional normal sequences
            normal_count = len(df[df['label'] == 'normal'])
            suspicious_count = len(df[df['label'].isin(['suspicious', 'suspicious_multiple'])])
            
            logger.info(f"Original distribution - Normal: {normal_count}, Suspicious: {suspicious_count}")
            
            # If we have too few normal sequences, create more by duplicating and varying normal ones
            if normal_count < suspicious_count * 0.4:  # Less than 40% normal
                normal_sequences = df[df['label'] == 'normal'].copy()
                additional_normal = []
                
                # Create variations of normal sequences
                needed_normal = int(suspicious_count * 0.5) - normal_count
                for i in range(needed_normal):
                    if len(normal_sequences) > 0:
                        # Pick a random normal sequence and create a variation
                        base_seq = normal_sequences.iloc[i % len(normal_sequences)].copy()
                        base_seq['image_id'] = len(df) + i + 1000  # Unique ID
                        base_seq['person_id'] = f'person_{(len(df) + i) // 10 + 100:03d}'  # New person ID
                        base_seq['frame_id'] = (len(df) + i) % 10
                        # Add slight variations to make it realistic
                        base_seq['lean_angle'] += np.random.normal(0, 1)
                        base_seq['head_turn_angle'] += np.random.normal(0, 2)
                        base_seq['confidence'] += np.random.normal(0, 0.05)
                        additional_normal.append(base_seq)
                
                if additional_normal:
                    additional_df = pd.DataFrame(additional_normal)
                    df = pd.concat([df, additional_df], ignore_index=True)
                    logger.info(f"Added {len(additional_normal)} additional normal sequences for balance")
            
            output_file = self.processed_dir / "roboflow_behavior_sequences.csv"
            df.to_csv(output_file, index=False)
            
            # Log final distribution
            final_normal = len(df[df['label'] == 'normal'])
            final_suspicious = len(df[df['label'].isin(['suspicious', 'suspicious_multiple'])])
            logger.info(f"Final distribution - Normal: {final_normal}, Suspicious: {final_suspicious}")
            
            logger.info(f"✓ Processed {len(all_sequences)} sequences")
            logger.info(f"✓ Saved to: {output_file}")
            return True
        else:
            logger.warning("⚠️ No sequences extracted. Using sample data.")
            return self.download_sample_dataset()
    
    def _extract_sequences_from_coco(self, coco_data: Dict) -> List[Dict]:
        """Extract behavior sequences from COCO format annotations."""
        sequences = []
        
        # Get image and annotation data
        images = {img['id']: img for img in coco_data.get('images', [])}
        annotations = coco_data.get('annotations', [])
        categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        
        # Group annotations by image
        image_annotations = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # Process each image
        for img_id, img_data in images.items():
            img_annotations = image_annotations.get(img_id, [])
            
            # Extract behavior indicators from annotations
            behavior_data = self._analyze_image_annotations(img_data, img_annotations, categories)
            
            if behavior_data:
                sequences.append(behavior_data)
        
        return sequences
    
    def _analyze_image_annotations(self, img_data: Dict, annotations: List[Dict], 
                                 categories: Dict) -> Dict:
        """Analyze annotations to extract behavior indicators."""
        # This is a simplified extraction - you would enhance this based on
        # the actual annotation structure of the cheating dataset
        
        behavior = {
            'image_id': img_data['id'],
            'image_file': img_data.get('file_name', ''),
            'width': img_data.get('width', 0),
            'height': img_data.get('height', 0),
            'lean_flag': 0,
            'look_flag': 0,
            'phone_flag': 0,
            'lean_angle': 0.0,
            'head_turn_angle': 0.0,
            'confidence': 0.8,
            'person_id': f'person_{(img_data["id"] // 10) + 1:03d}',  # Create more person IDs (10 frames per person)
            'frame_id': img_data['id'] % 10,  # Create frame sequences within each person (0-9)
            'label': 'normal'
        }
        
        # Analyze annotations for behavior indicators based on real Roboflow categories
        suspicious_behaviors = []
        has_explicit_normal = False
        
        for ann in annotations:
            category_name = categories.get(ann['category_id'], 'unknown')
            
            # Map real Roboflow categories to behaviors with better balance:
            if category_name == 'Normal':
                has_explicit_normal = True
            elif category_name in ['Left Hand Gesture', 'Right Hand Gesture']:
                behavior['lean_flag'] = 1  # Hand gestures could indicate reaching/leaning
                suspicious_behaviors.append('hand_gesture')
            elif category_name in ['Looking Left', 'Looking Right']:
                behavior['look_flag'] = 1
                behavior['head_turn_angle'] = -20.0 if category_name == 'Looking Left' else 20.0
                suspicious_behaviors.append('looking')
            # 'poses' category is just structural, doesn't indicate behavior
        
        # Determine final label based on what we found
        if suspicious_behaviors:
            # If there are suspicious behaviors, label as suspicious regardless of "Normal" category
            if len(suspicious_behaviors) > 1:
                behavior['label'] = 'suspicious_multiple'
            else:
                behavior['label'] = 'suspicious'
        elif has_explicit_normal:
            # Only use normal if explicitly labeled AND no suspicious behaviors
            behavior['label'] = 'normal'
        else:
            # If no annotations or only 'poses', assume normal behavior
            behavior['label'] = 'normal'
        
        return behavior
    
    def get_dataset_info(self) -> Dict:
        """Get information about the downloaded dataset."""
        info = {
            'data_dir': str(self.data_dir),
            'images_available': len(list(self.images_dir.glob("**/*.jpg"))) if self.images_dir.exists() else 0,
            'annotations_available': len(list(self.annotations_dir.glob("**/*.json"))) if self.annotations_dir.exists() else 0,
            'processed_files': list(self.processed_dir.glob("*.csv")) if self.processed_dir.exists() else []
        }
        
        return info

def main():
    parser = argparse.ArgumentParser(description='Download Roboflow cheating dataset')
    parser.add_argument('--api-key', type=str, help='Roboflow API key')
    parser.add_argument('--sample-only', action='store_true', 
                       help='Create sample data only (no download)')
    parser.add_argument('--process-only', action='store_true',
                       help='Process existing downloaded dataset (no download)')
    parser.add_argument('--output-dir', type=str, default='data/roboflow',
                       help='Output directory for dataset')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = RoboflowDatasetDownloader(api_key=args.api_key)
    
    if args.sample_only:
        logger.info("🎯 Creating sample dataset only...")
        success = downloader.download_sample_dataset()
    elif args.process_only:
        logger.info("🔄 Processing existing downloaded dataset...")
        success = downloader.process_annotations_to_sequences()
    else:
        logger.info("📥 Attempting to download full Roboflow dataset...")
        
        # Try to download full dataset first
        success = downloader.download_with_roboflow_pip()
        
        if success:
            # Process downloaded annotations
            success = downloader.process_annotations_to_sequences()
        else:
            logger.info("📝 Falling back to sample dataset...")
            success = downloader.download_sample_dataset()
    
    if success:
        # Show dataset info
        info = downloader.get_dataset_info()
        logger.info("📊 Dataset Information:")
        logger.info(f"  Data directory: {info['data_dir']}")
        logger.info(f"  Images: {info['images_available']}")
        logger.info(f"  Annotations: {info['annotations_available']}")
        logger.info(f"  Processed files: {len(info['processed_files'])}")
        
        for processed_file in info['processed_files']:
            logger.info(f"    - {processed_file}")
        
        logger.info("\n✅ Dataset setup complete!")
        logger.info("Next steps:")
        logger.info("1. Run: python train_lstm.py --data_path data/roboflow/processed/")
        logger.info("2. Test the trained model with: python test_webcam_realtime.py")
        
    else:
        logger.error("❌ Failed to setup dataset")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
