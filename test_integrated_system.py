#!/usr/bin/env python3
"""
Final Integration Test for Enhanced CheatGPT System
Tests the complete system with realistic LSTM (88.64% accuracy) and COCO enhancements
"""

import numpy as np
import cv2
import time
import json
from pathlib import Path
import logging
from typing import Dict, List

from realistic_lstm_integration import RealisticLSTMClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedCheatGPTSystem:
    """Complete integrated system with realistic LSTM and COCO enhancements"""
    
    def __init__(self):
        # Initialize realistic LSTM
        self.lstm_classifier = RealisticLSTMClassifier()
        
        # Detection history for temporal analysis
        self.detection_history = []
        self.frame_count = 0
        
        # Performance metrics
        self.metrics = {
            'total_frames': 0,
            'detections_made': 0,
            'lstm_predictions': 0,
            'fallback_predictions': 0,
            'average_confidence': 0.0,
            'behavior_counts': {
                'normal': 0,
                'suspicious_gesture': 0,
                'suspicious_looking': 0,
                'mixed_suspicious': 0
            }
        }
        
        logger.info("🚀 Integrated CheatGPT System initialized")
    
    def simulate_detection_data(self, behavior_type: str = "normal") -> Dict:
        """Simulate detection data for testing"""
        
        if behavior_type == "suspicious_gesture":
            return {
                'lean_flag': 0,
                'look_flag': 0,
                'phone_flag': 0,
                'gesture_flag': 1,
                'lean_angle': np.random.normal(0, 5),
                'head_turn_angle': np.random.normal(0, 10),
                'confidence': np.random.uniform(0.7, 0.95),
                'center_x': np.random.uniform(0.3, 0.7),
                'center_y': np.random.uniform(0.2, 0.6),
                'bbox_area': np.random.uniform(0.1, 0.6),
                'combined_suspicious': 1,
                'spatial_center_offset': np.random.uniform(0, 0.3)
            }
        elif behavior_type == "suspicious_looking":
            return {
                'lean_flag': 0,
                'look_flag': 1,
                'phone_flag': 0,
                'gesture_flag': 0,
                'lean_angle': np.random.normal(0, 3),
                'head_turn_angle': np.random.uniform(-45, 45),
                'confidence': np.random.uniform(0.6, 0.9),
                'center_x': np.random.uniform(0.25, 0.75),
                'center_y': np.random.uniform(0.2, 0.5),
                'bbox_area': np.random.uniform(0.08, 0.4),
                'combined_suspicious': 1,
                'spatial_center_offset': np.random.uniform(0, 0.4)
            }
        elif behavior_type == "mixed_suspicious":
            return {
                'lean_flag': 0,
                'look_flag': np.random.choice([0, 1]),
                'phone_flag': 0,
                'gesture_flag': np.random.choice([0, 1]),
                'lean_angle': np.random.normal(0, 8),
                'head_turn_angle': np.random.uniform(-30, 30),
                'confidence': np.random.uniform(0.5, 0.8),
                'center_x': np.random.uniform(0.2, 0.8),
                'center_y': np.random.uniform(0.25, 0.65),
                'bbox_area': np.random.uniform(0.06, 0.5),
                'combined_suspicious': np.random.choice([0, 1]),
                'spatial_center_offset': np.random.uniform(0, 0.5)
            }
        else:  # normal
            return {
                'lean_flag': 0,
                'look_flag': 0,
                'phone_flag': 0,
                'gesture_flag': 0,
                'lean_angle': np.random.normal(0, 2),
                'head_turn_angle': np.random.normal(0, 5),
                'confidence': np.random.uniform(0.8, 0.99),
                'center_x': np.random.uniform(0.4, 0.6),
                'center_y': np.random.uniform(0.4, 0.6),
                'bbox_area': np.random.uniform(0.15, 0.7),
                'combined_suspicious': 0,
                'spatial_center_offset': np.random.uniform(0, 0.2)
            }
    
    def process_sequence(self, behavior_sequence: List[str], sequence_length: int = 10) -> Dict:
        """Process a sequence of behaviors and get LSTM prediction"""
        
        # Generate detection sequence
        detection_sequence = []
        for behavior in behavior_sequence[-sequence_length:]:
            detection_data = self.simulate_detection_data(behavior)
            detection_sequence.append(detection_data)
        
        # Get LSTM prediction
        prediction_result = self.lstm_classifier.predict_behavior_sequence(detection_sequence)
        
        # Update metrics
        self._update_metrics(prediction_result)
        
        return {
            'sequence': behavior_sequence[-sequence_length:],
            'prediction_result': prediction_result,
            'sequence_length': len(detection_sequence),
            'timestamp': time.time()
        }
    
    def _update_metrics(self, prediction_result: Dict):
        """Update system metrics"""
        
        self.metrics['total_frames'] += 1
        self.metrics['detections_made'] += 1
        
        if prediction_result.get('coco_enhanced'):
            self.metrics['lstm_predictions'] += 1
        else:
            self.metrics['fallback_predictions'] += 1
        
        # Update confidence tracking
        confidence = prediction_result.get('confidence', 0)
        total = self.metrics['detections_made']
        self.metrics['average_confidence'] = (
            (self.metrics['average_confidence'] * (total - 1) + confidence) / total
        )
        
        # Update behavior counts
        prediction = prediction_result.get('prediction', 'normal')
        if prediction in self.metrics['behavior_counts']:
            self.metrics['behavior_counts'][prediction] += 1
    
    def run_comprehensive_test(self):
        """Run comprehensive test of the integrated system"""
        
        logger.info("🧪 Running Comprehensive Integration Test...")
        
        # Test scenarios
        test_scenarios = [
            {
                'name': 'Normal Behavior Sequence',
                'sequence': ['normal'] * 15,
                'expected': 'normal'
            },
            {
                'name': 'Gesture Detection Sequence',
                'sequence': ['normal'] * 5 + ['suspicious_gesture'] * 8 + ['normal'] * 2,
                'expected': 'suspicious_gesture'
            },
            {
                'name': 'Looking Behavior Sequence',
                'sequence': ['normal'] * 3 + ['suspicious_looking'] * 10 + ['normal'] * 2,
                'expected': 'suspicious_looking'
            },
            {
                'name': 'Mixed Suspicious Sequence',
                'sequence': ['normal'] * 2 + ['mixed_suspicious'] * 8 + ['suspicious_gesture'] * 3 + ['normal'] * 2,
                'expected': 'mixed_suspicious'
            },
            {
                'name': 'Transitional Sequence',
                'sequence': ['normal'] * 4 + ['suspicious_looking'] * 3 + ['suspicious_gesture'] * 4 + ['normal'] * 4,
                'expected': 'suspicious_gesture'  # Most recent dominant behavior
            }
        ]
        
        test_results = []
        
        print("\n=== COMPREHENSIVE INTEGRATION TEST ===")
        print(f"LSTM Model: Realistic (88.64% accuracy)")
        print(f"COCO Enhancement: {len(self.lstm_classifier.spatial_patterns)} spatial patterns")
        print(f"System Classes: {self.lstm_classifier.class_labels}")
        print()
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"Test {i}: {scenario['name']}")
            print(f"  Sequence: {scenario['sequence']}")
            
            # Process the sequence
            result = self.process_sequence(scenario['sequence'])
            prediction_result = result['prediction_result']
            
            predicted = prediction_result.get('prediction', 'unknown')
            confidence = prediction_result.get('confidence', 0)
            source = prediction_result.get('coco_enhanced', False)
            
            print(f"  Predicted: {predicted} (confidence: {confidence:.3f})")
            print(f"  Expected: {scenario['expected']}")
            print(f"  COCO Enhanced: {source}")
            print(f"  Match: {'✅' if predicted == scenario['expected'] else '❌'}")
            
            if 'auxiliary_predictions' in prediction_result:
                aux = prediction_result['auxiliary_predictions']
                print(f"  Auxiliary: gesture={aux.get('gesture_confidence', 0):.3f}, looking={aux.get('looking_confidence', 0):.3f}")
            
            test_results.append({
                'scenario': scenario['name'],
                'predicted': predicted,
                'expected': scenario['expected'],
                'confidence': confidence,
                'match': predicted == scenario['expected']
            })
            print()
        
        # Test summary
        matches = sum(1 for result in test_results if result['match'])
        accuracy = matches / len(test_results) * 100
        
        print("=== TEST SUMMARY ===")
        print(f"Test Accuracy: {accuracy:.1f}% ({matches}/{len(test_results)} scenarios)")
        print(f"Average Confidence: {self.metrics['average_confidence']:.3f}")
        print(f"LSTM Predictions: {self.metrics['lstm_predictions']}")
        print(f"Fallback Predictions: {self.metrics['fallback_predictions']}")
        
        print("\nBehavior Distribution:")
        for behavior, count in self.metrics['behavior_counts'].items():
            print(f"  {behavior}: {count}")
        
        return test_results
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        
        return {
            'system_type': 'Integrated CheatGPT with Realistic LSTM',
            'lstm_model': {
                'type': 'Enhanced Realistic LSTM',
                'accuracy': 88.64,
                'loaded': self.lstm_classifier.is_loaded,
                'classes': self.lstm_classifier.class_labels,
                'parameters': 228008
            },
            'coco_enhancement': {
                'enabled': True,
                'spatial_patterns': len(self.lstm_classifier.spatial_patterns),
                'behavior_lookup': len(self.lstm_classifier.behavior_lookup)
            },
            'performance_metrics': self.metrics,
            'features': {
                'realistic_accuracy': True,
                'spatial_analysis': True,
                'temporal_consistency': True,
                'multi_task_learning': True,
                'confidence_calibration': True
            }
        }

def main():
    """Main integration test"""
    
    # Initialize integrated system
    system = IntegratedCheatGPTSystem()
    
    # Display system info
    status = system.get_system_status()
    
    print("=== INTEGRATED CHEATGPT SYSTEM STATUS ===")
    print(f"System: {status['system_type']}")
    print(f"LSTM Accuracy: {status['lstm_model']['accuracy']}%")
    print(f"LSTM Loaded: {status['lstm_model']['loaded']}")
    print(f"COCO Patterns: {status['coco_enhancement']['spatial_patterns']}")
    print(f"Classes: {status['lstm_model']['classes']}")
    
    if status['lstm_model']['loaded']:
        print("\n✅ Full Enhanced System Ready!")
        
        # Run comprehensive test
        test_results = system.run_comprehensive_test()
        
        # Save test results
        output_path = Path("test_results/integration_test.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'system_status': status,
                'test_results': test_results,
                'timestamp': time.time()
            }, f, indent=2, default=str)
        
        print(f"\n📄 Test results saved: {output_path}")
        
    else:
        print("\n⚠️ LSTM not loaded - check model files")

if __name__ == "__main__":
    main()
