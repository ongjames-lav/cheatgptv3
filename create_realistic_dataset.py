#!/usr/bin/env python3
"""
Create a more realistic dataset by adding noise, transitions, and ambiguity
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_realistic_dataset():
    """Create a more realistic version of the enhanced dataset"""
    
    print("🔧 Creating Realistic Enhanced Dataset...")
    
    # Load original data
    data = pd.read_csv('data/roboflow/enhanced_processed/enhanced_sequences.csv')
    print(f"Original data: {len(data)} records, {data['sequence_id'].nunique()} sequences")
    
    # Create realistic copy
    realistic_data = data.copy()
    
    # 1. Add noise to perfect binary flags
    np.random.seed(42)
    
    # Add 10% noise to gesture flags
    gesture_noise = np.random.random(len(realistic_data)) < 0.1
    realistic_data.loc[gesture_noise, 'gesture_flag'] = 1 - realistic_data.loc[gesture_noise, 'gesture_flag']
    
    # Add 8% noise to look flags  
    look_noise = np.random.random(len(realistic_data)) < 0.08
    realistic_data.loc[look_noise, 'look_flag'] = 1 - realistic_data.loc[look_noise, 'look_flag']
    
    # 2. Add confidence variation
    confidence_noise = np.random.normal(0, 0.05, len(realistic_data))
    realistic_data['confidence'] = np.clip(realistic_data['confidence'] + confidence_noise, 0.3, 1.0)
    
    # 3. Create temporal transitions within sequences
    for seq_id in realistic_data['sequence_id'].unique():
        seq_mask = realistic_data['sequence_id'] == seq_id
        seq_data = realistic_data[seq_mask].copy()
        
        # Add some temporal transitions (30% chance)
        if np.random.random() < 0.3 and len(seq_data) >= 5:
            # Create transition in middle frames
            mid_start = len(seq_data) // 3
            mid_end = 2 * len(seq_data) // 3
            
            # Get indices for transition region
            seq_indices = seq_data.index.tolist()
            transition_indices = seq_indices[mid_start:mid_end]
            
            # Randomly reduce flags in transition region
            for idx in transition_indices:
                if np.random.random() < 0.3:  # 30% chance to zero out
                    realistic_data.loc[idx, 'gesture_flag'] = 0
                if np.random.random() < 0.3:  # 30% chance to zero out
                    realistic_data.loc[idx, 'look_flag'] = 0
    
    # 4. Update combined_suspicious based on modified flags
    realistic_data['combined_suspicious'] = (
        realistic_data['gesture_flag'] | 
        realistic_data['look_flag'] | 
        realistic_data['lean_flag']
    ).astype(int)
    
    # 5. Create more realistic labels based on sequence majority
    for seq_id in realistic_data['sequence_id'].unique():
        seq_mask = realistic_data['sequence_id'] == seq_id
        seq_data = realistic_data[seq_mask]
        
        # Calculate behavior ratios
        gesture_ratio = seq_data['gesture_flag'].mean()
        look_ratio = seq_data['look_flag'].mean()
        lean_ratio = seq_data['lean_flag'].mean()
        
        # Assign label based on predominant behavior (with thresholds)
        if gesture_ratio > 0.6:
            new_label = 'suspicious_gesture'
        elif look_ratio > 0.6:
            new_label = 'suspicious_looking'
        elif lean_ratio > 0.6:
            new_label = 'suspicious_lean'
        elif (gesture_ratio + look_ratio + lean_ratio) > 0.3:
            new_label = 'mixed_suspicious'  # New mixed category
        else:
            new_label = 'normal'
        
        realistic_data.loc[seq_mask, 'lstm_label'] = new_label
    
    # 6. Add some angle variations
    angle_noise = np.random.normal(0, 3, len(realistic_data))
    realistic_data['lean_angle'] += angle_noise
    
    head_angle_noise = np.random.normal(0, 5, len(realistic_data))
    realistic_data['head_turn_angle'] += head_angle_noise
    
    # 7. Update sequence behavior labels
    for seq_id in realistic_data['sequence_id'].unique():
        seq_mask = realistic_data['sequence_id'] == seq_id
        predominant_label = realistic_data[seq_mask]['lstm_label'].mode().iloc[0]
        realistic_data.loc[seq_mask, 'sequence_behavior'] = predominant_label
    
    # Save realistic dataset
    output_dir = Path('data/roboflow/realistic_processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'realistic_sequences.csv'
    realistic_data.to_csv(output_path, index=False)
    
    print(f"✅ Realistic dataset saved: {output_path}")
    
    # Analysis of new dataset
    print("\n=== REALISTIC DATASET ANALYSIS ===")
    print(f"Total records: {len(realistic_data)}")
    print(f"Unique sequence IDs: {realistic_data['sequence_id'].nunique()}")
    print()
    
    print("New label distribution:")
    print(realistic_data['lstm_label'].value_counts())
    print()
    
    print("Feature correlations (should be less perfect):")
    for label in realistic_data['lstm_label'].unique():
        label_data = realistic_data[realistic_data['lstm_label'] == label]
        print(f'\n{label.upper()}:')
        print(f'  Gesture flag: {label_data["gesture_flag"].mean():.3f} average')
        print(f'  Look flag: {label_data["look_flag"].mean():.3f} average')
        print(f'  Lean flag: {label_data["lean_flag"].mean():.3f} average')
    
    # Mixed sequences check
    mixed_sequences = 0
    for seq_id in realistic_data['sequence_id'].unique():
        seq_data = realistic_data[realistic_data['sequence_id'] == seq_id]
        unique_labels = seq_data['lstm_label'].nunique()
        if unique_labels > 1:
            mixed_sequences += 1
    
    print(f'\nMixed sequences: {mixed_sequences} out of {realistic_data["sequence_id"].nunique()} ({mixed_sequences/realistic_data["sequence_id"].nunique()*100:.1f}%)')
    
    return realistic_data

if __name__ == "__main__":
    create_realistic_dataset()
