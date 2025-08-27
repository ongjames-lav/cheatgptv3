#!/usr/bin/env python3
"""
Analyze the enhanced dataset to investigate 100% accuracy
"""

import pandas as pd
import numpy as np

def analyze_dataset():
    # Load the data
    data = pd.read_csv('data/roboflow/enhanced_processed/enhanced_sequences.csv')

    print('=== DATA ANALYSIS ===')
    print(f'Total records: {len(data)}')
    print(f'Unique sequence IDs: {data["sequence_id"].nunique()}')
    print()

    print('=== LABEL DISTRIBUTION ===')
    print('LSTM Label distribution:')
    print(data['lstm_label'].value_counts())
    print()

    print('Original category distribution:')
    print(data['original_category'].value_counts())
    print()

    print('=== SEQUENCE ANALYSIS ===')
    seq_lengths = data.groupby('sequence_id').size()
    print(f'Sequence lengths - Min: {seq_lengths.min()}, Max: {seq_lengths.max()}, Mean: {seq_lengths.mean():.1f}')
    print()

    print('=== FEATURE ANALYSIS ===')
    print('Gesture flag distribution:', data['gesture_flag'].value_counts())
    print('Look flag distribution:', data['look_flag'].value_counts())
    print('Lean flag distribution:', data['lean_flag'].value_counts())
    print()

    print('=== SEQUENCE LABEL DISTRIBUTION ===')
    seq_labels = data.groupby('sequence_id')['lstm_label'].first()
    print('Sequence-level label distribution:')
    print(seq_labels.value_counts())
    print()

    print('=== SUSPICIOUS PATTERNS ANALYSIS ===')
    # Check for data leakage patterns
    
    # 1. Check if features perfectly correlate with labels
    print('Feature-Label Correlations:')
    for label in data['lstm_label'].unique():
        label_data = data[data['lstm_label'] == label]
        print(f'\n{label.upper()} class patterns:')
        print(f'  Gesture flag: {label_data["gesture_flag"].mean():.3f} average')
        print(f'  Look flag: {label_data["look_flag"].mean():.3f} average')
        print(f'  Lean flag: {label_data["lean_flag"].mean():.3f} average')
        print(f'  Combined suspicious: {label_data["combined_suspicious"].mean():.3f} average')
    
    # 2. Check sequence homogeneity
    print('\n=== SEQUENCE HOMOGENEITY ===')
    mixed_sequences = 0
    for seq_id in data['sequence_id'].unique():
        seq_data = data[data['sequence_id'] == seq_id]
        unique_labels = seq_data['lstm_label'].nunique()
        if unique_labels > 1:
            mixed_sequences += 1
            print(f'Sequence {seq_id} has {unique_labels} different labels: {seq_data["lstm_label"].unique()}')
    
    print(f'Mixed sequences: {mixed_sequences} out of {data["sequence_id"].nunique()} ({mixed_sequences/data["sequence_id"].nunique()*100:.1f}%)')
    
    # 3. Check for perfect separation
    print('\n=== PERFECT SEPARATION CHECK ===')
    normal_data = data[data['lstm_label'] == 'normal']
    suspicious_data = data[data['lstm_label'] != 'normal']
    
    print(f'Normal samples with ANY suspicious flag: {(normal_data[["gesture_flag", "look_flag", "lean_flag"]].sum(axis=1) > 0).sum()}')
    print(f'Suspicious samples with NO suspicious flags: {(suspicious_data[["gesture_flag", "look_flag", "lean_flag"]].sum(axis=1) == 0).sum()}')
    
    # 4. Feature variance analysis
    print('\n=== FEATURE VARIANCE ANALYSIS ===')
    feature_cols = ['lean_angle', 'head_turn_angle', 'confidence', 'center_x', 'center_y', 'bbox_area']
    for col in feature_cols:
        if col in data.columns:
            variance = data[col].var()
            mean_val = data[col].mean()
            print(f'{col}: mean={mean_val:.3f}, variance={variance:.6f}')

if __name__ == "__main__":
    analyze_dataset()
