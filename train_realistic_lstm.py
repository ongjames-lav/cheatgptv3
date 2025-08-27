#!/usr/bin/env python3
"""
Train LSTM with realistic dataset to get proper accuracy metrics
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Import our enhanced LSTM from the original training script
from train_enhanced_lstm import EnhancedBehaviorLSTM, EnhancedBehaviorDataset, create_enhanced_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_realistic_sequences(data: pd.DataFrame, sequence_length: int = 10):
    """Create sequences from realistic dataset"""
    
    sequences = []
    main_labels = []
    gesture_labels = []
    looking_labels = []
    
    # Group by sequence_id
    grouped = data.groupby('sequence_id')
    
    for seq_id, group in grouped:
        if len(group) < sequence_length:
            continue
            
        # Take first sequence_length frames
        seq_data = group.head(sequence_length)
        
        # Create feature sequence
        features = create_enhanced_features(seq_data)
        sequences.append(features)
        
        # Main label (most common in sequence)
        main_label = seq_data['lstm_label'].mode().iloc[0]
        main_labels.append(main_label)
        
        # Auxiliary labels for multi-task learning
        gesture_flag = int(seq_data['gesture_flag'].mean() > 0.3)  # Gesture present
        looking_flag = int(seq_data['look_flag'].mean() > 0.3)     # Looking present
        
        gesture_labels.append(gesture_flag)
        looking_labels.append(looking_flag)
    
    return np.array(sequences), np.array(main_labels), np.array(gesture_labels), np.array(looking_labels)

def train_realistic_lstm():
    """Train LSTM with realistic dataset"""
    
    logger.info("🚀 Training LSTM with Realistic Dataset")
    
    # Load realistic dataset
    data_path = "data/roboflow/realistic_processed/realistic_sequences.csv"
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Realistic dataset not found: {data_path}")
    
    data = pd.read_csv(data_path)
    logger.info(f"Loaded realistic dataset: {len(data)} records")
    
    # Create sequences and labels
    sequences, main_labels, gesture_labels, looking_labels = create_realistic_sequences(data)
    logger.info(f"Created {len(sequences)} sequences with {sequences.shape[2]} features")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_main_labels = label_encoder.fit_transform(main_labels)
    
    logger.info(f"Label classes: {label_encoder.classes_}")
    logger.info(f"Label distribution: {np.bincount(encoded_main_labels)}")
    
    # Split data
    train_seq, test_seq, train_main, test_main, train_gesture, test_gesture, train_looking, test_looking = train_test_split(
        sequences, encoded_main_labels, gesture_labels, looking_labels, 
        test_size=0.2, random_state=42, stratify=encoded_main_labels
    )
    
    logger.info(f"Training: {len(train_seq)}, Testing: {len(test_seq)}")
    
    # Create datasets
    train_dataset = EnhancedBehaviorDataset(train_seq, train_main, train_gesture, train_looking)
    test_dataset = EnhancedBehaviorDataset(test_seq, test_main, test_gesture, test_looking)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Smaller batch for realistic data
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedBehaviorLSTM(
        input_size=12,
        hidden_size=64,  # Reduced size for realistic dataset
        num_layers=2,    # Fewer layers to prevent overfitting
        num_classes=len(label_encoder.classes_),
        sequence_length=sequences.shape[1],
        dropout=0.4      # Higher dropout for regularization
    ).to(device)
    
    logger.info(f"Model on {device}, Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    main_criterion = nn.CrossEntropyLoss()
    aux_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.02)  # Lower LR, higher weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    num_epochs = 100
    best_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            main_labels = batch['label'].to(device)
            gesture_labels = batch['gesture_label'].float().to(device)
            looking_labels = batch['looking_label'].float().to(device)
            
            optimizer.zero_grad()
            
            outputs = model(sequences)
            
            # Multi-task loss
            main_loss = main_criterion(outputs['main'], main_labels)
            gesture_loss = aux_criterion(outputs['gesture'][:, 1], gesture_labels)
            looking_loss = aux_criterion(outputs['looking'][:, 1], looking_labels)
            
            # Combined loss
            total_loss = main_loss + 0.2 * gesture_loss + 0.2 * looking_loss  # Reduced auxiliary weights
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            _, predicted = torch.max(outputs['main'], 1)
            train_total += main_labels.size(0)
            train_correct += (predicted == main_labels).sum().item()
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['sequence'].to(device)
                main_labels = batch['label'].to(device)
                gesture_labels = batch['gesture_label'].float().to(device)
                looking_labels = batch['looking_label'].float().to(device)
                
                outputs = model(sequences)
                
                main_loss = main_criterion(outputs['main'], main_labels)
                gesture_loss = aux_criterion(outputs['gesture'][:, 1], gesture_labels)
                looking_loss = aux_criterion(outputs['looking'][:, 1], looking_labels)
                
                total_loss = main_loss + 0.2 * gesture_loss + 0.2 * looking_loss
                
                test_loss += total_loss.item()
                _, predicted = torch.max(outputs['main'], 1)
                test_total += main_labels.size(0)
                test_correct += (predicted == main_labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(main_labels.cpu().numpy())
        
        # Calculate accuracies
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        if epoch % 5 == 0:  # Print every 5 epochs
            logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
            logger.info(f"  Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_accuracy:.2f}%")
            logger.info(f"  Test: Loss={test_loss/len(test_loader):.4f}, Acc={test_accuracy:.2f}%")
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
            
            # Save realistic model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_accuracy,
                'loss': test_loss,
                'model_config': {
                    'input_size': 12,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'num_classes': len(label_encoder.classes_),
                    'sequence_length': sequences.shape[1],
                    'dropout': 0.4
                }
            }, 'weights/realistic_lstm_behavior.pth')
            
            logger.info(f"✅ New best model: {test_accuracy:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 15:  # More patience for realistic data
            logger.info("Early stopping triggered")
            break
    
    # Save label encoder
    with open('weights/realistic_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Final evaluation
    logger.info("\n=== REALISTIC LSTM TRAINING COMPLETE ===")
    logger.info(f"Best Test Accuracy: {best_accuracy:.2f}%")
    
    # Classification report
    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    return model, label_encoder, best_accuracy

if __name__ == "__main__":
    Path("weights").mkdir(exist_ok=True)
    model, encoder, accuracy = train_realistic_lstm()
    
    print(f"\n🎯 REALISTIC ACCURACY: {accuracy:.2f}%")
    print("This should be much more realistic than 100%!")
