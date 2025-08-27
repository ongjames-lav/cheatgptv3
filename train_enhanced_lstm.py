#!/usr/bin/env python3
"""
Enhanced LSTM Training for 6-Category Roboflow Cheating Dataset
Utilizes the rich behavioral categories for better temporal pattern recognition
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
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBehaviorLSTM(nn.Module):
    """Enhanced LSTM for 6-category cheating behavior recognition"""
    
    def __init__(self, input_size: int = 12, hidden_size: int = 128, num_layers: int = 3, 
                 num_classes: int = 3, sequence_length: int = 10, dropout: float = 0.3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size  
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.dropout = dropout
        
        # Enhanced feature embedding layer
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, hidden_size // 2)
        )
        
        # Bidirectional LSTM for better temporal understanding
        self.lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Enhanced temporal modeling
        )
        
        # Attention mechanism for important frame focus
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced classification head with residual connections
        lstm_output_size = hidden_size * 2  # Bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Behavior-specific auxiliary heads for multi-task learning
        self.gesture_head = nn.Linear(lstm_output_size, 2)  # Gesture vs no gesture
        self.looking_head = nn.Linear(lstm_output_size, 2)  # Looking vs not looking
        
        self._init_weights()
    
    def _init_weights(self):
        """Enhanced weight initialization"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name and len(param.shape) >= 2:
                torch.nn.init.xavier_uniform_(param.data)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Feature embedding
        embedded = self.feature_embedding(x)  # [batch, seq, hidden//2]
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(embedded)  # [batch, seq, hidden*2]
        
        # Attention mechanism for important frames
        attended_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )  # [batch, seq, hidden*2]
        
        # Global max pooling and average pooling combined
        max_pooled = torch.max(attended_out, dim=1)[0]  # [batch, hidden*2]
        avg_pooled = torch.mean(attended_out, dim=1)    # [batch, hidden*2]
        
        # Combine different pooling strategies
        combined_features = max_pooled + avg_pooled  # [batch, hidden*2]
        
        # Main classification
        main_output = self.classifier(combined_features)
        
        # Auxiliary outputs for multi-task learning
        gesture_output = self.gesture_head(combined_features)
        looking_output = self.looking_head(combined_features)
        
        return {
            'main': main_output,
            'gesture': gesture_output, 
            'looking': looking_output,
            'attention_weights': attention_weights,
            'features': combined_features
        }

class EnhancedBehaviorDataset(Dataset):
    """Enhanced dataset for 6-category behavior sequences"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, 
                 gesture_labels: np.ndarray, looking_labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.gesture_labels = torch.LongTensor(gesture_labels)
        self.looking_labels = torch.LongTensor(looking_labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx],
            'gesture_label': self.gesture_labels[idx],
            'looking_label': self.looking_labels[idx]
        }

def create_enhanced_features(data: pd.DataFrame) -> np.ndarray:
    """Create enhanced 12-dimensional feature vectors"""
    
    feature_columns = [
        # Basic rule-based flags
        'lean_flag', 'look_flag', 'phone_flag', 'gesture_flag',
        
        # Continuous angle features  
        'lean_angle', 'head_turn_angle', 'confidence',
        
        # Spatial features
        'center_x', 'center_y', 'bbox_area',
        
        # Derived behavioral features
        'combined_suspicious', 'spatial_center_offset'
    ]
    
    # Ensure all columns exist
    for col in feature_columns:
        if col not in data.columns:
            logger.warning(f"Missing column {col}, filling with zeros")
            data[col] = 0.0
    
    features = data[feature_columns].values.astype(np.float32)
    
    # Enhanced normalization
    features[:, 4] = np.clip(features[:, 4] / 45.0, -1, 1)    # lean_angle
    features[:, 5] = np.clip(features[:, 5] / 90.0, -1, 1)    # head_turn_angle
    features[:, 6] = np.clip(features[:, 6], 0, 1)            # confidence
    features[:, 7] = np.clip(features[:, 7], 0, 1)            # center_x
    features[:, 8] = np.clip(features[:, 8], 0, 1)            # center_y
    features[:, 9] = np.clip(features[:, 9], 0, 1)            # bbox_area
    features[:, 11] = np.clip(features[:, 11], 0, 2)          # spatial_center_offset
    
    return features

def create_enhanced_sequences(data: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create enhanced sequences with multi-task labels"""
    
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

def train_enhanced_lstm(data_path: str = "data/roboflow/enhanced_processed/enhanced_sequences.csv"):
    """Train enhanced LSTM with multi-task learning"""
    
    logger.info("🚀 Starting Enhanced LSTM Training for 6-Category Dataset")
    
    # Load enhanced sequences
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Enhanced sequences not found: {data_path}")
    
    data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(data)} sequence frames")
    
    # Create sequences and labels
    sequences, main_labels, gesture_labels, looking_labels = create_enhanced_sequences(data)
    logger.info(f"Created {len(sequences)} sequences of length {sequences.shape[1]} with {sequences.shape[2]} features")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_main_labels = label_encoder.fit_transform(main_labels)
    
    logger.info(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Split data
    train_seq, test_seq, train_main, test_main, train_gesture, test_gesture, train_looking, test_looking = train_test_split(
        sequences, encoded_main_labels, gesture_labels, looking_labels, 
        test_size=0.2, random_state=42, stratify=encoded_main_labels
    )
    
    logger.info(f"Training sequences: {len(train_seq)}, Test sequences: {len(test_seq)}")
    
    # Create datasets
    train_dataset = EnhancedBehaviorDataset(train_seq, train_main, train_gesture, train_looking)
    test_dataset = EnhancedBehaviorDataset(test_seq, test_main, test_gesture, test_looking)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize enhanced model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedBehaviorLSTM(
        input_size=12,  # Enhanced feature size
        hidden_size=128,
        num_layers=3,
        num_classes=len(label_encoder.classes_),
        sequence_length=sequences.shape[1]
    ).to(device)
    
    logger.info(f"Model initialized on {device}")
    model_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {model_params:,}")
    
    # Enhanced loss function with multi-task learning
    main_criterion = nn.CrossEntropyLoss()
    aux_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 50
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
            
            # Combined loss with weights
            total_loss = main_loss + 0.3 * gesture_loss + 0.3 * looking_loss
            
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
                
                total_loss = main_loss + 0.3 * gesture_loss + 0.3 * looking_loss
                
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
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logger.info(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%")
        logger.info(f"  Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_accuracy:.2f}%")
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
            
            # Save enhanced model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_accuracy,
                'loss': test_loss,
                'model_config': {
                    'input_size': 12,
                    'hidden_size': 128,
                    'num_layers': 3,
                    'num_classes': len(label_encoder.classes_),
                    'sequence_length': sequences.shape[1],
                    'dropout': 0.3
                }
            }, 'weights/enhanced_lstm_behavior.pth')
            
            logger.info(f"✅ New best model saved! Accuracy: {test_accuracy:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 10:
            logger.info("Early stopping triggered")
            break
    
    # Save label encoder
    with open('weights/enhanced_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Final evaluation
    logger.info("\n=== ENHANCED LSTM TRAINING COMPLETE ===")
    logger.info(f"Best Test Accuracy: {best_accuracy:.2f}%")
    
    # Detailed classification report
    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    logger.info("✅ Enhanced LSTM ready for integration!")
    return model, label_encoder

if __name__ == "__main__":
    # Ensure weights directory exists
    Path("weights").mkdir(exist_ok=True)
    
    # Train enhanced model
    model, encoder = train_enhanced_lstm()
