"""
LSTM Training Script for CheatGPT3 Behavior Classification
===========================================================

This script trains an LSTM classifier to predict exam behaviors from extracted 
pose + detection features using the Roboflow Cheating Dataset.

Dataset: https://universe.roboflow.com/rp-fcesi/cheating-n3uwx
Features: lean_flag, look_flag, phone_flag
Labels: normal, leaning, looking_around, phone_use, cheating

Usage:
    python train_lstm.py --data_path /path/to/dataset.csv --epochs 30 --batch_size 32
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BehaviorLSTM(nn.Module):
    """LSTM network for temporal behavior pattern recognition."""
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, num_classes=5, 
                 sequence_length=20, dropout=0.3):
        super(BehaviorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights with Xavier/Orthogonal initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize classifier weights
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """Forward pass through LSTM."""
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Classify
        output = self.classifier(last_output)
        return output

class BehaviorDataset(Dataset):
    """Dataset class for behavior sequences."""
    
    def __init__(self, sequences, labels, label_encoder=None):
        self.sequences = torch.FloatTensor(sequences)
        
        # Handle labels - if they're already encoded, use them directly
        if isinstance(labels[0], (int, np.integer)):
            self.labels = torch.LongTensor(labels)
            self.label_encoder = label_encoder
        else:
            # Encode string labels
            if label_encoder is None:
                self.label_encoder = LabelEncoder()
                # Ensure all labels are strings
                labels = [str(label) for label in labels]
                self.labels = torch.LongTensor(self.label_encoder.fit_transform(labels))
            else:
                self.label_encoder = label_encoder
                # Ensure all labels are strings
                labels = [str(label) for label in labels]
                self.labels = torch.LongTensor(self.label_encoder.transform(labels))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def load_dataset(data_path: str, sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load and preprocess the Roboflow dataset into sequences.
    
    Args:
        data_path: Path to CSV file or directory with processed Roboflow data
        sequence_length: Number of frames per sequence
        
    Returns:
        Tuple of (sequences, labels, label_encoder)
    """
    logger.info(f"Loading dataset from {data_path}")
    
    # Check if data_path is a directory (Roboflow processed data)
    if os.path.isdir(data_path):
        # Look for processed CSV files in the directory
        csv_files = list(Path(data_path).glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {data_path}")
        
        # Use the first CSV file found (or combine multiple if needed)
        data_path = str(csv_files[0])
        logger.info(f"Using CSV file: {data_path}")
    
    # Load the CSV data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError("Dataset must be CSV or JSON format")
    
    logger.info(f"Loaded {len(df)} rows from dataset")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Validate required columns for Roboflow dataset
    required_cols = ['frame_id', 'person_id', 'lean_flag', 'look_flag', 'phone_flag', 'label']
    
    # Check for alternative column names from Roboflow processing
    if 'behavior' in df.columns and 'label' not in df.columns:
        df['label'] = df['behavior']
        logger.info("Using 'behavior' column as 'label'")
    
    if 'sequence_id' in df.columns and 'person_id' not in df.columns:
        df['person_id'] = df['sequence_id'].apply(lambda x: f"person_{x:03d}")
        logger.info("Generated 'person_id' from 'sequence_id'")
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean the data
    df = df.dropna()
    df = df.sort_values(['person_id', 'frame_id'])
    
    # Enhanced feature engineering for Roboflow data
    feature_cols = ['lean_flag', 'look_flag', 'phone_flag']
    
    # Add angle features if available, otherwise estimate from flags
    if 'lean_angle' not in df.columns:
        # More realistic angle estimation based on lean patterns
        df['lean_angle'] = df.apply(lambda row: 
            np.random.normal(15, 3) if row['lean_flag'] else np.random.normal(0, 2), axis=1)
        logger.info("Generated lean_angle from lean_flag patterns")
    
    if 'head_turn_angle' not in df.columns:
        # More realistic head angle estimation
        df['head_turn_angle'] = df.apply(lambda row:
            np.random.normal(25, 5) if row['look_flag'] else np.random.normal(0, 3), axis=1)
        logger.info("Generated head_turn_angle from look_flag patterns")
    
    if 'confidence' not in df.columns:
        # Dynamic confidence based on behavior consistency
        df['confidence'] = df.apply(lambda row:
            0.9 if (row['lean_flag'] + row['look_flag'] + row['phone_flag']) >= 2 
            else 0.8 if (row['lean_flag'] + row['look_flag'] + row['phone_flag']) == 1
            else 0.7, axis=1)
        logger.info("Generated confidence scores based on behavior patterns")
    
    # Add temporal movement features for better body movement detection
    if 'sequence_id' in df.columns or len(df['person_id'].unique()) > 1:
        logger.info("Adding temporal movement features...")
        
        # Calculate movement velocity features
        df['lean_velocity'] = 0.0
        df['head_velocity'] = 0.0
        
        for person_id in df['person_id'].unique():
            person_data = df[df['person_id'] == person_id].sort_values('frame_id')
            
            # Calculate angle differences (movement velocity)
            if len(person_data) > 1:
                lean_diff = person_data['lean_angle'].diff().fillna(0)
                head_diff = person_data['head_turn_angle'].diff().fillna(0)
                
                df.loc[df['person_id'] == person_id, 'lean_velocity'] = lean_diff
                df.loc[df['person_id'] == person_id, 'head_velocity'] = head_diff
    
    feature_cols.extend(['lean_angle', 'head_turn_angle', 'confidence'])
    
    # Add movement velocity features if available
    if 'lean_velocity' in df.columns and 'head_velocity' in df.columns:
        feature_cols.extend(['lean_velocity', 'head_velocity'])
        logger.info("Added movement velocity features for body movement detection")
    
    # Normalize features
    for col in ['lean_angle', 'head_turn_angle']:
        if col in df.columns:
            max_val = df[col].abs().max()
            if max_val > 0:
                df[col] = df[col] / max_val
    
    # Normalize velocity features
    for col in ['lean_velocity', 'head_velocity']:
        if col in df.columns:
            max_val = df[col].abs().max()
            if max_val > 0:
                df[col] = df[col] / max_val
    
    # Create sequences with improved temporal modeling
    sequences = []
    labels = []
    
    logger.info("Creating enhanced temporal sequences...")
    
    for person_id in df['person_id'].unique():
        person_data = df[df['person_id'] == person_id].sort_values('frame_id')
        
        # Use sliding window with stride for better coverage
        stride = max(1, sequence_length // 4)  # 25% overlap
        
        for i in range(0, len(person_data) - sequence_length + 1, stride):
            sequence_data = person_data.iloc[i:i+sequence_length]
            
            # Skip sequences with too much missing data
            if len(sequence_data) < sequence_length:
                continue
            
            # Extract features
            features = sequence_data[feature_cols].values
            
            # Use majority vote for sequence label (more robust)
            sequence_labels = sequence_data['label'].tolist()
            label_counts = pd.Series(sequence_labels).value_counts()
            
            # Calculate percentages for better balance
            total_frames = len(sequence_labels)
            suspicious_count = label_counts.get('suspicious', 0) + label_counts.get('suspicious_multiple', 0)
            normal_count = label_counts.get('normal', 0)
            
            # More balanced labeling approach
            if normal_count / total_frames >= 0.5:  # 50% or more normal frames
                final_label = 'normal'
            elif suspicious_count / total_frames >= 0.4:  # 40% or more suspicious frames
                final_label = 'suspicious'
            else:  # Mixed behavior - use majority but ensure we include these too
                final_label = label_counts.index[0]
            
            # Include ALL sequences for maximum data utilization and balance
            sequences.append(features)
            labels.append(final_label)
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    logger.info(f"Created {len(sequences)} enhanced sequences of length {sequence_length}")
    logger.info(f"Feature shape: {sequences.shape}")
    logger.info(f"Features used: {feature_cols}")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Create label encoder with enhanced string handling
    labels = [str(label).strip().lower() for label in labels]  # Normalize labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    logger.info(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    return sequences, encoded_labels, label_encoder

def create_sample_dataset(output_path: str, num_samples: int = 1000):
    """Create a sample dataset for testing if no real dataset is available."""
    logger.info(f"Creating sample dataset with {num_samples} samples")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    data = []
    
    labels = ['normal', 'leaning', 'looking_around', 'phone_use', 'cheating']
    label_weights = [0.4, 0.2, 0.2, 0.1, 0.1]  # Normal behavior is most common
    
    for i in range(num_samples):
        frame_id = i % 100
        person_id = f"person_{i // 100:03d}"
        
        # Sample label based on weights
        label = np.random.choice(labels, p=label_weights)
        
        # Generate features based on label
        if label == 'normal':
            lean_flag = 0
            look_flag = 0
            phone_flag = 0
        elif label == 'leaning':
            lean_flag = 1
            look_flag = np.random.choice([0, 1], p=[0.7, 0.3])
            phone_flag = 0
        elif label == 'looking_around':
            lean_flag = np.random.choice([0, 1], p=[0.7, 0.3])
            look_flag = 1
            phone_flag = 0
        elif label == 'phone_use':
            lean_flag = np.random.choice([0, 1], p=[0.5, 0.5])
            look_flag = np.random.choice([0, 1], p=[0.5, 0.5])
            phone_flag = 1
        else:  # cheating
            lean_flag = 1
            look_flag = 1
            phone_flag = np.random.choice([0, 1], p=[0.3, 0.7])
        
        data.append({
            'frame_id': frame_id,
            'person_id': person_id,
            'lean_flag': lean_flag,
            'look_flag': look_flag,
            'phone_flag': phone_flag,
            'label': label
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Sample dataset saved to {output_path}")
    return output_path

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device, label_encoder):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    # Calculate F1 score
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # Create classification report - handle case where not all classes are present
    class_names = label_encoder.classes_
    unique_labels = sorted(set(all_targets + all_preds))
    
    if len(unique_labels) < len(class_names):
        # Only use class names for labels that actually appear
        active_class_names = [class_names[i] for i in unique_labels]
        report = classification_report(all_targets, all_preds, 
                                     target_names=active_class_names, 
                                     labels=unique_labels, 
                                     zero_division=0)
    else:
        report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    return avg_loss, accuracy, f1, report, cm, all_targets, all_preds

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_model(model, optimizer, epoch, loss, accuracy, save_path):
    """Save the model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'model_config': {
            'input_size': model.lstm.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'num_classes': model.num_classes,
            'sequence_length': model.sequence_length
        }
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")

def load_model(model_path, device='cpu'):
    """Load a trained model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with saved configuration
    config = checkpoint['model_config']
    model = BehaviorLSTM(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.2f}%")
    
    return model, checkpoint

def main():
    parser = argparse.ArgumentParser(description='Train LSTM for behavior classification using Roboflow dataset')
    parser.add_argument('--data_path', type=str, default='data/roboflow/processed',
                       help='Path to the dataset CSV file or Roboflow processed directory')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=20,
                       help='Length of input sequences (frames)')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--save_dir', type=str, default='weights',
                       help='Directory to save model weights')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create a sample dataset for testing')
    parser.add_argument('--use_roboflow', action='store_true',
                       help='Use real Roboflow dataset (requires API key)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Handle dataset selection
    if args.use_roboflow:
        logger.info("🤖 Using Roboflow dataset...")
        # Check if Roboflow data exists
        if not os.path.exists(args.data_path):
            logger.warning(f"Roboflow data not found at {args.data_path}")
            logger.info("Please run: python download_roboflow_dataset.py")
            logger.info("Falling back to sample dataset...")
            args.create_sample = True
    elif args.create_sample:
        logger.info("📝 Creating sample dataset...")
        args.data_path = create_sample_dataset('data/sample_behavior_dataset.csv', 2000)
    
    # Check if dataset exists
    if not os.path.exists(args.data_path):
        logger.warning(f"Dataset not found at {args.data_path}")
        if not args.use_roboflow:
            logger.info("Creating sample dataset for demonstration...")
            os.makedirs('data', exist_ok=True)
            args.data_path = create_sample_dataset('data/sample_behavior_dataset.csv', 2000)
        else:
            logger.error("❌ Roboflow dataset not available. Please run:")
            logger.error("   python download_roboflow_dataset.py")
            return 1
    
    # Load dataset
    try:
        sequences, labels, label_encoder = load_dataset(args.data_path, args.sequence_length)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        if not args.create_sample:
            logger.info("Creating sample dataset as fallback...")
            os.makedirs('data', exist_ok=True)
            args.data_path = create_sample_dataset('data/sample_behavior_dataset.csv', 2000)
            sequences, labels, label_encoder = load_dataset(args.data_path, args.sequence_length)
        else:
            raise
    
    # Create dataset
    dataset = BehaviorDataset(sequences, labels, label_encoder)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create model
    num_classes = len(label_encoder.classes_)
    input_size = sequences.shape[2]  # Number of features
    
    model = BehaviorLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        sequence_length=args.sequence_length,
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training setup
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, 'training_log.txt')
    best_accuracy = 0
    best_model_path = os.path.join(args.save_dir, 'lstm_behavior.pth')
    
    # Training log
    with open(log_file, 'w') as f:
        f.write(f"LSTM Behavior Classification Training Log\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Classes: {label_encoder.classes_}\n")
        f.write("="*80 + "\n\n")
    
    logger.info("Starting training...")
    
    # Training loop
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc, test_f1, test_report, test_cm, _, _ = evaluate(
            model, test_loader, criterion, device, label_encoder
        )
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Log progress
        logger.info(f'Epoch {epoch+1}/{args.epochs}:')
        logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, F1: {test_f1:.4f}')
        logger.info(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save to log file
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{args.epochs}:\n")
            f.write(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n")
            f.write(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, F1: {test_f1:.4f}\n")
            f.write(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")
            
            if (epoch + 1) % 5 == 0:  # Log detailed report every 5 epochs
                f.write(f"\nClassification Report (Epoch {epoch+1}):\n")
                f.write(test_report)
                f.write(f"\nConfusion Matrix (Epoch {epoch+1}):\n")
                f.write(str(test_cm))
                f.write("\n" + "-"*50 + "\n")
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_model(model, optimizer, epoch, test_loss, test_acc, best_model_path)
            
            # Save confusion matrix for best model
            cm_path = os.path.join(args.save_dir, 'confusion_matrix_best.png')
            plot_confusion_matrix(test_cm, label_encoder.classes_, cm_path)
        
        # Early stopping check
        if epoch > 10 and test_acc < max(test_accuracies[-10:]) - 5:
            logger.info("Early stopping triggered")
            break
    
    # Final evaluation
    logger.info("Training completed!")
    logger.info(f"Best test accuracy: {best_accuracy:.2f}%")
    
    # Load best model and do final evaluation
    best_model, checkpoint = load_model(best_model_path, device)
    final_loss, final_acc, final_f1, final_report, final_cm, _, _ = evaluate(
        best_model, test_loader, criterion, device, label_encoder
    )
    
    logger.info(f"Final evaluation - Accuracy: {final_acc:.2f}%, F1: {final_f1:.4f}")
    
    # Save final results
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"FINAL RESULTS\n")
        f.write(f"Best Accuracy: {best_accuracy:.2f}%\n")
        f.write(f"Final Accuracy: {final_acc:.2f}%\n")
        f.write(f"Final F1 Score: {final_f1:.4f}\n")
        f.write(f"\nFinal Classification Report:\n")
        f.write(final_report)
        f.write(f"\nFinal Confusion Matrix:\n")
        f.write(str(final_cm))
        f.write(f"\nCompleted: {datetime.now()}\n")
    
    # Save label encoder
    label_encoder_path = os.path.join(args.save_dir, 'label_encoder.pkl')
    import pickle
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train')
    plt.plot(test_accuracies, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Final Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training complete! Model saved to {best_model_path}")
    logger.info(f"Training log saved to {log_file}")
    logger.info(f"Label encoder saved to {label_encoder_path}")

if __name__ == "__main__":
    main()
