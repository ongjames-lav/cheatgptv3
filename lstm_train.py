#!/usr/bin/env python3
"""
Enhanced 3-Class LSTM Training Script for Cheating Detection
Classes: 0=normal, 1=suspicious, 2=cheating
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import pickle
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThreeClassBehaviorDataset(Dataset):
    """Dataset for 3-class behavior classification"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx]
        }

class ThreeClassBehaviorLSTM(nn.Module):
    """Enhanced LSTM for 3-class behavior classification"""
    
    def __init__(self, input_size, hidden_dim, num_layers, dropout=0.3, sequence_length=10):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Dropout and batch norm
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        
        # Final classification layer - 3 classes: normal, suspicious, cheating
        self.fc = nn.Linear(hidden_dim * 2, 3)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(attn_out, dim=1)
        
        # Batch normalization and dropout
        pooled = self.batch_norm(pooled)
        pooled = self.dropout(pooled)
        
        # Final classification
        output = self.fc(pooled)
        
        return output

def create_enhanced_features(data: pd.DataFrame) -> np.ndarray:
    """Create enhanced feature vectors from behavior data"""
    
    features = []
    
    for _, row in data.iterrows():
        feature_vector = [
            # Head pose features
            row.get('head_yaw', 0.0),
            row.get('head_pitch', 0.0),
            row.get('head_roll', 0.0),
            
            # Body pose features
            row.get('lean_angle', 0.0),
            row.get('shoulder_tilt', 0.0),
            
            # Movement features
            row.get('movement_speed', 0.0),
            row.get('stability', 1.0),
            
            # Behavioral flags
            float(row.get('look_flag', 0)),
            float(row.get('lean_flag', 0)),
            float(row.get('gesture_flag', 0)),
            float(row.get('phone_flag', 0)),
            
            # Context
            row.get('confidence_score', 0.5)
        ]
        
        features.append(feature_vector)
    
    return np.array(features)

def create_three_class_sequences(data: pd.DataFrame, sequence_length: int = 10):
    """Create sequences with 3-class labels: normal, suspicious, cheating"""
    
    sequences = []
    labels = []
    
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
        
        # Determine 3-class label based on behavior patterns
        look_ratio = seq_data['look_flag'].mean()
        lean_ratio = seq_data['lean_flag'].mean()
        gesture_ratio = seq_data['gesture_flag'].mean()
        phone_ratio = seq_data['phone_flag'].mean()
        
        # 3-class classification logic (more balanced)
        if (look_ratio >= 0.5 and lean_ratio >= 0.5) or (gesture_ratio >= 0.7) or (phone_ratio >= 0.7):
            # High confidence cheating: multiple behaviors or strong single behavior
            label = 2  # cheating
        elif (look_ratio >= 0.3 or lean_ratio >= 0.3 or gesture_ratio >= 0.4 or phone_ratio >= 0.4):
            # Suspicious: some concerning behaviors
            label = 1  # suspicious
        else:
            # Normal behavior
            label = 0  # normal
        
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

def compute_class_weights(labels, device):
    """Compute class weights for handling imbalanced dataset"""
    counts = Counter(labels)
    total = sum(counts.values())
    
    # Inverse frequency weighting
    weights = [total / counts[c] for c in sorted(counts.keys())]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    logger.info(f"Class distribution: {dict(counts)}")
    logger.info(f"Class weights: {weights}")
    
    return class_weights

def create_weighted_sampler(labels):
    """Create weighted random sampler for balanced training"""
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Remove unused subplots
    ax3.remove()
    ax4.remove()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Training curves saved to: {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - 3-Class Behavior Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to: {save_path}")

def plot_per_class_metrics(precision, recall, f1, class_names, save_path):
    """Plot per-class precision, recall, and F1-score"""
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Per-class metrics plot saved to: {save_path}")

def early_stopping_check(val_losses, patience=5):
    """Check if early stopping should be triggered"""
    if len(val_losses) < patience + 1:
        return False
    
    # Check if validation loss hasn't improved for 'patience' epochs
    best_loss = min(val_losses[:-patience])
    recent_losses = val_losses[-patience:]
    
    return all(loss >= best_loss for loss in recent_losses)

def save_training_report(report_text, metrics_dict, save_path):
    """Save comprehensive training report"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("3-CLASS BEHAVIOR CLASSIFICATION TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Training Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CLASS MAPPING:\n")
        f.write("0 -> normal\n")
        f.write("1 -> suspicious\n")
        f.write("2 -> cheating\n\n")
        
        f.write("FINAL METRICS:\n")
        f.write(f"Best Validation Accuracy: {metrics_dict['best_accuracy']:.2f}%\n")
        f.write(f"Final Training Accuracy: {metrics_dict['final_train_acc']:.2f}%\n")
        f.write(f"Total Epochs: {metrics_dict['total_epochs']}\n\n")
        
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report_text)
        f.write("\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write(str(metrics_dict['confusion_matrix']))
        f.write("\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        for i, class_name in enumerate(['normal', 'suspicious', 'cheating']):
            f.write(f"{class_name.upper()}:\n")
            f.write(f"  Precision: {metrics_dict['precision'][i]:.3f}\n")
            f.write(f"  Recall: {metrics_dict['recall'][i]:.3f}\n")
            f.write(f"  F1-Score: {metrics_dict['f1'][i]:.3f}\n\n")
    
    logger.info(f"Training report saved to: {save_path}")

def train_three_class_lstm():
    """Train 3-class LSTM with all enhancements"""
    
    logger.info("🚀 Training 3-Class LSTM for Cheating Detection")
    logger.info("Classes: 0=normal, 1=suspicious, 2=cheating")
    
    # Create directories
    Path("weights").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Load dataset
    data_path = "data/roboflow/realistic_processed/realistic_sequences.csv"
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    data = pd.read_csv(data_path)
    logger.info(f"Loaded dataset: {len(data)} records")
    
    # Create 3-class sequences
    sequences, labels = create_three_class_sequences(data, sequence_length=10)
    logger.info(f"Created {len(sequences)} sequences with {sequences.shape[2]} features")
    
    # Class distribution
    class_counts = Counter(labels)
    logger.info(f"Class distribution: {dict(class_counts)}")
    
    # Split data
    train_seq, val_seq, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Training: {len(train_seq)}, Validation: {len(val_seq)}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Compute class weights for loss function
    class_weights = compute_class_weights(train_labels, device)
    
    # Create datasets
    train_dataset = ThreeClassBehaviorDataset(train_seq, train_labels)
    val_dataset = ThreeClassBehaviorDataset(val_seq, val_labels)
    
    # Create weighted sampler for balanced training
    weighted_sampler = create_weighted_sampler(train_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        sampler=weighted_sampler,
        num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize model
    model = ThreeClassBehaviorLSTM(
        input_size=12,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
        sequence_length=10
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Loss function with class weights and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_accuracy = 0.0
    patience = 5
    
    logger.info("Starting training...")
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Logging
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
            logger.info(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_accuracy:.2f}%")
            logger.info(f"  Val: Loss={avg_val_loss:.4f}, Acc={val_accuracy:.2f}%")
            
            # Per-class metrics for validation
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average=None, zero_division=0
            )
            class_names = ['normal', 'suspicious', 'cheating']
            for i, name in enumerate(class_names):
                logger.info(f"  {name}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_accuracy,
                'loss': avg_val_loss,
                'class_weights': class_weights,
                'model_config': {
                    'input_size': 12,
                    'hidden_dim': 128,
                    'num_layers': 3,
                    'dropout': 0.3,
                    'sequence_length': 10
                }
            }, 'weights/lstm_cheatgpt3.pt')
            
            logger.info(f"✅ New best model saved: {val_accuracy:.2f}%")
        
        # Early stopping check
        if early_stopping_check(val_losses, patience):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Final evaluation and reporting
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - FINAL EVALUATION")
    logger.info("="*60)
    
    # Load best model for final evaluation
    checkpoint = torch.load('weights/lstm_cheatgpt3.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final predictions
    model.eval()
    final_predictions = []
    final_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(sequences)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            final_predictions.extend(predicted.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())
    
    # Generate comprehensive metrics
    class_names = ['normal', 'suspicious', 'cheating']
    
    # Classification report
    report = classification_report(
        final_labels, final_predictions, 
        target_names=class_names, 
        digits=3
    )
    
    # Confusion matrix
    cm = confusion_matrix(final_labels, final_predictions)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        final_labels, final_predictions, average=None, zero_division=0
    )
    
    # Overall accuracy
    overall_accuracy = 100 * np.sum(np.array(final_predictions) == np.array(final_labels)) / len(final_labels)
    
    # Print final metrics
    logger.info(f"\nBest Validation Accuracy: {best_accuracy:.2f}%")
    logger.info(f"Final Overall Accuracy: {overall_accuracy:.2f}%")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # Print per-class metrics clearly
    logger.info("\nPER-CLASS PERFORMANCE:")
    for i, class_name in enumerate(class_names):
        logger.info(f"{class_name.upper()}:")
        logger.info(f"  Precision: {precision[i]:.3f}")
        logger.info(f"  Recall: {recall[i]:.3f}")
        logger.info(f"  F1-Score: {f1[i]:.3f}")
        logger.info(f"  Support: {support[i]}")
    
    # Save visualizations
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        'results/training_curves.png'
    )
    
    plot_confusion_matrix(
        final_labels, final_predictions, class_names,
        'results/confusion_matrix.png'
    )
    
    plot_per_class_metrics(
        precision, recall, f1, class_names,
        'results/per_class_metrics.png'
    )
    
    # Save comprehensive training report
    metrics_dict = {
        'best_accuracy': best_accuracy,
        'final_train_acc': train_accs[-1],
        'total_epochs': len(train_losses),
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    save_training_report(report, metrics_dict, 'logs/training_report.txt')
    
    # Save label encoder (for 3 classes)
    label_mapping = {0: 'normal', 1: 'suspicious', 2: 'cheating'}
    with open('weights/three_class_label_mapping.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)
    
    logger.info("\n🎯 3-CLASS LSTM TRAINING COMPLETED SUCCESSFULLY!")
    logger.info(f"📊 Best Accuracy: {best_accuracy:.2f}%")
    logger.info("📁 Files saved:")
    logger.info("   - weights/lstm_cheatgpt3.pt")
    logger.info("   - results/confusion_matrix.png")
    logger.info("   - results/training_curves.png")
    logger.info("   - results/per_class_metrics.png")
    logger.info("   - logs/training_report.txt")
    
    return model, best_accuracy

if __name__ == "__main__":
    model, accuracy = train_three_class_lstm()
    print(f"\n🎯 FINAL 3-CLASS ACCURACY: {accuracy:.2f}%")
    print("✅ Training completed with comprehensive evaluation!")
