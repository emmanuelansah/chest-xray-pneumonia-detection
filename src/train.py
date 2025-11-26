"""
Training Script for Chest X-Ray Classification

This script provides:
- Complete training loop with validation
- Weighted loss for class imbalance
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Training history tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            confusion_matrix, classification_report)

from data_loader import get_dataloaders
from model import get_model, save_model


class Trainer:
    """
    Trainer class for chest X-ray classification
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        class_weights: Weights for handling class imbalance
        device: Device to train on
        learning_rate: Initial learning rate
        num_epochs: Number of training epochs
        patience: Patience for early stopping
        save_dir: Directory to save checkpoints
    """
    
    def __init__(self, model, train_loader, val_loader, class_weights, 
                 device='cuda', learning_rate=0.0001, num_epochs=20, 
                 patience=5, save_dir='../models'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # maximize validation accuracy
            patience=3,
            factor=0.5,
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Early stopping
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        
        print("=" * 70)
        print("TRAINER INITIALIZED")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Learning rate: {learning_rate}")
        print(f"Num epochs: {num_epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"Checkpoint directory: {self.save_dir.absolute()}")
        print("=" * 70)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        return epoch_loss, epoch_acc, precision, recall, f1
    
    def train(self):
        """Complete training loop"""
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, precision, recall, f1 = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_precision'].append(precision)
            self.history['val_recall'].append(recall)
            self.history['val_f1'].append(f1)
            self.history['learning_rates'].append(current_lr)
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"Precision:  {precision:.4f} | Recall:    {recall:.4f} | F1: {f1:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            
            # Model checkpointing
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                
                # Save best model
                checkpoint_path = self.save_dir / 'best_model.pth'
                save_model(self.model, self.optimizer, epoch, val_acc, checkpoint_path)
                print(f"★ New best model! Val Acc: {val_acc:.4f}")
            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve} epoch(s)")
            
            # Early stopping
            if self.epochs_no_improve >= self.patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                break
        
        # Training complete
        training_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total training time: {training_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Save final model
        final_path = self.save_dir / 'final_model.pth'
        save_model(self.model, self.optimizer, epoch, val_acc, final_path)
        
        # Save training history
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"✓ Training history saved to {history_path}")
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.history
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision, Recall, F1
        axes[1, 0].plot(epochs, self.history['val_precision'], 'g-', label='Precision')
        axes[1, 0].plot(epochs, self.history['val_recall'], 'b-', label='Recall')
        axes[1, 0].plot(epochs, self.history['val_f1'], 'r-', label='F1-Score')
        axes[1, 0].set_title('Validation Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(epochs, self.history['learning_rates'], 'k-')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = self.save_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"✓ Training curves saved to {plot_path}")
        
        plt.close()


def main():
    """Main training function"""
    
    # Configuration
    DATA_PATH = '../../chest_xray_organized'  # Update this path
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")
    
    # Load data
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_weighted_sampler=True
    )
    
    # Create model
    model = get_model(num_classes=2, pretrained=True, device=device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        save_dir='../models'
    )
    
    # Train
    history = trainer.train()
    
    print("\n✅ Training pipeline completed successfully!")
    print("\n🚀 Next steps:")
    print("  1. Evaluate on test set: python evaluate.py")
    print("  2. Review training curves: ../models/training_curves.png")
    print("  3. Build Gradio app for deployment")


if __name__ == "__main__":
    main()
