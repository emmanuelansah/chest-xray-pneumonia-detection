"""
Evaluation Script for Chest X-Ray Classification

This script provides:
- Comprehensive model evaluation
- Confusion matrix visualization
- Classification report
- ROC curve and AUC
- Per-class metrics
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            confusion_matrix, classification_report, roc_curve, auc)

from data_loader import get_dataloaders
from model import get_model, load_model


class Evaluator:
    """
    Evaluator class for model evaluation
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        class_names: List of class names
    """
    
    def __init__(self, model, test_loader, device='cuda', class_names=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        
        if class_names is None:
            self.class_names = ['NORMAL', 'PNEUMONIA']
        else:
            self.class_names = class_names
        
        print("=" * 70)
        print("EVALUATOR INITIALIZED")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Test batches: {len(test_loader)}")
        print("=" * 70)
    
    def evaluate(self):
        """
        Evaluate model on test set
        
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("\n" + "=" * 70)
        print("EVALUATING MODEL")
        print("=" * 70)
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluating')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Collect predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        results = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        # Print results
        self._print_results(results)
        
        # Generate visualizations
        self._plot_confusion_matrix(all_labels, all_preds)
        self._plot_roc_curve(all_labels, all_probs)
        
        return results
    
    def _calculate_metrics(self, labels, preds, probs):
        """Calculate all evaluation metrics"""
        results = {}
        
        # Overall metrics
        results['accuracy'] = accuracy_score(labels, preds)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None
        )
        
        results['per_class'] = {
            self.class_names[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
            for i in range(len(self.class_names))
        }
        
        # Macro averages
        results['macro_precision'] = np.mean(precision)
        results['macro_recall'] = np.mean(recall)
        results['macro_f1'] = np.mean(f1)
        
        # Weighted averages
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        results['weighted_precision'] = precision_w
        results['weighted_recall'] = recall_w
        results['weighted_f1'] = f1_w
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(labels, preds)
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        results['roc_auc'] = auc(fpr, tpr)
        results['fpr'] = fpr
        results['tpr'] = tpr
        
        return results
    
    def _print_results(self, results):
        """Print evaluation results"""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
        print(f"ROC AUC Score:    {results['roc_auc']:.4f}")
        
        print("\n" + "-" * 70)
        print("Per-Class Metrics:")
        print("-" * 70)
        
        for class_name in self.class_names:
            metrics = results['per_class'][class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  Support:   {metrics['support']}")
        
        print("\n" + "-" * 70)
        print("Average Metrics:")
        print("-" * 70)
        print(f"Macro Precision:    {results['macro_precision']:.4f}")
        print(f"Macro Recall:       {results['macro_recall']:.4f}")
        print(f"Macro F1-Score:     {results['macro_f1']:.4f}")
        print(f"\nWeighted Precision: {results['weighted_precision']:.4f}")
        print(f"Weighted Recall:    {results['weighted_recall']:.4f}")
        print(f"Weighted F1-Score:  {results['weighted_f1']:.4f}")
        
        print("\n" + "-" * 70)
        print("Confusion Matrix:")
        print("-" * 70)
        print(results['confusion_matrix'])
        print("=" * 70)
    
    def _plot_confusion_matrix(self, labels, preds, save_path='../models/confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def _plot_roc_curve(self, labels, probs, save_path='../models/roc_curve.png'):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
        
        plt.close()


def main():
    """Main evaluation function"""
    
    # Configuration
    DATA_PATH = '../../chest_xray_organized'  # Update this path
    MODEL_PATH = '../models/best_model.pth'  # Path to saved model
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")
    
    # Load data
    _, _, test_loader, _ = get_dataloaders(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_weighted_sampler=False  # No sampling for test set
    )
    
    # Create model
    model = get_model(num_classes=2, pretrained=False, device=device)
    
    # Load trained weights
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model from {model_path}")
        print(f"  Trained for {checkpoint['epoch']} epochs")
        print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}\n")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=['NORMAL', 'PNEUMONIA']
    )
    
    # Evaluate
    results = evaluator.evaluate()
    
    print("\n✅ Evaluation completed successfully!")
    print("\n📊 Generated files:")
    print("  • confusion_matrix.png")
    print("  • roc_curve.png")


if __name__ == "__main__":
    main()
