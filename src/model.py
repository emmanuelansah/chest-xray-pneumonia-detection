"""
Model Architecture for Chest X-Ray Classification

This module provides:
- ResNet-50 based model with transfer learning
- Model initialization and configuration
- Model loading and saving utilities
"""

import torch
import torch.nn as nn
from torchvision import models


class ChestXRayModel(nn.Module):
    """
    Chest X-Ray Classification Model based on ResNet-50
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Whether to use pretrained ImageNet weights
        freeze_backbone (bool): Whether to freeze backbone layers initially
    """
    
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        super(ChestXRayModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained ResNet-50
        self.model = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Replace final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
        print(f"✓ Model initialized: ResNet-50")
        print(f"  Pretrained: {pretrained}")
        print(f"  Freeze backbone: {freeze_backbone}")
        print(f"  Output classes: {num_classes}")
        print(f"  Total parameters: {self.count_parameters():,}")
        print(f"  Trainable parameters: {self.count_trainable_parameters():,}")
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone layers"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen - all layers are now trainable")
    
    def freeze_until_layer(self, layer_name):
        """
        Freeze layers until specified layer
        
        Args:
            layer_name (str): Name of the layer to start unfreezing from
        """
        freeze = True
        for name, param in self.model.named_parameters():
            if layer_name in name:
                freeze = False
            param.requires_grad = not freeze
        
        print(f"✓ Frozen all layers until {layer_name}")


def get_model(num_classes=2, pretrained=True, device='cuda'):
    """
    Get model instance
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        device (str): Device to load model on
    
    Returns:
        ChestXRayModel: Model instance
    """
    model = ChestXRayModel(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=False
    )
    
    model = model.to(device)
    return model


def save_model(model, optimizer, epoch, val_acc, save_path):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        val_acc: Validation accuracy
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Model checkpoint saved to {save_path}")


def load_model(model, optimizer, checkpoint_path, device='cuda'):
    """
    Load model checkpoint
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        checkpoint_path: Path to checkpoint file
        device: Device to load on
    
    Returns:
        tuple: (model, optimizer, epoch, val_acc)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    
    print(f"✓ Model checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {epoch}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    
    return model, optimizer, epoch, val_acc


if __name__ == "__main__":
    """Test the model"""
    
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    model = get_model(num_classes=2, pretrained=True, device=device)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"✓ Forward pass successful!")
