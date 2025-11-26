"""
Data Loading Module for Chest X-Ray Classification

This module provides:
- Custom Dataset class for chest X-rays
- Data transformations with augmentation
- DataLoader creation with class weighting
- Utilities for handling class imbalance
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np
from collections import Counter


class ChestXRayDataset(Dataset):
    """
    Custom Dataset for Chest X-Ray images
    
    Args:
        data_dir (str): Path to data directory containing NORMAL and PNEUMONIA folders
        transform (callable, optional): Optional transform to apply to images
        class_to_idx (dict, optional): Mapping from class names to indices
    """
    
    def __init__(self, data_dir, transform=None, class_to_idx=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Define class mapping
        if class_to_idx is None:
            self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())
        
        # Collect all image paths and labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files
            for img_path in class_dir.glob('*.jpeg'):
                self.images.append(str(img_path))
                self.labels.append(class_idx)
            
            for img_path in class_dir.glob('*.jpg'):
                self.images.append(str(img_path))
                self.labels.append(class_idx)
            
            for img_path in class_dir.glob('*.png'):
                self.images.append(str(img_path))
                self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images from {self.data_dir}")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print class distribution in the dataset"""
        counter = Counter(self.labels)
        total = len(self.labels)
        
        for class_name, class_idx in self.class_to_idx.items():
            count = counter[class_idx]
            percentage = (count / total) * 100
            print(f"  {class_name:10s}: {count:4d} ({percentage:.1f}%)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self):
        """
        Calculate class weights for handling imbalance
        
        Returns:
            torch.Tensor: Class weights for weighted loss
        """
        counter = Counter(self.labels)
        total = len(self.labels)
        
        # Calculate weights inversely proportional to class frequency
        weights = []
        for class_idx in range(len(self.classes)):
            count = counter[class_idx]
            weight = total / (len(self.classes) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self):
        """
        Get sample weights for WeightedRandomSampler
        
        Returns:
            list: Weight for each sample
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for label in self.labels]
        return sample_weights


def get_transforms(train=True, img_size=224):
    """
    Get data transforms for training or validation/test
    
    Args:
        train (bool): Whether to include training augmentation
        img_size (int): Target image size (default: 224)
    
    Returns:
        torchvision.transforms.Compose: Composition of transforms
    """
    
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloaders(data_path, batch_size=32, num_workers=4, use_weighted_sampler=True):
    """
    Create DataLoaders for train, validation, and test sets
    
    Args:
        data_path (str): Path to organized dataset directory
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        use_weighted_sampler (bool): Whether to use weighted sampling for training
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights)
    """
    
    data_path = Path(data_path)
    
    print("=" * 70)
    print("CREATING DATALOADERS")
    print("=" * 70)
    print(f"Data path: {data_path}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Weighted sampling: {use_weighted_sampler}\n")
    
    # Create datasets
    train_dataset = ChestXRayDataset(
        data_dir=data_path / 'train',
        transform=get_transforms(train=True)
    )
    
    val_dataset = ChestXRayDataset(
        data_dir=data_path / 'val',
        transform=get_transforms(train=False)
    )
    
    test_dataset = ChestXRayDataset(
        data_dir=data_path / 'test',
        transform=get_transforms(train=False)
    )
    
    # Get class weights for loss function
    class_weights = train_dataset.get_class_weights()
    print(f"\nClass weights for loss function:")
    for class_name, weight in zip(train_dataset.classes, class_weights):
        print(f"  {class_name:10s}: {weight:.4f}")
    
    # Create samplers
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
        print("\n✓ Using WeightedRandomSampler for balanced training")
    else:
        sampler = None
        shuffle = True
        print("\n✓ Using standard random shuffling")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ DataLoaders created successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print("=" * 70)
    
    return train_loader, val_loader, test_loader, class_weights
