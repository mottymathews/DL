#!/usr/bin/env python3
"""
Analyze class distribution in the road dataset
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from homework.datasets.road_dataset import load_data

def analyze_class_distribution():
    """Analyze the distribution of classes in the segmentation masks"""
    print("Loading dataset...")
    dataset = load_data(
        "drive_data/train",
        transform_pipeline="default", 
        return_dataloader=False,
        batch_size=1
    )
    
    class_counts = np.zeros(3)  # 3 classes: 0, 1, 2
    total_pixels = 0
    
    print(f"Analyzing {len(dataset)} samples...")
    
    # Sample a subset for faster analysis
    sample_indices = np.linspace(0, len(dataset)-1, min(100, len(dataset)), dtype=int)
    
    for i, idx in enumerate(sample_indices):
        sample = dataset[idx]
        track_mask = sample['track']  # Already a numpy array
        
        # Count pixels for each class
        unique, counts = np.unique(track_mask, return_counts=True)
        for class_id, count in zip(unique, counts):
            class_counts[int(class_id)] += count
        
        total_pixels += track_mask.size
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(sample_indices)} samples...")
    
    # Calculate percentages
    percentages = (class_counts / total_pixels) * 100
    
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*50)
    print(f"Total pixels analyzed: {total_pixels:,}")
    print()
    
    class_names = ["Background", "Left Lane", "Right Lane"]
    for i, (name, count, pct) in enumerate(zip(class_names, class_counts, percentages)):
        print(f"Class {i} ({name:12}): {count:8.0f} pixels ({pct:5.2f}%)")
    
    print("\n" + "="*50)
    print("RECOMMENDED CLASS WEIGHTS")
    print("="*50)
    
    # Calculate inverse frequency weights
    total_samples = class_counts.sum()
    inv_freq_weights = total_samples / (len(class_counts) * class_counts)
    
    # Normalize so background weight is reasonable
    normalized_weights = inv_freq_weights / inv_freq_weights[0]
    
    print("Inverse frequency weights (normalized):")
    for i, (name, weight) in enumerate(zip(class_names, normalized_weights)):
        print(f"Class {i} ({name:12}): {weight:6.2f}")
    
    print(f"\nSuggested torch.tensor: {normalized_weights}")
    
    # Show current weights from your training script
    current_weights = np.array([0.8, 10.0, 10.0])
    print(f"Your current weights:    {current_weights}")
    
    return class_counts, percentages, normalized_weights

if __name__ == "__main__":
    analyze_class_distribution()
