#!/usr/bin/env python3
"""
Visualization script to understand the road detection dataset
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from homework.datasets.road_dataset import load_data

def visualize_road_sample(sample_idx=0, dataset_path="drive_data/train"):
    """
    Visualize a sample from the road dataset to understand:
    1. Input image
    2. Depth map 
    3. Segmentation mask (track boundaries)
    """
    # Load dataset
    dataset = load_data(
        dataset_path,
        transform_pipeline="default", 
        return_dataloader=False,
        batch_size=1
    )
    
    # Get a sample
    sample = dataset[sample_idx]
    
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Depth shape: {sample['depth'].shape}")
    print(f"Track shape: {sample['track'].shape}")
    
    # Convert to numpy and reorder dimensions if needed
    image = sample['image'] 
    if hasattr(image, 'permute'):  # PyTorch tensor
        image = image.permute(1, 2, 0).numpy()  # (H, W, C)
    else:  # Already numpy
        image = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    
    depth = sample['depth'] if not hasattr(sample['depth'], 'numpy') else sample['depth'].numpy()
    track = sample['track'] if not hasattr(sample['track'], 'numpy') else sample['track'].numpy()
    
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"Track labels: {np.unique(track)}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image (96x128)")
    axes[0, 0].axis('off')
    
    # Depth map
    depth_vis = axes[0, 1].imshow(depth, cmap='plasma', vmin=0, vmax=1)
    axes[0, 1].set_title("Depth Map (0=close, 1=far)")
    axes[0, 1].axis('off')
    plt.colorbar(depth_vis, ax=axes[0, 1], fraction=0.046)
    
    # Track segmentation
    track_vis = axes[1, 0].imshow(track, cmap='tab10', vmin=0, vmax=2)
    axes[1, 0].set_title("Track Segmentation\n(0=bg, 1=left boundary, 2=right boundary)")
    axes[1, 0].axis('off')
    plt.colorbar(track_vis, ax=axes[1, 0], fraction=0.046)
    
    # Overlay visualization
    axes[1, 1].imshow(image)
    # Overlay track boundaries with transparency
    overlay = np.zeros_like(image)
    overlay[track == 1] = [1, 0, 0]  # Red for left boundary
    overlay[track == 2] = [0, 0, 1]  # Blue for right boundary
    axes[1, 1].imshow(overlay, alpha=0.5)
    axes[1, 1].set_title("Image + Track Boundaries Overlay")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('road_sample_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print class distribution
    total_pixels = track.shape[0] * track.shape[1]
    bg_pixels = np.sum(track == 0)
    left_pixels = np.sum(track == 1) 
    right_pixels = np.sum(track == 2)
    
    print(f"\nClass Distribution:")
    print(f"Background (0): {bg_pixels:5d} pixels ({bg_pixels/total_pixels*100:.1f}%)")
    print(f"Left boundary (1): {left_pixels:3d} pixels ({left_pixels/total_pixels*100:.1f}%)")
    print(f"Right boundary (2): {right_pixels:3d} pixels ({right_pixels/total_pixels*100:.1f}%)")
    
    return sample

def analyze_dataset_statistics(dataset_path="drive_data/train", num_samples=50):
    """
    Analyze statistics across multiple samples
    """
    dataset = load_data(
        dataset_path,
        transform_pipeline="default", 
        return_dataloader=False,
        batch_size=1
    )
    
    depth_stats = []
    class_distributions = []
    
    print(f"Analyzing {min(num_samples, len(dataset))} samples...")
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        depth = sample['depth'] if not hasattr(sample['depth'], 'numpy') else sample['depth'].numpy()
        track = sample['track'] if not hasattr(sample['track'], 'numpy') else sample['track'].numpy()
        
        depth_stats.append([depth.mean(), depth.std(), depth.min(), depth.max()])
        
        total_pixels = track.size
        class_dist = [
            np.sum(track == 0) / total_pixels,  # background
            np.sum(track == 1) / total_pixels,  # left boundary  
            np.sum(track == 2) / total_pixels   # right boundary
        ]
        class_distributions.append(class_dist)
    
    depth_stats = np.array(depth_stats)
    class_distributions = np.array(class_distributions)
    
    print(f"\nDepth Statistics (across {len(depth_stats)} samples):")
    print(f"Mean depth: {depth_stats[:, 0].mean():.3f} ± {depth_stats[:, 0].std():.3f}")
    print(f"Depth std: {depth_stats[:, 1].mean():.3f} ± {depth_stats[:, 1].std():.3f}")
    print(f"Min depth: {depth_stats[:, 2].mean():.3f} ± {depth_stats[:, 2].std():.3f}")
    print(f"Max depth: {depth_stats[:, 3].mean():.3f} ± {depth_stats[:, 3].std():.3f}")
    
    print(f"\nClass Distribution (across {len(class_distributions)} samples):")
    print(f"Background: {class_distributions[:, 0].mean()*100:.1f}% ± {class_distributions[:, 0].std()*100:.1f}%")
    print(f"Left boundary: {class_distributions[:, 1].mean()*100:.1f}% ± {class_distributions[:, 1].std()*100:.1f}%")
    print(f"Right boundary: {class_distributions[:, 2].mean()*100:.1f}% ± {class_distributions[:, 2].std()*100:.1f}%")

if __name__ == "__main__":
    print("=== Road Detection Dataset Visualization ===")
    print("\n1. Visualizing a sample...")
    try:
        sample = visualize_road_sample(sample_idx=0)
        
        print("\n2. Analyzing dataset statistics...")
        analyze_dataset_statistics(num_samples=20)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have downloaded the drive_data dataset!")
