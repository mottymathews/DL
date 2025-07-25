#!/usr/bin/env python3
"""
Script to explore info.npz files in the drive dataset
cd "/Users/mottymathews/Documents/Personal/UT Austin/DL/Homework/homework3" && python3 explore_npz.py --file "drive_data/train/cornfield_crossing_00/info.npz"
"""

import numpy as np
import argparse
from pathlib import Path

def explore_npz(npz_path):
    """
    Load and explore the contents of an NPZ file
    
    Args:
        npz_path: Path to the .npz file
    """
    try:
        # Load the NPZ file (allow pickle for object arrays)
        data = np.load(npz_path, allow_pickle=True)
        
        print(f"📁 File: {npz_path}")
        print(f"📊 Keys in NPZ file: {list(data.keys())}")
        print("-" * 50)
        
        # Explore each array in the file
        for key in data.keys():
            array = data[key]
            print(f"🔑 Key: '{key}'")
            print(f"   📏 Shape: {array.shape}")
            print(f"   🎯 Data type: {array.dtype}")
            
            if array.size > 0:
                if array.dtype.kind in ['f', 'i']:  # numeric data
                    print(f"   📈 Min: {np.min(array):.4f}")
                    print(f"   📈 Max: {np.max(array):.4f}")
                    print(f"   📈 Mean: {np.mean(array):.4f}")
                    
                if array.size <= 20:  # Small arrays - show full content
                    print(f"   📋 Content: {array}")
                else:  # Large arrays - show sample
                    if len(array.shape) == 1:
                        print(f"   📋 Sample: {array[:5]}...")
                    elif len(array.shape) == 2:
                        print(f"   📋 Sample (first 3 rows):")
                        print(f"       {array[:3]}")
                    else:
                        print(f"   📋 Sample shape: {array[:2].shape}")
            
            print()
        
        # Close the file
        data.close()
        
    except Exception as e:
        print(f"❌ Error loading {npz_path}: {e}")

def list_cornfield_files():
    """List all cornfield_crossing info.npz files"""
    base_path = Path("drive_data")
    
    if not base_path.exists():
        print("❌ drive_data directory not found!")
        print("💡 Make sure you're in the homework3 directory")
        return []
    
    # Find all cornfield_crossing info.npz files
    cornfield_files = []
    for split in ["train", "val"]:
        split_path = base_path / split
        if split_path.exists():
            for folder in split_path.iterdir():
                if folder.is_dir() and "cornfield_crossing" in folder.name:
                    info_file = folder / "info.npz"
                    if info_file.exists():
                        cornfield_files.append(info_file)
    
    return sorted(cornfield_files)

def main():
    parser = argparse.ArgumentParser(description="Explore info.npz files in drive dataset")
    parser.add_argument("--file", type=str, help="Specific NPZ file to explore")
    parser.add_argument("--all", action="store_true", help="Explore all cornfield_crossing files")
    parser.add_argument("--list", action="store_true", help="List all available cornfield files")
    
    args = parser.parse_args()
    
    if args.list:
        files = list_cornfield_files()
        print("🌽 Available cornfield_crossing info.npz files:")
        for i, file_path in enumerate(files, 1):
            print(f"  {i}. {file_path}")
        return
    
    if args.file:
        # Explore specific file
        file_path = Path(args.file)
        if file_path.exists():
            explore_npz(file_path)
        else:
            print(f"❌ File not found: {args.file}")
    
    elif args.all:
        # Explore all cornfield files
        files = list_cornfield_files()
        if not files:
            print("❌ No cornfield_crossing files found!")
            return
            
        for file_path in files:
            explore_npz(file_path)
            print("=" * 70)
    
    else:
        # Default: explore first cornfield file
        files = list_cornfield_files()
        if files:
            print("🌽 Exploring first cornfield_crossing file...")
            explore_npz(files[0])
        else:
            print("❌ No cornfield_crossing files found!")
            print("💡 Use --list to see available files")

if __name__ == "__main__":
    main()
