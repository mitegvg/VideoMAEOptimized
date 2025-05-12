import os
import subprocess
import csv
from pathlib import Path
import random
import re

def extract_label_from_filename(filename):
    """Extract label from filename format: MovieName__#timestamp_label_X.mp4"""
    try:
        # Extract the label part after '_label_' and before '.mp4'
        label = filename.split('_label_')[-1].replace('.mp4', '')
        return label
    except:
        print(f"Warning: Could not extract label from filename {filename}")
        return None

def resize_video(input_path, output_path, height=256, width=320):
    """Resize video to exact dimensions of height x width"""
    if os.path.exists(output_path):
        print(f"Video already exists, skipping: {output_path}")
        return

    cmd = [
        'ffmpeg', '-i', input_path,
        '-vf', f'scale={width}:{height}',  # Set exact dimensions
        '-c:v', 'libx264', '-preset', 'medium',
        '-c:a', 'copy',
        output_path,
        '-y'  # Overwrite output file if it exists
    ]
    subprocess.run(cmd)

def save_original_mapping(videos, output_dir):
    """Save mapping between original and new video paths"""
    mapping_path = Path(output_dir) / 'original.csv'
    with open(mapping_path, 'w') as f:
        for orig_path, new_path, label in videos:
            orig_filename = Path(orig_path).name
            f.write(f"{orig_filename} {new_path} {label}\n")
    print(f"Created original.csv with {len(videos)} video mappings")

def generate_csv_files(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """Generate CSV files with splits using new normalized paths"""
    print("Step 1: Generating CSV files with labels...")
    
    # Get all video files and their labels
    videos = []
    global_idx = 0
    
    for video_path in Path(input_dir).glob('*.mp4'):
        label = extract_label_from_filename(video_path.name)
        if label:
            # Generate new normalized filename right from the start
            new_filename = f'video_{global_idx:06d}.mp4'
            rel_path = os.path.join('videos', new_filename)
            videos.append((str(video_path), rel_path, label))
            global_idx += 1
    
    if not videos:
        raise ValueError("No valid video files found with labels in the input directory")
    
    # Save original path mapping before shuffling
    save_original_mapping(videos, output_dir)
    
    # Shuffle videos
    random.shuffle(videos)
    
    # Split into train/val/test
    n_total = len(videos)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    splits = {
        'train': videos[:n_train],
        'val': videos[n_train:n_train + n_val],
        'test': videos[n_train + n_val:]
    }
    
    # Save CSV files with new paths
    for split_name, split_videos in splits.items():
        csv_path = Path(output_dir) / f'{split_name}.csv'
        with open(csv_path, 'w') as f:
            for _, new_path, label in split_videos:
                f.write(f"{new_path} {label}\n")
        print(f"Created {split_name}.csv with {len(split_videos)} videos")
    
    return splits

def process_videos(splits, output_dir):
    """Process and resize videos based on the splits"""
    print("Step 2: Processing and resizing videos...")
    video_dir = Path(output_dir) / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Process videos for each split
    for split_name, videos in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Process each video
        for old_path, new_rel_path, label in videos:
            # Get full output path
            new_path = str(Path(output_dir) / new_rel_path)
            
            # Resize video if it doesn't exist
            print(f"Processing {old_path} -> {new_path}")
            resize_video(old_path, new_path)

def update_csv_delimiter(input_dir, output_dir):
    """Update CSV files to use space as delimiter"""
    print("Step 1: Updating CSV files with space delimiter...")
    
    for split_name in ['train', 'val', 'test']:
        csv_path = Path(output_dir) / f'{split_name}.csv'
        if csv_path.exists():
            # Read existing CSV and update delimiter
            videos = []
            with open(csv_path, 'r') as f:
                for line in f:
                    if ',' in line:  # If comma-separated
                        path, label = line.strip().split(',')
                    else:  # If space-separated or other format
                        parts = line.strip().split()
                        path, label = parts[0], parts[-1]
                    videos.append((path, label))
            
            # Write back with space delimiter
            with open(csv_path, 'w') as f:
                for path, label in videos:
                    f.write(f"{path} {label}\n")
            print(f"Updated delimiter in {split_name}.csv")

def preprocess_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Preprocess videos and create train/val/test splits
    Args:
        input_dir: Directory containing original videos
        output_dir: Directory to save processed videos and annotations
        train_ratio: Ratio of videos for training
        val_ratio: Ratio of videos for validation (remaining will be test)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate splits with normalized paths
    splits = generate_csv_files(input_dir, output_dir, train_ratio, val_ratio)
    
    # Process and resize videos using the normalized paths
    process_videos(splits, output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory containing original videos')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed videos and annotations')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of videos for training')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of videos for validation')
    args = parser.parse_args()
    
    preprocess_dataset(args.input_dir, args.output_dir, args.train_ratio, args.val_ratio)