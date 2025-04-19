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

def generate_csv_files(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """Generate CSV files with splits"""
    print("Step 1: Generating CSV files with labels...")
    
    # Get all video files and their labels
    videos = []
    for video_path in Path(input_dir).glob('*.mp4'):
        label = extract_label_from_filename(video_path.name)
        if label:
            videos.append((str(video_path), label))
    
    if not videos:
        raise ValueError("No valid video files found with labels in the input directory")
    
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
    
    # Save CSV files
    for split_name, split_videos in splits.items():
        csv_path = Path(output_dir) / f'{split_name}.csv'
        with open(csv_path, 'w') as f:
            for video_path, label in split_videos:
                f.write(f"{video_path} {label}\n")
        print(f"Created {split_name}.csv with {len(split_videos)} videos")
    
    return splits

def process_videos(splits, output_dir):
    """Process and resize videos based on the CSV files"""
    print("Step 2: Processing and resizing videos...")
    video_dir = Path(output_dir) / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Process videos for each split
    for split_name, videos in splits.items():
        print(f"\nProcessing {split_name} split...")
        new_rows = []
        
        # Process each video
        for idx, (old_path, label) in enumerate(videos):
            # Create new filename in format video_XXXXXX.mp4
            new_filename = f'video_{idx:06d}.mp4'
            new_path = str(video_dir / new_filename)
            
            # Resize video if it doesn't exist
            print(f"Processing {old_path} -> {new_path}")
            resize_video(old_path, new_path)
            
            # Add to new rows with relative path
            rel_path = os.path.join('videos', new_filename)
            new_rows.append([rel_path, label])
        
        # Update the CSV file with space-separated values
        csv_path = Path(output_dir) / f'{split_name}.csv'
        with open(csv_path, 'w') as f:
            for rel_path, label in new_rows:
                f.write(f"{rel_path} {label}\n")
        
        print(f"Updated {split_name}.csv with {len(new_rows)} processed videos")

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
    
    # Step 1: Update CSV delimiters if files exist
    if any((output_dir / f"{split}.csv").exists() for split in ['train', 'val', 'test']):
        update_csv_delimiter(input_dir, output_dir)
    else:
        print("No existing CSV files found. Please run the full preprocessing first.")
        return

    # Step 2: Check if any videos need processing
    video_dir = output_dir / 'videos'
    if not video_dir.exists():
        print("Video directory doesn't exist. Please run full preprocessing first.")
        return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory containing original videos')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed videos and annotations')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of videos for training')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of videos for validation')
    args = parser.parse_args()
    
    preprocess_dataset(args.input_dir, args.output_dir, args.train_ratio, args.val_ratio)