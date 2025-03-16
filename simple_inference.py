import torch
import cv2
import numpy as np
import time
import pandas as pd
import os
import sys
import argparse
import json
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, VideoMAEConfig
from collections import OrderedDict

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_args():
    parser = argparse.ArgumentParser('VideoMAE inference script for video classification')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to sample')
    parser.add_argument('--sampling_rate', type=int, default=4, help='Frame sampling rate')
    parser.add_argument('--checkpoint', default='checkpoint-vitb-2.pth', help='Path to local checkpoint file')
    parser.add_argument('--use_huggingface', action='store_true', help='Use Hugging Face model instead of local checkpoint')
    parser.add_argument('--huggingface_dir', default=None, help='Path to local Hugging Face model directory')
    parser.add_argument('--video', default='sample_0002.mp4', help='Path to input video file')
    parser.add_argument('--save_model', action='store_true', help='Save the Hugging Face model as a local .pth file')
    parser.add_argument('--output_pth', default='videomae_huggingface.pth', help='Path to save the model as .pth file')
    parser.add_argument('--compare_models', action='store_true', help='Compare results between Hugging Face and local checkpoint')
    
    args = parser.parse_args()
    return args

# Load the Kinetics400 class labels
def load_kinetics_classes():
    df = pd.read_csv("../kinetics400.csv")
    class_names = {int(row['id']): row['name'] for _, row in df.iterrows()}
    return class_names

def save_model_as_pth(model, processor, output_path):
    """
    Save a Hugging Face model as a PyTorch .pth file
    """
    print(f"Saving model to {output_path}...")
    
    # Extract the state dict
    state_dict = model.state_dict()
    
    # Save the model configuration
    config_dict = model.config.to_dict()
    
    # Save the processor configuration
    processor_config = processor.to_dict()
    
    # Create a dictionary with the model weights and configurations
    checkpoint = {
        'model': state_dict,
        'config': config_dict,
        'processor_config': processor_config,
        'model_name': 'videomae-base-finetuned-kinetics',
        'num_classes': 400,
    }
    
    # Save the model
    torch.save(checkpoint, output_path)
    
    # Also save the configurations as separate JSON files for easier inspection
    config_path = output_path.replace('.pth', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    processor_path = output_path.replace('.pth', '_processor_config.json')
    with open(processor_path, 'w') as f:
        json.dump(processor_config, f, indent=2)
    
    print(f"Model saved successfully to {output_path}")
    print(f"Config saved to {config_path}")
    print(f"Processor config saved to {processor_path}")

def load_model_from_pth(checkpoint_path):
    """
    Load a model from a .pth file saved by save_model_as_pth
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if this is a Hugging Face model saved with our function
    if 'config' in checkpoint and 'processor_config' in checkpoint:
        # Create the model with the saved configuration
        config = VideoMAEConfig.from_dict(checkpoint['config'])
        model = VideoMAEForVideoClassification(config)
        
        # Load the weights
        model.load_state_dict(checkpoint['model'])
        
        # Create the processor
        processor = VideoMAEImageProcessor.from_dict(checkpoint['processor_config'])
        
        print("Successfully loaded Hugging Face model from .pth file")
    else:
        # This is a different format, try to load it with the old method
        print("This doesn't appear to be a Hugging Face model saved with our function")
        print("Trying to load with the old method...")
        
        # Create the model with the default configuration
        config = VideoMAEConfig.from_pretrained(
            "MCG-NJU/videomae-base",
            num_labels=400,  # Kinetics-400 classes
        )
        model = VideoMAEForVideoClassification(config)
        
        # Extract model weights
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        elif 'module' in checkpoint:
            checkpoint_model = checkpoint['module']
        else:
            checkpoint_model = checkpoint
        
        # Print some keys to help with debugging
        print(f"Checkpoint keys (first 10): {list(checkpoint_model.keys())[:10]}")
        
        # Create a mapping from checkpoint keys to model keys
        new_dict = OrderedDict()
        for key, value in checkpoint_model.items():
            # Skip keys that don't match expected dimensions
            if 'head' in key and value.dim() == 2 and value.size(0) == 400:
                if key == 'head.weight':
                    new_dict['classifier.weight'] = value
                elif key == 'head.bias':
                    new_dict['classifier.bias'] = value
            elif 'patch_embed' in key:
                # Map patch embedding weights
                new_key = key.replace('patch_embed', 'videomae.embeddings')
                new_dict[new_key] = value
            elif 'blocks' in key:
                # Map transformer blocks
                new_key = key.replace('blocks', 'videomae.encoder.layer')
                new_dict[new_key] = value
            elif 'norm' in key:
                # Map layer normalization
                if key == 'norm.weight':
                    new_dict['videomae.layernorm.weight'] = value
                elif key == 'norm.bias':
                    new_dict['videomae.layernorm.bias'] = value
            else:
                # Keep other keys as is
                new_dict[key] = value
        
        # Load the weights with flexible matching
        missing, unexpected = model.load_state_dict(new_dict, strict=False)
        print(f"Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys")
        
        # Create a simple processor
        processor = VideoMAEImageProcessor()
    
    # Set model to evaluation mode
    model.eval()
    return model, processor

def load_model(args):
    """
    Load the VideoMAE model from a local checkpoint or Hugging Face
    """
    # Hugging Face model URL for reference
    huggingface_model_url = "https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics"
    print(f"Hugging Face model URL: {huggingface_model_url}")
    print("You can download this model locally using:")
    print("  git lfs install")
    print(f"  git clone {huggingface_model_url}")
    
    if args.use_huggingface or args.save_model or args.compare_models:
        if args.huggingface_dir and os.path.exists(args.huggingface_dir):
            print(f"Loading VideoMAE model from local Hugging Face directory: {args.huggingface_dir}")
            model = VideoMAEForVideoClassification.from_pretrained(args.huggingface_dir)
            processor = VideoMAEImageProcessor.from_pretrained(args.huggingface_dir)
        else:
            print("Loading VideoMAE model from Hugging Face online repository...")
            model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        
        # Save the model as a .pth file if requested
        if args.save_model:
            save_model_as_pth(model, processor, args.output_pth)
        
        if args.compare_models and os.path.exists(args.checkpoint):
            # Also load the local checkpoint for comparison
            local_model, local_processor = load_model_from_pth(args.checkpoint)
            return (model, processor), (local_model, local_processor)
            
        return model, processor
    else:
        # Load from local checkpoint
        if os.path.exists(args.checkpoint):
            return load_model_from_pth(args.checkpoint)
        else:
            print(f"Checkpoint not found at {args.checkpoint}, falling back to Hugging Face model")
            model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            return model, processor

def load_video(video_path, num_frames=16, sample_rate=4):
    """
    Load video frames with consistent sampling strategy
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video has {total_frames} frames at {fps} FPS")
    
    # Calculate center frame and create sampling indices
    if total_frames >= num_frames * sample_rate:
        # If we have enough frames, sample from the center
        center_frame = total_frames // 2
        start_idx = max(0, center_frame - (num_frames * sample_rate) // 2)
        indices = np.arange(start_idx, start_idx + num_frames * sample_rate, sample_rate)
        indices = np.clip(indices, 0, total_frames - 1)
    else:
        # If we don't have enough frames, sample uniformly and possibly repeat
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    print(f"Sampling frames at indices: {indices}")
    
    # Extract frames at the calculated indices
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame at index {idx}")
            # If we can't read a frame, duplicate the last one or use a blank frame
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            continue
            
        # Resize and convert BGR to RGB
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return frames

def run_inference(model, processor, frames, class_names):
    """
    Run inference on a set of frames
    """
    # Process frames with the VideoMAE processor
    inputs = processor(frames, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    
    # Get top 5 predictions
    top_probs, top_indices = torch.topk(probs, k=5)
    
    # Convert to Python types
    results = []
    for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
        results.append((class_names[idx], prob))
    
    return results

def main():
    # Parse arguments
    args = get_args()
    
    # Override video path if provided as positional argument
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        args.video = sys.argv[1]
        # Remove the positional argument to avoid parser errors
        sys.argv.remove(args.video)
    
    if not os.path.exists(args.video) and not args.save_model:
        print(f"Video file not found: {args.video}")
        return
    
    start_time = time.time()
    
    # Load the model and processor
    if args.compare_models:
        (hf_model, hf_processor), (local_model, local_processor) = load_model(args)
    else:
        model, processor = load_model(args)
    
    after_model_load = time.time()
    print(f"Model loaded in {after_model_load - start_time:.2f} seconds")
    
    # If we're just saving the model, we can exit now
    if args.save_model and not os.path.exists(args.video):
        print("Model saved. Exiting without running inference.")
        return
    
    # Load and preprocess the video
    frames = load_video(args.video, num_frames=args.num_frames, sample_rate=args.sampling_rate)
    
    # Load class names
    class_names = load_kinetics_classes()
    
    if args.compare_models:
        # Run inference with both models
        print("\nRunning inference with Hugging Face model...")
        hf_results = run_inference(hf_model, hf_processor, frames, class_names)
        
        print("\nRunning inference with local checkpoint...")
        local_results = run_inference(local_model, local_processor, frames, class_names)
        
        # Print results side by side
        print("\nComparison of top 5 predictions:")
        print("Hugging Face Model vs Local Checkpoint")
        print("-" * 60)
        for i, ((hf_class, hf_prob), (local_class, local_prob)) in enumerate(zip(hf_results, local_results)):
            print(f"{i+1}. {hf_class}: {hf_prob:.4f} | {local_class}: {local_prob:.4f}")
    else:
        # Run inference with the selected model
        results = run_inference(model, processor, frames, class_names)
        
        # Print results
        print("\nTop 5 predictions:")
        for i, (class_name, prob) in enumerate(results):
            print(f"{i+1}. {class_name}: {prob:.4f}")
    
    end_time = time.time()
    print(f"\nInference time: {end_time - after_model_load:.2f} seconds")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 