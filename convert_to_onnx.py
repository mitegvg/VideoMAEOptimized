import torch
from transformers import VideoMAEForVideoClassification
import os

def convert_videomae_to_onnx():
    # Load the model
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model.eval()

    # Create dummy input tensor
    # VideoMAE expects input shape: (batch_size, num_frames, num_channels, height, width)
    dummy_input = torch.randn(1, 16, 3, 224, 224)

    # Export the model
    torch.onnx.export(
        model,                     # model being run
        dummy_input,              # model input
        "js/videomae.onnx",      # where to save the model
        export_params=True,       # store the trained parameter weights inside the model file
        opset_version=14,         # using ONNX opset 14 for scaled_dot_product_attention support
        do_constant_folding=True, # whether to execute constant folding for optimization
        input_names=['input'],    # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("Model converted to ONNX format and saved as js/videomae.onnx")

if __name__ == "__main__":
    os.makedirs("js", exist_ok=True)
    convert_videomae_to_onnx()