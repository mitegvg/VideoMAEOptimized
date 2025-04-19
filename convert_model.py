import torch
from transformers import VideoMAEForVideoClassification
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import os

class VideoMAEWrapper(tf.keras.Model):
    def __init__(self):
        super(VideoMAEWrapper, self).__init__()
        # Define the layers that match VideoMAE's architecture
        self.patch_embed = tf.keras.layers.Conv3D(
            filters=768,
            kernel_size=(2, 16, 16),
            strides=(2, 16, 16),
            padding='valid',
            name='patch_embedding'
        )
        
        self.pos_embed = tf.keras.layers.Dense(768, name='position_embedding')
        
        # 12 transformer blocks
        self.transformer_blocks = []
        for i in range(12):
            block = tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.MultiHeadAttention(
                    num_heads=12,
                    key_dim=64,
                    dropout=0.0
                ),
                tf.keras.layers.Dense(3072, activation='gelu'),
                tf.keras.layers.Dense(768),
                tf.keras.layers.LayerNormalization()
            ], name=f'transformer_block_{i}')
            self.transformer_blocks.append(block)
        
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.classifier = tf.keras.layers.Dense(400, activation='softmax')

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 16, 3, 224, 224], dtype=tf.float32)])
    def call(self, inputs):
        # Normalize inputs
        x = (inputs - 0.5) / 0.5
        
        # Rearrange dimensions for TensorFlow
        x = tf.transpose(x, perm=[0, 2, 3, 4, 1])  # [B, C, H, W, F] -> [B, C, H, W, F]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Flatten patches
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, 768])
        
        # Add position embeddings
        positions = tf.range(tf.shape(x)[1], dtype=tf.float32)
        pos_embed = self.pos_embed(positions[None, :, None])
        x = x + pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Layer norm
        x = self.layer_norm(x)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Classification
        return self.classifier(x)

def convert_videomae():
    print("Creating TensorFlow model...")
    tf_model = VideoMAEWrapper()
    
    # Load PyTorch model
    print("Loading PyTorch model...")
    torch_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    torch_model.eval()
    
    print("Transferring weights...")
    # Create a dummy input to get the model to build its layers
    dummy_input = tf.random.normal((1, 16, 3, 224, 224))
    _ = tf_model(dummy_input)
    
    # Transfer weights from PyTorch to TensorFlow
    with torch.no_grad():
        # Patch embedding weights
        tf_model.patch_embed.kernel.assign(
            torch_model.videomae.embeddings.patch_embeddings.projection.weight.permute(2, 3, 4, 1, 0).numpy()
        )
        
        # Position embedding weights
        tf_model.pos_embed.kernel.assign(
            torch_model.videomae.embeddings.position_embeddings.weight.numpy().T
        )
        
        # Transformer blocks
        for i, (tf_block, torch_block) in enumerate(zip(tf_model.transformer_blocks, torch_model.videomae.encoder.layer)):
            # Attention weights
            tf_block.layers[1].key_dense.kernel.assign(torch_block.attention.attention.key.weight.numpy().T)
            tf_block.layers[1].query_dense.kernel.assign(torch_block.attention.attention.query.weight.numpy().T)
            tf_block.layers[1].value_dense.kernel.assign(torch_block.attention.attention.value.weight.numpy().T)
            tf_block.layers[1].combine_heads.kernel.assign(torch_block.attention.output.dense.weight.numpy().T)
            
            # MLP weights
            tf_block.layers[2].kernel.assign(torch_block.intermediate.dense.weight.numpy().T)
            tf_block.layers[3].kernel.assign(torch_block.output.dense.weight.numpy().T)
            
            # Layer norm weights
            tf_block.layers[0].gamma.assign(torch_block.layernorm_before.weight.numpy())
            tf_block.layers[0].beta.assign(torch_block.layernorm_before.bias.numpy())
            tf_block.layers[4].gamma.assign(torch_block.layernorm_after.weight.numpy())
            tf_block.layers[4].beta.assign(torch_block.layernorm_after.bias.numpy())
        
        # Final layer norm
        tf_model.layer_norm.gamma.assign(torch_model.videomae.layernorm.weight.numpy())
        tf_model.layer_norm.beta.assign(torch_model.videomae.layernorm.bias.numpy())
        
        # Classifier
        tf_model.classifier.kernel.assign(torch_model.classifier.weight.numpy().T)
        tf_model.classifier.bias.assign(torch_model.classifier.bias.numpy())
    
    print("Saving model...")
    os.makedirs("js/model", exist_ok=True)
    
    # Save as SavedModel
    tf.saved_model.save(
        tf_model,
        "js/model/saved_model",
        signatures=tf_model.call
    )
    
    # Convert to TensorFlow.js
    print("Converting to TensorFlow.js format...")
    tfjs.converters.convert_tf_saved_model(
        "js/model/saved_model",
        "js/model/tfjs"
    )
    
    print("Conversion complete. Model saved in TensorFlow.js format at js/model/tfjs")

if __name__ == "__main__":
    convert_videomae()