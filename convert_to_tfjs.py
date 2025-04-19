from optimum.exporters import TasksManager
from transformers import VideoMAEForVideoClassification
from optimum.exporters.tensorflow import convert_pytorch_to_tensorflow
import tensorflow as tf
import tensorflowjs as tfjs
import os

def convert_videomae_to_tfjs():
    print("Loading VideoMAE model...")
    model_id = "MCG-NJU/videomae-base-finetuned-kinetics"
    pytorch_model = VideoMAEForVideoClassification.from_pretrained(model_id)
    
    print("Converting to TensorFlow...")
    tf_model = convert_pytorch_to_tensorflow(pytorch_model, model_id)
    
    # Save the model in TensorFlow SavedModel format first
    print("Saving as SavedModel...")
    os.makedirs("js/model", exist_ok=True)
    tf.saved_model.save(tf_model, "js/model/saved_model")
    
    # Convert SavedModel to TensorFlow.js format
    print("Converting to TensorFlow.js format...")
    tfjs.converters.convert_tf_saved_model(
        "js/model/saved_model",
        "js/model/tfjs"
    )
    
    print("Model converted and saved in TensorFlow.js format at js/model/tfjs")

if __name__ == "__main__":
    convert_videomae_to_tfjs()