import numpy as np
import onnxruntime as ort
import os

def run_inference(onnx_model_path="model/cnn_model_pytorch.onnx"):
    # Convert path to be relative to the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_model_path = os.path.join(script_dir, onnx_model_path)
    
    # Load the ONNX model and start a session
    session = ort.InferenceSession(onnx_model_path)
    
    # Get the input layer name and expected shape
    input_details = session.get_inputs()[0]
    input_name = input_details.name
    input_shape = input_details.shape
    
    print(f"Model Input Name: {input_name}")
    print(f"Model Expected Shape: {input_shape}")
    
    # Handle dynamic batch size (often represented as None or a string like 'batch')
    shape = list(input_shape)
    if not isinstance(shape[0], int):
        shape[0] = 1  # batch size of 1
    
    # Generate dummy data for inference testing (e.g., a batch of 1 image of 28x28 grayscale)
    # usually MNIST expects shape [batch_size, 1, 28, 28] with float32 type
    dummy_input = np.random.randn(*shape).astype(np.float32)
    
    # Perform the inference
    outputs = session.run(None, {input_name: dummy_input})
    
    # The output is a list containing the result tensors
    logits = outputs[0]
    predicted_class = np.argmax(logits, axis=1)
    
    print(f"Logits: {logits}")
    print(f"Predicted class index: {predicted_class[0]}")

if __name__ == "__main__":
    run_inference()
