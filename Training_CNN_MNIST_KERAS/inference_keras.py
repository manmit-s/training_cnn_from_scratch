import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt

def build_model():
    """Rebuild the model architecture as defined in the training notebook."""
    model = Sequential()
    # CONV_1: 32 filters, 3x3 kernel, same padding, ReLU, 28x28x1 input
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    # POOL_1: 2x2 max pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # CONV_2: 64 filters, 3x3 kernel, same padding, ReLU
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # POOL_2: 2x2 max pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten for dense layers
    model.add(Flatten())
    # FC_1: 64 units, ReLU
    model.add(Dense(64, activation='relu'))
    # FC_2: 10 units, Softmax (output classes 0-9)
    model.add(Dense(10, activation='softmax'))
    
    return model

def run_inference(weights_path, image_index=0):
    """
    Load weights, get an image from MNIST dataset, and predict its class.
    """
    # 1. Load MNIST data to get a sample image (simulating user input)
    print("Loading MNIST dataset...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 2. Pick a sample image and preprocess it
    # The model expects (batch, height, width, channels) and normalized [0, 1]
    sample_img = x_test[image_index]
    processed_img = sample_img.astype('float32') / 255.0
    processed_img = np.expand_dims(processed_img, axis=(0, -1)) # Add batch and channel dims
    
    # 3. Build model and load weights
    print(f"Loading weights from {weights_path}...")
    model = build_model()
    model.load_weights(weights_path)
    
    # 4. Predict
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # 5. Show result
    print("-" * 30)
    print(f"Sample Index: {image_index}")
    print(f"Actual Label: {y_test[image_index]}")
    print(f"Predicted Label: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("-" * 30)
    
    return sample_img, y_test[image_index], predicted_class

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to the saved weights relative to the script location
    weights = os.path.join(script_dir, "model", "model.weights.best.keras")
    
    if not os.path.exists(weights):
        print(f"Error: Weights file not found at {weights}")
    else:
        # Run inference on a random image from the test set
        idx = np.random.randint(0, 10000)
        img, actual, pred = run_inference(weights, image_index=idx)
    
    # Optionally save the image to check visually if needed (though not asked)
    # plt.imshow(img, cmap='gray')
    # plt.title(f"Actual: {actual}, Pred: {pred}")
    # plt.show()
