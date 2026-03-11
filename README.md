# Deep Learning: Training CNNs

This repository contains various projects focused on training Convolutional Neural Networks (CNNs) using different datasets and frameworks like Keras and PyTorch.

## 📁 Repository Structure

The repository is organized into several projects:

- **[dog-cat_Classifier/](dog-cat_Classifier/)**: A project to classify images of cats and dogs.
  - Includes experiments with Data Augmentation and Batch Normalization.
  - Uses the `ImageDataGenerator` for real-time augmentation and streaming.
  - Contains a `PROJECT_REVIEW.md` with lessons learned.
- **[Training_CNN_MNIST_KERAS/](Training_CNN_MNIST_KERAS/)**: Handwritten digit classification using the MNIST dataset and Keras.
- **[Training_CNN_MNIST_PyTorch/](Training_CNN_MNIST_PyTorch/)**: MNIST digit classification implemented in PyTorch.
- **[Training_CNN_CIFAR-10_KERAS/](Training_CNN_CIFAR-10_KERAS/)**: Object classification using the CIFAR-10 dataset (RGB images) and Keras.

## 🚀 Featured Project: Dog vs. Cat Classifier

Located in `dog-cat_Classifier/`, this project focuses on building a robust binary classifier on a custom dataset of ~1,000 images.

### Key Milestones:
- **Accuracy Improvement**: Achieved **81.10%** validation accuracy through iterative improvements.
- **Overfitting Solutions**: Successfully addressed overfitting by implementing:
  - **Data Augmentation**: (Rotation, Zoom, Flip, Shear).
  - **Batch Normalization**: Stabilized training across convolutional layers.
- **Model Checkpointing**: Used `ModelCheckpoint` to ensure the best weights are saved even if later epochs overfit.

## 🛠️ Requirements & Tools

- **Frameworks**: TensorFlow/Keras, PyTorch
- **Libraries**: NumPy, Matplotlib, Scipy
- **Environment**: Conda (`aiENV`), Windows with GPU acceleration (RTX 3050 detected).

## 📓 Usage

Each folder contains a `notebook/` directory with a Jupyter Notebook. 
Open the notebooks to view the training pipeline, from data preparation and manual train/test splitting to model evaluation and inference.

---
*As a part of the Deep Learning learning journey.*
