# Binary Classification Project: Dogs vs. Cats

## Overview
This project demonstrates a simple image classification model that uses Convolutional Neural Networks (CNN) to classify images as either **cats** or **dogs**.

## Files in the Repository
- **`CATS_VS_DOGS.ipynb`**: Jupyter notebook containing the code for the CNN model, data loading, training, and evaluation.
- **`dataset/`**: Contains subfolders `cats/` and `dogs/` with respective images for training the model.
- **`README.md`**: This file with project details and instructions.

## Installation and Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourUsername/Binary-Classification-Project.git
   ```
2. **Install Dependencies**:
   The project requires the following libraries:
   - TensorFlow
   - Keras
   - Matplotlib
   - OpenCV
   - numpy

## Usage
1. **Preparing the Dataset**:
   - Make sure your dataset is organized into two subfolders: `cats/` and `dogs/` containing respective images.

2. **Run the Google Colab**:
   - Open `CATS_VS_DOGS.ipynb` in a Google Colab environment (or Jupyter Notebook).

## Model Architecture
- The model is a Convolutional Neural Network (CNN) with layers like Conv2D, MaxPooling2D, Dense, etc.

## Results
The model trains for 10 epochs and plots the training and validation accuracy/loss curves. It achieves good performance in distinguishing between images of cats and dogs.

## Acknowledgements
- Dataset: The `Dogs vs. Cats` dataset is from Kaggle.
- Libraries: TensorFlow, Keras, Matplotlib, OpenCV, and NumPy.

