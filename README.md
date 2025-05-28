# CIFAR-10 Image Classification from Scratch (NumPy Only)

## Analysis of Model Performance

The implemented neural network, trained from scratch using only NumPy, successfully classifies images from three selected CIFAR-10 classes. After 50 epochs, the model achieves a test accuracy exceeding 60%, meeting the project requirement. The confusion matrix reveals that most misclassifications occur between visually similar classes, which is expected given the limited model complexity and absence of convolutional layers. Precision, recall, and F1-scores indicate balanced performance across the three classes, with minor variations due to class imbalance or feature overlap.

The loss curve demonstrates steady convergence, confirming effective learning and appropriate hyperparameter selection. Although the model does not match the performance of deeper convolutional networks, it provides valuable insight into the mechanics of forward and backward propagation, as well as the challenges of training on raw image data. Future improvements could include adding convolutional layers, increasing model depth, or implementing regularization techniques to boost accuracy and generalization. Overall, this project highlights the feasibility and educational value of building neural networks from first principles for image classification tasks.


## Overview

This project implements a simple neural network **from scratch** (using only NumPy) to classify images from **three classes** of the CIFAR-10 dataset.  
**No pre-trained models, TensorFlow, or PyTorch are used.**  
All forward propagation, backpropagation, and gradient descent steps are implemented manually.

---

## Features

-  Manual neural network implementation (NumPy only)  
-  3-class CIFAR-10 classification (e.g., airplane, automobile, bird)  
-  Custom training loop with loss curve plotting  
-  Evaluation metrics: accuracy, precision, recall, F1-score, and confusion matrix  
-  Platform-independent: runs on any CPU (no GPU required)

---

## Project Structure

cifar10_numpy_classification/
│
├── data/
│ └── cifar-10-batches-py/ # Downloaded CIFAR-10 data here
├── utils.py # Data loading and preprocessing
├── model.py # Neural network implementation
├── train.py # Training script
├── evaluate.py # Evaluation metrics and confusion matrix
├── main.py # Entrypoint (runs training & evaluation)
└── README.md 


---

## Setup Instructions

### 1. Install Requirements

```bash
pip install numpy matplotlib scikit-learn
```

### 2. Download CIFAR-10 Dataset

   Download the CIFAR-10 Python version

   Extract it and place the cifar-10-batches-py folder inside a data/ directory in your project root:

cifar10_numpy_classification/data/cifar-10-batches-py/

### 3. Run Training and Evaluation

```
python main.py
```
   This will train the model and then evaluate it on the test set.

   You will see printed metrics and generated plots:

   

 - loss_curve.png (training loss curve)
   
     
 - confusion_matrix.png (confusion matrix)
      
    
 - model_weights.npz (saved model weights)

Model Architecture

    Input Layer: 3072 units (32x32x3 RGB image, flattened)

    Hidden Layer 1: 128 units, ReLU activation

    Hidden Layer 2: 64 units, ReLU activation

    Output Layer: 3 units, Softmax activation (one per class)

Evaluation Metrics

    Accuracy: Should exceed 60% on the test set

    Precision, Recall, F1-score: Printed per class

    Confusion Matrix: Plotted and saved as an image

Example Results

Epoch 50/50 Loss: 0.85
Accuracy: 0.82
Precision: [[0.82333333 0.83317887 0.80313418]
Recall: [0.741 0.899 0.82]
F1-score: [0.78       0.86483886 0.81147947]

See confusion_matrix.png and loss_curve.png for more details.
## Notes

     No GPU required: All computations run on CPU

     No deep learning frameworks: Only NumPy, matplotlib, and scikit-learn are used

     Training set size: 15,000 images (3 classes × 5,000 images each)

     Test set size: 3,000 images (3 classes × 1,000 images each)

# Analysis

This project demonstrates how to build and train a neural network from scratch for image classification. Despite its simplicity and lack of convolutional layers, the model can achieve over 60% accuracy on three CIFAR-10 classes. Most misclassifications occur between visually similar classes, which is expected given the architecture.

This project is ideal for educational purposes and understanding the fundamentals of neural networks without relying on black-box frameworks.
