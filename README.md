# PRODIGY_ML_04: Hand Gesture Recognition using CNN

## Project Overview

This task focused on developing a hand gesture recognition system using deep learning. The model is designed to classify static hand gestures captured in images, enabling natural and intuitive human-computer interaction.

The solution leverages a Convolutional Neural Network (CNN) trained on the LeapGestRecog dataset to accurately detect and classify multiple hand gesture categories.

## Highlights

- **Dataset**: LeapGestRecog dataset from Kaggle  
- **Architecture**: Multi-layered CNN with feature extraction and classification blocks  
- **Framework**: PyTorch  
- **Accuracy**: Achieved 99.02% test accuracy

## Key Features

- Image preprocessing with PyTorch `transforms`
- Dataset loading with custom `DataLoader`
- CNN model definition, training, and evaluation
- Training and validation accuracy tracking
- Performance evaluation on test data

## Technologies Used

- Python 3
- PyTorch
- torchvision
- matplotlib
- numpy
- scikit-learn

## Dataset

The project uses the [LeapGestRecog](https://www.kaggle.com/datasets/kmader/leapgestrecog) dataset, which consists of:
- 10 classes of hand gestures
- Over 20,000 grayscale gesture images
- Images captured under consistent lighting conditions using a Leap Motion sensor

## Model Architecture

The CNN architecture includes:
- Multiple convolutional and pooling layers for hierarchical feature extraction
- Batch normalization and dropout for regularization
- Fully connected layers for final classification

## Results

| Metric       | Value     |
|--------------|-----------|
| Test Accuracy| 99.02%    |

The high accuracy indicates effective feature learning and generalization by the model.
The model demonstrates high generalization on unseen data and strong classification performance across gesture classes.


## Learning Outcomes

* Developed and fine-tuned a deep CNN architecture for image classification
* Learned effective preprocessing using PyTorch’s data pipeline tools
* Understood key evaluation metrics and techniques to assess deep learning model performance
