## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Model Architecture](#model-architecture)
- [Class Imbalance Handling](#class-imbalance-handling)


## Overview
The script processes multispectral data data to generate edge masks and other necessary preprocessing steps. Additionally, it includes a TensorFlow-based implementation of a U-Net model for semantic segmentation.

## Prerequisites
- Python 3.6+
- Google Colab (if running on Google Drive)
- Required Python libraries: `numpy`, `xarray`, `scikit-learn`, `tensorflow`

## Model Architecture
The U-Net model architecture used in this script consists of the following:

1. Contracting Path: Series of convolutional and max-pooling layers.
2. Bottleneck: Convolutional layers with dropout for regularization.
3. Expansive Path: Series of transposed convolutional layers and concatenations with the corresponding layers from the contracting path.
   
The model is designed to capture spatial-spectral characteristics from Sentinel-2 imagery for semantic segmentation.

##  Class Imbalance Handling
The script computes class weights based on the class distribution in the training data to handle class imbalance. These weights are incorporated into the custom loss function used for training the model.
