# Safety-Helmet-Detection-with-Deep-Learning
Automated detection of safety helmet compliance using transfer learning and data augmentation.

Overview

This project provides an automated solution for detecting safety helmet compliance among workers in hazardous environments using deep learning. The system uses transfer learning with VGG16, a fully connected neural network, and data augmentation to classify images as "With Helmet" or "Without Helmet". The goal is to support real-time monitoring and improve workplace safety by ensuring regulatory compliance.

Business Context

Industry Need: Manual helmet compliance checks are prone to human error and inefficiency, especially on large sites.

Objective: 
Deploy an image classification model that reliably distinguishes between workers with and without helmets, enabling automated, scalable safety enforcement.

Dataset
Total Images: 631

With Helmet: 311

Without Helmet: 320

Characteristics:

Diverse environments (construction, industrial, factories)

Variations in lighting, camera angles, and worker activities

Format:

Images as NumPy arrays (images_proj.npy)

Labels in CSV (Labels_proj.csv)

Data Preprocessing
Image Formats: Both grayscale and RGB processed.

Normalization: Pixel values scaled to .

Splitting: Stratified split into training, validation, and test sets.

Model Development
1. Baseline CNN
Simple Conv2D and Dense layers.

Poor recall and high validation loss.

2. VGG16 Transfer Learning
Pre-trained VGG16 (ImageNet) with frozen layers.

Added flatten and dense layers for binary classification.

High accuracy but risk of overfitting.

3. VGG16 + FFNN
Additional dense layers with batch normalization and dropout.

Slight improvement in generalization.

4. VGG16 + FFNN + Data Augmentation (Final Model)
Data augmentation: flipping, shifting, rotation, shear, zoom.

High accuracy, recall, and precision across all splits.

Improved robustness and generalization.
