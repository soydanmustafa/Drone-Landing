# Segmentation with Drone Landing Data

## Overview
This project aims to develop a robust and efficient drone landing system utilizing deep learning techniques for image segmentation and safe landing site identification. The project employs a U-Net architecture, a state-of-the-art convolutional neural network (CNN), to perform semantic segmentation on aerial images. The segmented output classifies various ground features such as trees, buildings, water bodies, and open ground, which are critical for determining safe landing zones.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
  - [Data Preparation and Augmentation](#data-preparation-and-augmentation)
  - [Model](#model)
  - [Training](#training)
  - [Landing Zones](#landing-zones)
- [Data](#data)
  - [CSV Data](#csv-data)
  - [Images and Masks](#images-and-masks)
  - [Augmentation](#augmentation)
  - [Data Division](#data-division)
  - [Drone Landing](#drone-landing)
- [Models](#models)
  - [U-Net](#u-net)
  - [U-Net with ResNet Backbone](#u-net-with-resnet-backbone)
- [Training](#training)
- [Experiment Results](#experiment-results)
  - [First Experiment](#first-experiment)
  - [Second Experiment](#second-experiment)
  - [Third Experiment](#third-experiment)
- [Discussions](#discussions)
- [References](#references)

## Introduction
This project focuses on leveraging deep learning models, specifically U-Net, for semantic segmentation of aerial images to identify suitable landing zones. The primary objective is to classify different ground features such as trees, buildings, water bodies, and open ground, and subsequently determine the safest landing spots for drones.

## Methodology

### Data Preparation and Augmentation
- Dataset: 208 aerial images with mask pairs, augmented to 416 images.
- Augmentation: Rotation, cropping, brightness, and contrast adjustments.
- Image resizing: From 6000x4000 to 256x256 for computational efficiency.
- Batch size: 64.

### Model
- U-Net model and an advanced U-Net model with a ResNet50 backbone.
- Encoder: ResNet50 layers with pre-trained weights.
- Decoder: Transpose convolution layers.

### Training
- Metrics: MeanIoU, Dice Coefficient.
- Learning Rate: Exponential Decay Learning Rate.
- Custom Callback: ImageMaskCallback for visualizing predictions after each epoch.
- Training and validation: Split into training and validation sets, trained for 15 epochs with a batch size of 16.

### Landing Zones
- Evaluation: Best model loaded to generate predictions.
- Classification: Segmented output classified into different terrain types using region properties.
- Safe zone calculation: Dangerous zones identified and safest landing zone determined as the farthest point from dangerous zones.

## Data

### CSV Data
- Classes: 24 classes including trees, buildings, water bodies, etc.
- Each class has specific RGB values for segmentation.

### Images and Masks
- Images: Aerial images from a drone’s perspective, resized to 256x256 pixels.
- Channels: RGB (3 channels).

### Augmentation
- Techniques: Rotating, randomly cropping, and adjusting contrast and brightness.
- Tools: Functions gathered from Tensorboard library.

### Data Division
- Split: 80% training and 20% validation.
- Techniques: Batch size consideration and repeat function applied for insufficient data during training.

### Drone Landing
- Classification: Dangerous and safe areas identified based on segmented output.
- Safe zone: Calculated as the farthest point from dangerous zones.

## Models

### U-Net
- Structure: Encoder-decoder with skip connections.
- Encoder: Convolutional and max-pooling layers.
- Decoder: Transpose convolution layers.

### U-Net with ResNet Backbone
- Structure: ResNet layers for encoding and U-Net for decoding.
- Weights: Pretrained on Imagenet and retrained.

## Training
- Epochs: 100 epochs for each model.
- Metrics: Intersection over Union, Dice Coefficient, and Accuracy.
- Learning Rate: Exponential decay with early stopping to prevent overfitting.
- Model selection: Best model saved based on validation loss.

## Experiment Results

### First Experiment
- Parameters: Batch size = 16, Epoch = 10, Model = U-Net.
- Results: Accuracy: 0.3476, Loss: 3.0945.
- Best landing point: [255,0].

### Second Experiment
- Parameters: Batch size = 8, Epoch = 100, Model = U-Net with ResNet Backbone.
- Results: Validation Loss: 1.20, Training Accuracy: 74%, Validation Accuracy: 33%, Dice Coefficient: 5.46, IoU: 14%, Test Accuracy: 67%.
- Best landing point: [156, 129].

### Third Experiment
- Parameters: Batch size = 8, Epoch = 100, Model = U-Net (more complex than the first experiment).
- Results: Validation Loss: 2.32, Training Accuracy: 36%, Validation Accuracy: 43%, Dice Coefficient: 2.93, IoU: 3%, Test Accuracy: 41%.
- Best landing point: [19, 255].

## Discussions
- Best Model: U-Net with ResNet backbone due to pretrained weights and complex structure.
- Challenges: Lack of data and computational power.
- Results: Successful in avoiding dangerous zones for drone landing despite low segmentation metrics.
- Variations: Different batch sizes, augmentation methods, models, and training varieties experimented.

## References
1. Ayushdabra. (n.d.). GitHub - ayushdabra/drone-images-semantic-segmentation: Multi-class semantic segmentation performed on "Semantic Drone Dataset." GitHub.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-NET: Convolutional Networks for Biomedical Image Segmentation. In Lecture notes in computer science.
3. Neven, R., & Goedemé, T. (2021). A Multi-Branch U-Net for Steel Surface Defect Type and Severity Segmentation. Metals.
