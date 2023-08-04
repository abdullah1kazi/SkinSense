# SkinSense

## Overview
SkinSense is a machine learning tool designed to assist doctors in diagnosing skin diseases using image analysis. Our goal is to provide a reliable, easily-accessible tool that can help reduce the diagnosis time and improve patient outcomes.

## Model Details
Our model is built on the ResNet50 architecture and refined with advanced data augmentation techniques such as random rotations, color adjustments, sharpness adjustments, blurring, and image normalization. The model was trained through multiple cycles with a learning rate of 0.009, dynamically adjusted based on learning progress. 

In performance testing, the model achieved perfect accuracy in differentiating between harmful and non-harmful skin lesions and an accuracy of 0.8402 in diagnosing specific types of skin disease.

## Installation
To install the required libraries, use pip and the provided requirements.txt file:

```bash
pip install -r requirements.txt
