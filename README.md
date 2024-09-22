# Entri-Elevate-CNN-Model-Handwritten-Digit-Recognition

# Handwritten Digit Recognition using CNN

## Project Overview
The goal of this project is to build a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. The MNIST dataset contains 70,000 grayscale images of digits (0-9), each 28x28 pixels in size. Using TensorFlow and Keras, we will create and train a deep learning model to classify these digits accurately. The project involves data preprocessing, model building, training, and evaluation to understand CNNs and their applications.

## Dataset
- **MNIST Dataset**: Contains 70,000 grayscale images of handwritten digits (0-9).
- Each image is 28x28 pixels.

## Steps Involved
1. **Load the MNIST dataset** correctly.
2. **Normalize** the pixel values between 0 and 1.
3. **Build a CNN** with:
   - Convolutional layers
   - MaxPooling layers
   - Dense layers and the correct output layer with 10 neurons and softmax activation.
4. Use the **'adam' optimizer**.
5. Set the **loss function** to **'categorical_crossentropy'**.
6. Track **accuracy** as the metric.
7. **Train the model**.
8. Track both **training and validation accuracy/loss**.
9. **Evaluate the model** on test data.
10. Plot **accuracy** and **loss graphs**.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- PIL (Pillow)

## Example of Results
After running the script, the model will be trained, and you will see plots of training/validation accuracy and loss. The model's test accuracy will also be displayed.

## Conclusion
In this project, we successfully built a Convolutional Neural Network to classify handwritten digits using the MNIST dataset. Through careful data preprocessing, model architecture design, and training, we achieved a high level of accuracy. This project demonstrates the effectiveness of CNNs in image classification tasks and provides a solid foundation for further exploration in deep learning and computer vision.
