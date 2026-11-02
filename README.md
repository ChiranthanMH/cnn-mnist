#CNN on MNIST: Handwritten Digit Recognition

A convolutional neural network built with PyTorch to recognize handwritten digits (0 to 9) from the MNIST dataset.

#Results

- Test accuracy: **98.85%**
- Plain fully connected network on the same data: 97.67%

#Architecture

Conv (1 to 32) → MaxPool → Conv (32 to 64) → MaxPool → Fully connected (128) → Output (10)

#How to Run

pip install torch torchvision matplotlib
python cnn.py

#What I Learned

Convolutional filters pick up local patterns like edges and curves wherever they appear in the image, which is why the CNN beat the fully connected network.
