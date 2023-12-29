# Digit-Classifier-Neural-Network
Main.py demonstrates a simple multilayer perceptron model that classifies grayscale digits. The neural network uses no special libraries like PyTorch or TensorFlow, only NumPy for matrix calculation. The architecture is rather simple and concise, using the Stochastic Gradient Descent and Backpropagation algorithms. 

Writing this neural network from scratch taught me the mathematical nature of machine learning algorithms. In particular, I wanted to highlight the calculus behind backpropagation. The main objective of creating a neural network is to find the optimal parameters (weights and biases) to minimize the cost function. When determining each weight and bias in the network, the partial derivative of the cost function with respect to these weights and biases is calculated. In other words, we observe how sensitive the cost function is to small changes in each weight and bias. These can all be mathematically derived, hence creating a beautifully structured code that takes only 80 lines of actual code. 

The delicate interplay between machine learning and mathematics is revealed through the process of writing neural networks from scratch. 
