import numpy as np
import random

# Load the NIST dataset
data = np.loadtxt("NIST.txt")

# Split the data into inputs (X) and targets (y)
X = data[:, :-1]
y = data[:, -1]

# Initialize the weights and biases randomly
weights = np.random.rand(X.shape[1], 1)
bias = random.random()

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the loss function
def loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Define the forward pass
def forward(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias)

# Define the backward pass to update the weights and biases
def backward(X, y_pred, y_true, weights, bias, learning_rate):
    d_weights = 2 * np.dot(X.T, y_pred - y_true) / len(y_true)
    d_bias = 2 * np.mean(y_pred - y_true)
    weights -= learning_rate * d_weights
    bias -= learning_rate * d_bias
    return weights, bias

# Train the model for a set number of epochs
epochs = 100
learning_rate = 0.1
for epoch in range(epochs):
    y_pred = forward(X, weights, bias)
    current_loss = loss(y_pred, y)
    weights, bias = backward(X, y_pred, y, weights, bias, learning_rate)
    print("Epoch %d: Loss %f" % (epoch, current_loss))

# Use the trained model to make predictions on new data
new_data = np.array([[0.5, 0.6, 0.7, 0.8]])
prediction = forward(new_data, weights, bias)
print("Prediction:", prediction)
