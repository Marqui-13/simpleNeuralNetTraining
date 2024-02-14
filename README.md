# Simple Neural Network Training Example

This repository contains a Python script for training a simple single-layer neural network using the NIST dataset. The network uses a sigmoid activation function, mean squared error loss function, and gradient descent for optimization. It demonstrates the fundamental concepts of forward and backward propagation in neural network training.

## Prerequisites

- Python 3.x
- NumPy

## Installation

1. Clone this repository to your local machine:


git clone https://github.com/Marqui-13/simple-neural-network.git


2. Navigate to the cloned directory:


cd simple-neural-network


3. Ensure you have Python installed. You can download Python from [here](https://www.python.org/downloads/).

4. Install NumPy if you haven't already:


pip install numpy


## Usage

To run the neural network training script, execute the following command in your terminal:


python neural_network_training.py


The script will load data from the `NIST.txt` file (ensure this file is in the same directory as the script or update the script with the correct path), split it into inputs and targets, initialize the model parameters, and start the training process for 100 epochs. The loss will be printed after each epoch to monitor the training progress. At the end of training, the script will output a prediction for a new data point.

## Data Format

The `NIST.txt` file should be formatted with input features in columns followed by the target variable in the last column. Each row represents a single data point.

## Contributing

Contributions to improve the script or enhance documentation are welcome. Please feel free to fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NumPy team for providing the fundamental package for scientific computing with Python.
- The creators of the NIST dataset for providing a benchmark dataset for machine learning and neural network training.