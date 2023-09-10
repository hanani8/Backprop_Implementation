# Backpropagation Implementation with Sigmoid Activation, Softmax Output, and Cross-Entropy Loss

This is an implementation of a backpropagation neural network with the following characteristics:
- Activation Function: Sigmoid for hidden layers
- Output Activation Function: Softmax
- Loss Function: Cross-Entropy

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Implementation Details](#implementation-details)
- [License](#license)

## Introduction

This implementation demonstrates a simple feedforward neural network using backpropagation for training. It consists of two hidden layers with a sigmoid activation function and an output layer with softmax activation. The network is trained using cross-entropy loss.

The implementation is designed to work with any dataset by simply modifying the input and output data matrices.

## Usage

To use this implementation, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have the required dependencies installed (see [Dependencies](#dependencies)).
3. Modify the `X` (input) and `Y` (output) matrices to suit your dataset.
4. Adjust the number of layers, inputs, outputs, and neurons in each layer in the `no_of_layers`, `no_of_inputs`, `no_of_outputs`, and `no_of_neurons_in_each_layer` variables.
5. Run the code to train the neural network using backpropagation.
6. You can adjust the number of training epochs by changing the `max_epochs` variable.

For a complete example, check the provided [Jupyter Notebook](BACKPROP.ipynb).

## Dependencies

This implementation relies on the following libraries:

- NumPy: For numerical operations.

You can install NumPy using pip:

```bash
pip install numpy
```

## Implementation Details

- `sigmoid(x)`: The sigmoid activation function used in the hidden layers.
- `softmax(x)`: The softmax activation function used in the output layer.
- `grad_sigmoid(x)`: The gradient of the sigmoid function.
- `linear(w, x, b)`: Computes the linear combination of weights, inputs, and biases.
- `error(h)`: Computes the cross-entropy loss.

The neural network architecture and parameters are defined in the `no_of_layers`, `no_of_inputs`, `no_of_outputs`, and `no_of_neurons_in_each_layer` variables.

- `forward_propagation(input)`: Performs forward propagation through the neural network.
- `backward_propagation(a, h)`: Performs backward propagation to compute gradients.
- Training is performed using gradient descent.

---
