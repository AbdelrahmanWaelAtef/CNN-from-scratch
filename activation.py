from layer import Layer
import numpy as np

# Create activation layer
class Activation(Layer):
    
    # Constructor takes the activation function and its derivative
    def __init__(self, activation, activationPrime):
        self.activation = activation
        self.activationPrime = activationPrime
    
    # Forward function only applies the activation to the input
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
     
    # Backward function uses the activationPrime and simply perform element wise multiplication with the outputGradient
    def backward(self, outputGradient, learningRate):
        return np.multiply(outputGradient, self.activationPrime(self.input))