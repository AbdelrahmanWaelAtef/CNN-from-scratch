from layer import Layer
import numpy as np

# Create dense layer
class Dense(Layer):
    def __init__(self, inputSize, outputSize):
        
        # Random initial weights and bias
        self.weights = np.random.randn(outputSize, inputSize)
        self.bias = np.random.randn(outputSize, 1)
    
    # Forward propagation using simple matrix multiplication and addition
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    # Backward propagation using simple matrix multiplication
    def backward(self, outputGradient, learningRate):
        
        # Calculate the derivative of the error with respect to weights
        weightsGradient = np.dot(outputGradient, self.input.T)
        
        # The derivative of the error with respect to bias is the outputGradient
        
        # Update the parameters with gradient descent
        self.weights -= learningRate * weightsGradient
        self.bias -= learningRate * outputGradient
        
        # Return the derivative of the error with respect to the input
        return np.dot(self.weights.T, outputGradient)