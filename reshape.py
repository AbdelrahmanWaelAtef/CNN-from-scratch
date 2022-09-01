import re
from turtle import forward
import numpy as np
from layer import Layer

class Reshape(Layer):
    def __init__(self, inputShape, outputShape):
        self.inputShape = inputShape
        self.outputShape = outputShape
    
    def forward(self, input):
        return np.reshape(input, self.outputShape)
    
    def backward(self, outputGradient):
        return np.reshape(outputGradient, self.inputShape)