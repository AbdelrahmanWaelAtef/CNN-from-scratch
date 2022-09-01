from activation import Activation
import numpy as np

# Create hyperbolic tangent activation function
class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanhPrime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanhPrime)