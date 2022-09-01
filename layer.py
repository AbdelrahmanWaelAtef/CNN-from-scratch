# Create layer class
class Layer:
    def __init__(self):
        # Any layer should have input and output attributes
        self.input = None
        self.output = None
        
    # Forward propagation
    def forward(self, input):
        pass
    
    # Backward propagation
    def backward(self, outputGradient, learningRate):
        pass        