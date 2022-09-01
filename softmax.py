# Importing libraries
import enum
import numpy as np
from layer import Layer

class Softmax(Layer):
    def __init__(self, inputNode, softmaxNode):
        self.weight = np.random.randn(inputNode, softmaxNode)/inputNode
        self.bias = np.zeros(softmaxNode)
    
    def forward(self, image):
        self.originalImageShape = image.shape
        imageModified = image.flatten()
        self.modInput = imageModified
        outValue = np.dot(imageModified, self.weight) + self.bias
        self.out = outValue
        expOut = np.exp(outValue)
        return expOut/np.sum(expOut, axis=0)
    
    def backward(self, dl_dout, learningRate):
        for i, grad in enumerate(dl_dout):
            if grad == 0:
                continue
            transformationEquation = np.exp(self.out)
            totalSum = np.sum(transformationEquation)
            
            # Gradients with respect to output z
            dy_dz = - transformationEquation[i] * transformationEquation / (totalSum ** 2)
            dy_dz[i] = transformationEquation[i] * (totalSum - transformationEquation[i]) / (totalSum ** 2)
            
            # Gradients of totals against weight, biases, inputs
            dz_dw = self.modInput
            dz_db = 1
            dz_dinp = self.weight
            
            # Gradients of loss against totals
            dl_dz = grad * dy_dz
            
            # Gradients of loss against weights, biases, inputs
            dl_dw = dz_dw[np.newaxis].T @ dl_dz[np.newaxis]
            dl_db = dl_dz * dz_db
            dl_dinp = dz_dinp @ dl_dz
            
            self.weight -= learningRate * dl_dw
            self.bias -= learningRate * dl_db
            
            return dl_dinp.reshape(self.originalImageShape)