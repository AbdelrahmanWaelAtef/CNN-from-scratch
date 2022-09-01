# Importing libraries
import numpy as np
from layer import Layer

class Convolution(Layer):
    def __init__(self, numFilters, filterSize):
        self.numFilters = numFilters
        self.filterSize = filterSize
        
        # Normalise the coefficients
        self.convFilter = np.random.randn(numFilters, filterSize, filterSize)/(filterSize * filterSize)
    
    # Extracting patches from image (Generator function)
    def imageRegion(self, image):
        height, width = image.shape
        self.image = image
        for j in range(height - self.filterSize + 1):
            for k in range(width - self.filterSize + 1):
                imagePatch = image[j: (j + self.filterSize), k: (k + self.filterSize)]
                yield imagePatch, j, k
    
    def forward(self, image):
        height, width = image.shape
        convOut = np.zeros((height - self.filterSize + 1, width - self.filterSize + 1, self.numFilters))
        for imagePatch, i, j in self.imageRegion(image):
            convOut[i, j] = np.sum(imagePatch * self.convFilter, axis = (1, 2))
        return convOut
    
    def backward(self, dl_dout , learningRate):
        # dl_dout is coming from max-pool 
        dL_dF_parameters = np.zeros(self.convFilter.shape)
        for imagePatch, i, j in self.imageRegion(self.image):
            for k in range(self.numFilters):
                dL_dF_parameters[k] += imagePatch * dl_dout[i, j, k]
        
        # updating filter parameters
        self.convFilter -= learningRate * dL_dF_parameters
        return dL_dF_parameters