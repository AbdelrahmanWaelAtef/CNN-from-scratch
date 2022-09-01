# Importing libraries
import numpy as np
from layer import Layer

class Maxpool(Layer):
    def __init__(self, filterSize):
       self.filterSize = filterSize
       
    def imageRegion(self, image):
        newHeight = image.shape[0] // self.filterSize
        newWidth = image.shape[1] // self.filterSize
        self.image = image
        for i in range(newHeight):
            for j in range(newWidth):
                imagePatch = image[(i * self.filterSize) : (i * self.filterSize + self.filterSize), (j * self.filterSize) : (j * self.filterSize + self.filterSize)]
                yield imagePatch, i, j
                
    def forward(self, image):
        height, width, numFilters = image.shape
        output = np.zeros((height// self.filterSize, width//self.filterSize, numFilters))
        
        for imagePatch, i, j in self.imageRegion(image):
            output[i, j]  = np.amax(imagePatch, axis= (0, 1))
        
        return output
    
    def backward(self, dl_dout):
        dl_dmax_pool = np.zeros(self.image.shape)
        for imagePatch, i, j in self.imageRegion(self.image):
            height, width, numFilters = imagePatch.shape
            maxValue = np.amax(imagePatch, axis= (0, 1))
            
            for i1 in range(height):
                for j1 in range(width):
                    for k1 in range(numFilters):
                        if imagePatch[i1, j1, k1] == maxValue[k1]:
                            dl_dmax_pool[i * self.filterSize + i1, j * self.filterSize + j1, k1] = dl_dout[i, j, k1]
                            
        return dl_dmax_pool