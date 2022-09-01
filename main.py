## Importing files and libraries ##

# Importing important libraries
import numpy as np
import os
import cv2
import random

# Importing files
from CNN import Convolution
from maxpool import Maxpool
from softmax import Softmax
from train import train, test

## Functions ##

# Processing the images and createing dataset
def create_training_data(dataDir, categories, imgSize, trainingData):
    for category in categories:
        path = os.path.join(dataDir, category)
        classNum = categories.index(category)
        for img in os.listdir(path):
            try:
                imgArray = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                newArray = cv2.resize(imgArray, (imgSize, imgSize))
                trainingData.append([newArray, classNum])
            except Exception as e:
                pass
            
## Preprocessing ##

# Inputs for the processing function
dataDir = 'PetImages'
categories  =  ["Dog", "Cat"]
imgSize = 50
trainingData = []
create_training_data(dataDir, categories, imgSize, trainingData)

# Shuffling the dataset
random.shuffle(trainingData)

# Initiate our Xs and ys
X = []
y = []
for features, label in trainingData:
    X.append(features)
    y.append(label)

# Turn list X into numpy array
X = np.array(X)

# Split data into training and testing sets
Xtest, Xtrain = X[:500], X[24000:]

# Turn list y into numpy array
y = np.array(y)

# Split data into training and testing sets
ytest, ytrain = y[:500], y[24000:]

## Neural network ##
conv = Convolution(8, 3)
pool = Maxpool(2)
soft = Softmax(24 * 24 * 8, 10)
train(1, Xtrain, ytrain, 0.01, conv, pool, soft)
test(Xtest, ytest, conv, pool, soft)