import cv2 
import os

# Function that processes the images from a directory
def createTrainingData(dataDir, categories, imgSize, trainingData):
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