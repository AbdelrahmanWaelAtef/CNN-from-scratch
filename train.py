import numpy as np

def cnnForward(image, label, conv, pool, soft):
    output = conv.forward((image/255) - 0.5)
    output = pool.forward(output)
    output = soft.forward(output)
    
    # Calculate cross entropy loss
    crossEntropyLoss = - np.log(output[label])
    accuracyEvaluation = 1 if np.argmax(output) == label else 0
    
    return output, crossEntropyLoss, accuracyEvaluation


def training(image, label, learningRate, conv, pool, soft):
    # Forward
    out, loss, acc = cnnForward(image, label, conv, pool, soft)
    
    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]
    
    # Backward
    gradBack = soft.backward(gradient, learningRate)
    gradBack = pool.backward(gradBack)
    gradBack = conv.backward(gradBack, learningRate)
    
    return loss, acc
    
def train(epochs, trainImages, trainLabels, learningRate, conv, pool, soft):
    for epoch in range(epochs):
        print('Epoch %d --->' %(epoch + 1))
        
        # Training the CNN
        loss = 0
        numCorrect = 0
        for i, (im, label) in enumerate(zip(trainImages, trainLabels)):
            if i % 100 == 0:
                print('%d steps out of 100 steps: Average loss %.3f and accuracy: %d%%' %(i + 1, loss/100, numCorrect))
                loss = 0
                numCorrect = 0
            l1, accu = training(im, label, learningRate, conv, pool, soft)
            loss += l1
            numCorrect += accu
            
def test(testImages, testLabels, conv, pool, soft):
    print('**Testing phase')
    loss = 0
    numCorrect = 0
    for im, label in zip(testImages, testLabels):
        _, l1, accu = cnnForward(im, label, conv, pool, soft)
        loss += l1
        numCorrect += accu
    numTests = len(testImages)
    print('Test loss: ', loss/numTests)
    print('Test accuracy: ', numCorrect/numTests)