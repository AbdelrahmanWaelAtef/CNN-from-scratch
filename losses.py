import numpy as np

# Create mean square error function
def mse(yTrue, yPred):
    return np.mean(np.power(yTrue - yPred, 2))

# Create derivative of the mean square error function
def msePrime(yTrue, yPred):
    return 2 * (yPred - yTrue) / np.size(yTrue)

# Create binary cross entropy loss function
def bce(yTrue, yPred):
    return -np.mean(yTrue * np.log(yPred) + (1 - yTrue) * np.log(1 - yPred))

# Create derivative of binary cross entropy loss function
def bcePrime(yTrue, yPred):
    return ((1 - yTrue)/(1 - yPred) - yTrue / yPred) / np.size(yTrue)