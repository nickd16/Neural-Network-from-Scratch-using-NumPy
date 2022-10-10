import numpy as np

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def relu(Z):
    return (np.maximum(0,Z))

def sigmoidDerivative(dA, Z):
    return dA * (sigmoid(Z) * (1-sigmoid(Z)))

def reluDerivative(dA, Z):
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    return dZ

def lossDerivative(Y, Y_hat):
    return - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))









