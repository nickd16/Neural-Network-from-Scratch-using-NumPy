import numpy as np
import math
from activations import *

# "he" initialization with randomly shuffled mini-batches
def initialize(layers, X_train, Y_train, mini_batch_size):
    parameters = {}
    m = Y_train.shape[1]
    X, Y = [], []
    for l in range(1,len(layers)):
        parameters["W" + str(l)] = np.random.randn(layers[l], layers[l-1]) * np.sqrt(2/layers[l-1])
        parameters["b" + str(l)] = np.zeros((layers[l],1))
    permutation = list(np.random.permutation(m))
    shuffled_X = X_train[:, permutation]
    shuffled_Y = Y_train[:, permutation].reshape((1,m))
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for i in range(0, num_complete_minibatches):
        X.append(shuffled_X[:,i*mini_batch_size : (i+1)*mini_batch_size])
        Y.append(shuffled_Y[:,i*mini_batch_size : (i+1)*mini_batch_size])
    if m % mini_batch_size != 0:
        X.append(shuffled_X[:,int(m/mini_batch_size)*mini_batch_size:])
        Y.append(shuffled_Y[:,int(m/mini_batch_size)*mini_batch_size:])
    return parameters, X, Y

def forward_propogation(X, parameters):
    cache = []
    A_prev = X
    L = len(parameters) // 2
    for l in range (1, L):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        A = relu(Z)
        layer_cache = [W,b,Z,A_prev]
        cache.append(layer_cache)
        A_prev = A
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    Z = np.dot(W, A_prev) + b
    Y_hat = sigmoid(Z)
    Y_hat = np.absolute(Y_hat)
    Y_hat = Y_hat + 1e-8
    cache.append([W,b,Z,A_prev])
    return Y_hat, cache

# Cost function with L2 regularization 
def cost(Y_hat, Y, parameters, lambd):
    Y_predictions = np.copy(Y_hat)
    Y_predictions[Y_predictions<.5] = 0
    Y_predictions[Y_predictions>=.5] = 1
    accuracy = np.equal(Y, Y_predictions).mean()
    m = Y.shape[1]
    current_cost = (-1/m) * (np.dot(Y, np.log(Y_hat).T) + np.dot((1-Y), np.log(1-Y_hat).T))
    L2_regularized = 0
    for l in range(1, len(parameters) // 2):
        L2_regularized += np.sum(np.square(parameters['W' + str(l)]))
    L2_regularized *= lambd/(2*Y.shape[1])
    current_cost += L2_regularized
    return [np.squeeze(current_cost), accuracy]

def back_propogation(Y_hat, Y, caches, lambd):
    grads = {}
    L = len(caches)
    m = Y_hat.shape[1]
    Y = Y.reshape(Y_hat.shape)
    cache = caches[L-1]
    Z = cache[2]
    dAL = lossDerivative(Y, Y_hat)
    dZ = sigmoidDerivative(dAL, Z)
    grads["dW" + str(L)] = (1/m) * np.dot(dZ, cache[3].T) + (lambd/m)*cache[0]
    grads["db" + str(L)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    grads["dA" + str(L-1)] = np.dot(cache[0].T,dZ)
    for l in reversed(range(L-1)):
        cache = caches[l]
        m = cache[3].shape[1]
        Z = cache[2]
        dZ = reluDerivative(grads["dA" + str(l+1)], Z)
        grads["dW" + str(l+1)] = (1/m) * np.dot(dZ, cache[3].T) + (lambd/m)*cache[0]
        grads["db" + str(l+1)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        grads["dA" + str(l)] = np.dot(cache[0].T,dZ)
    return grads

def update(parameters, grads, learning_rate):
    for l in range(1, (len(parameters) // 2)+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - (learning_rate * grads["dW" + str(l)])
        parameters["b" + str(l)] = parameters["b" + str(l)] - (learning_rate * grads["db" + str(l)])
    return parameters
