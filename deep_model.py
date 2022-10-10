from nn_functions import *
from matplotlib import pyplot as plt
import numpy as np

def model(X, Y, layers, learning_rate, iterations):
    parameters, mini_batch_X, mini_batch_Y = initialize(layers, X, Y, 128)
    costs = []
    for i in range(0, iterations):
        for j in range(len(mini_batch_X)):
            Y_hat, cache = forward_propogation(mini_batch_X[j], parameters)
            current_cost = cost(Y_hat, mini_batch_Y[j], parameters, 0.7)
            grads = back_propogation(Y_hat, mini_batch_Y[j], cache, lambd = 0.7)
            parameters = update(parameters, grads, learning_rate)
            if(j % 100 == 0):
                costs.append(current_cost[0])
        if(i % 100 == 0):
            print("Cost after epoch %i: %f Accuracy: %g" %(i+100, current_cost[0], current_cost[1]))
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("epochs")
    plt.title("Cost per epoch")
    plt.show()
    return parameters

def test(X, Y, parameters):
    Y_hat, _ = forward_propogation(X, parameters)
    current_cost = cost(Y_hat, Y, parameters, lambd = 0.7)
    print("Test Accuracy = " + str(current_cost[1]))