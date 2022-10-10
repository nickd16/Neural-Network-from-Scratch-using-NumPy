from nn_functions import *
from deep_model import *
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets

def load_dataset():
    np.random.seed(2)
    train_X, train_Y = datasets.make_circles(n_samples=800, noise=.05)
    test_X, test_Y = datasets.make_circles(n_samples=1200, noise=.05)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

train_X, train_Y, test_X, test_Y = load_dataset()
layers = [2,10,10,10,1]
parameters = model(train_X, train_Y, layers, 0.01, 2000)
test(test_X, test_Y, parameters)




