import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from nn_functions import *
from deep_model import *

def inputs():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=None 
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=None 
    )
    train_X_orig = training_data.data.numpy()
    test_X_orig = test_data.data.numpy()
    train_Y_orig = training_data.train_labels.numpy()
    test_Y_orig = test_data.test_labels.numpy()
    train_X = train_X_orig.reshape(train_X_orig.shape[1]*train_X_orig.shape[2],train_X_orig.shape[0])
    test_X = test_X_orig.reshape(test_X_orig.shape[1]*test_X_orig.shape[2],test_X_orig.shape[0])
    train_Y = train_Y_orig.reshape(1, train_Y_orig.shape[0])
    test_Y = test_Y_orig.reshape(1, test_Y_orig.shape[0])
    train_Y[train_Y >= 1] = 10
    train_Y[train_Y <= 0] = 1
    train_Y[train_Y == 10] = 0
    test_Y[test_Y >= 1] = 10
    test_Y[test_Y <= 0] = 1
    test_Y[test_Y == 10] = 0
    return train_X, train_Y, test_X, test_Y

train_X, train_Y, test_X, test_Y = inputs()
layers = [784,1024,512,128,1]
parameters = model(train_X, train_Y, layers, 0.2, 10)
test(test_X, test_Y, parameters)


