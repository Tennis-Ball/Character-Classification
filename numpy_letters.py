import numpy as np
import random

import cv2
import os

# layers in the format of [('activation function', size int)]
layers = [("sigmoid", 10), ("sigmoid", 10), ("sigmoid", 10)]
biases = [np.random.randn(y,) for (_, y) in layers]
weights = [np.random.randn(y, x) for ((_,x), (_,y)) in zip(layers[:-1], layers[1:])]

def activation(activation, input):
    #switch statement
    if activation == 'sigmoid':
        return 1 / (1 + np.exp(-input))
    elif activation == 'relu':
        return np.where(input > 0, input, 0)
    else:
        return input


print(feed(np.array([1, 1])))
