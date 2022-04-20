import numpy as np
import random

class NeuralNetwork:

    # layers in the format of [('activation function', size int)]
    def __init__ (self, layers):
        # init weights based on sizes and activations
        self.layers = layers
        self.biases = [np.random.randn(y,) for (_, y) in layers]
        self.weights = [np.random.randn(y, x) for ((_,x), (_,y)) in zip(layers[:-1], layers[1:])]

    def feed(self, input):
        # feed forward through the network
        for (b, w, l) in zip(self.biases, self.weights, self.layers):
            input = self.activation(l[0], np.dot(w, input) + b)
        return input

    def activation(self, activation, input):
        #switch statement
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-input))
        elif activation == 'relu':
            return np.where(input > 0, input, 0)
        else:
            return input


nn = NeuralNetwork([('sigmoid', 2), ('sigmoid', 2), ('sigmoid', 2)])
print(nn.feed(np.array([1, 1])))
