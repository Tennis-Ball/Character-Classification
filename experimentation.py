import numpy as np


layers = [("sigmoid", 10), ("sigmoid", 10), ("sigmoid", 10)]
biases = [np.random.randn(y,) for (_, y) in layers]

print(list(zip(layers[:-1], layers[1:])))

print(np.random.randn(10))
