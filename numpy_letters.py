import numpy as np
import cv2
import os


def backprop(x, y):
    # nabla_b = [np.zeros(b.shape) for b in biases]
    # nabla_w = [np.zeros(w.shape) for w in weights]
    # feedforward
    activation = x
    activations = [x]  # list to store all the activations, layer by layer
    zs = []  # list to store all the z vectors, layer by layer
    for b, w, l in zip(biases, weights, layers):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = activation_func(l[0], z)
        activations.append(activation)
    # backward pass
    delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    for l in range(2, len(layers)):
        z = zs[-l]
        sp = activation_func_prime(layers[-l][0],z)
        delta = np.dot(weights[-l + 1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

    return (nabla_b, nabla_w)


def cost_derivative(output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
    return output_activations - y


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.where(z < 0, 0, z)


def relu_prime(z):
    return np.where(z < 0, 0, 1)


def activation_func(activation, input):
    if activation == 'sigmoid':
        return sigmoid(input)
    elif activation == 'relu':
        return relu(input)
    else:
        return input
def activation_func_prime(activation, input):
    if activation == 'sigmoid':
        return sigmoid_prime(input)
    elif activation == 'relu':
        return relu_prime(input)
    else:
        return input


def feedforward(a, biases, weights):
    for b, w in zip(biases, weights):
        a = sigmoid(np.dot(w, a) + b)
    return a


def get_data():
    images = []
    labels = []
    for apple_image in os.listdir('Apples/'):
        image = cv2.imread('Apples/' + apple_image)
        image = cv2.resize(image, (0, 0), None, 0.25, 0.25)  # reduce image dimension by 75% to 25x25
        # image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        image = np.reshape(image, (1875, 1))
        image = image / 255
        images.append(image)
        labels.append(0)
    for orange_image in os.listdir('Oranges/'):
        image = cv2.imread('Oranges/' + orange_image)
        image = cv2.resize(image, (0, 0), None, 0.25, 0.25)  # reduce image dimension by 75% to 25x25
        # image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        image = np.reshape(image, (1875, 1))
        image = image / 255
        images.append(image)
        labels.append(1)

    return images, labels


# layers in the format of [('activation function', size int)]
layers = [("sigmoid", 1875), ("sigmoid", 64), ("sigmoid", 16), ("sigmoid", 2)]
np.random.seed(1)  # 100%
# np.random.seed(2)  # 0%
# np.random.seed(3)  # 50%
biases = [np.random.randn(y, 1) for (_, y) in layers[1:]]
# biases = [np.random.standard_normal(size=(y, 1)) for (_, y) in layers[1:]]
# weights = [np.random.standard_normal(size=(y, x)) for (_, x), (_, y) in zip(layers[:-1], layers[1:])]
weights = [np.random.randn(y, x) / np.sqrt(x) for (_, x), (_, y) in zip(layers[:-1], layers[1:])]

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

epochs = 15
batch_size = 32
learning_rate = 1  # learning rate
training_images, training_labels = get_data()
training_labels = [vectorized_result(y) for y in training_labels]
training_data = list(zip(training_images, training_labels))
np.random.shuffle(training_data)

training_data = training_data[:int(len(training_data) * 0.8)]
test_data = training_data[int(len(training_data) * 0.8):]

for epoch in range(epochs):
    print('Epoch:', epoch, 'of', epochs, end=' - ')
    np.random.shuffle(training_data)
    batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
    for batch in batches:
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        weights = [w - (learning_rate / len(batch)) * nw for w, nw in zip(weights, nabla_w)]
        biases = [b - (learning_rate / len(batch)) * nb for b, nb in zip(biases, nabla_b)]

    test_results = [(np.argmax(feedforward(x, biases, weights)), y) for (x, y) in test_data]
    print(test_results)
    print('accuracy:', sum(int(x == y) for (x, y) in test_results) / len(test_results))


test_results = [(np.argmax(feedforward(x, biases, weights)), y) for (x, y) in test_data]
print(test_results)
print('accuracy:', sum(int(x == y) for (x, y) in test_results) / len(test_results))
