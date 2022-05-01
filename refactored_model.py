# get inputs and expected outputs
# initialize layers with inputs and outputs
# split inputs into batches
# apply relu and softmax activations on neurons
# feed forward to get output layer output
# calculate loss with categorical crossentropy (negative log)
# optimize w & b
import numpy as np
import nnfs
from nnfs.datasets import spiral_data  # dataset
np.random.seed(0)
nnfs.init()


class DenseLayer:  # layer class
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # initialize 10% standard normal dist. weights
        self.biases = np.zeros((1, n_neurons))  # initialize all biases as 0

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases  # forward pass multiplying weights, adding biases

    def backward(self, delta_values):
        self.delta_weights = np.dot(self.inputs.transpose(), delta_values)  
        self.delta_biases = np.sum(delta_values, axis=0, keepdims=True)
        self.delta_inputs = np.dot(delta_values, self.weights.transpose())


class ReLUActivation:  # ReLU activation
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)  # 0 if x<0, otherwise x

    def backward(self, delta_values):
        self.delta_inputs = delta_values.copy()
        self.delta_inputs[self.output <= 0] = 0 # if x<0, delta will be 0 because curve is flat

class SigmoidActivation:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, delta_values):
        self.delta_inputs = delta_values * (self.output * (1 - self.output))

class SoftmaxActivation:  # Softmax activation
    def forward(self, inputs):
        # e^inputs where max(inputs) = 0 so 0 <= e^inputs <= 1 (normalized outputs preventing overflow)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # axis=1 specifies rows (batches) and keepdims keeps the shape structure for matrix operations
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # probability 0-1 of each class
        self.output = probabilities

    def backward(self, delta_values):
        self.delta_inputs = np.empty_like(delta_values)
        for i, (output, delta_value) in enumerate(zip(self.output, delta_values)):
            output = output.reshape(-1, 1)  # flatten output
            jacobian = np.diagflat(output) - np.dot(output, output.transpose())  # jacobian matrix
            self.delta_inputs[i] = np.dot(jacobian, delta_value) 

class Loss:
    def calculate(self, output, y):
        # sample_losses is a loss for each softmax output, one-hot encoded or sparse
        sample_losses = self.forward(output, y)  # called in CategoricalCrossentropy
        data_loss = np.mean(sample_losses)  # get average of all losses
        return data_loss


class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)  # clip into range [1e-7:1 - 1e-7] to prevent log0 (infinite)

        # correct_confidences is the largest value (most confident) in output batch
        if len(y_true.shape) == 1:  # if sparse [0, 1]
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:  # if one-hot encoded [[1, 0], [0, 1]]
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)  # loss
        return negative_log_likelihoods

    def backward(self, delta_values, y_true):
        samples = len(delta_values)
        labels = len(delta_values[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.delta_inputs = -y_true / delta_values / samples 

class Model: 
    def __init__(self):
        self.layers = []

    def add(self, layer, activation):
        self.layers.append(layer)
        self.layers.append(activation)

    def set_loss(self, loss_function):
        self.loss_function = loss_function

    def loss(self, y):
        return self.loss_function.calculate(self.layers[-1].output, y)

    def forward(self, inputs):
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output
        return inputs

    def backward(self, y):
        self.loss_function.backward(self.layers[-1].output, y)
        delta_inputs = self.loss_function.delta_inputs
        for layer in reversed(self.layers):
            layer.backward(delta_inputs)
            delta_inputs = layer.delta_inputs
        #TODO: optimizer

            



# X, y = "training data", "training labels"
X, y = spiral_data(100, 3)  # 100 x, y coordinates per class, with 3 classes in a spiral pattern

model = Model()
model.add(DenseLayer(2, 3), ReLUActivation())
model.add(DenseLayer(3, 3), ReLUActivation())
model.add(DenseLayer(3, 3), SoftmaxActivation())
model.set_loss(CategoricalCrossentropy())

output = model.forward(X)
print(output)

loss = model.loss(y)
print(loss)
