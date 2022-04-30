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

    def feed_forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases  # forward pass multiplying weights, adding biases


class ReLUActivation:  # ReLU activation
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)  # 0 if x<0, otherwise x


class SoftmaxActivation:  # Softmax activation
    def forward(self, inputs):
        # e^inputs where max(inputs) = 0 so 0 <= e^inputs <= 1 (normalized outputs preventing overflow)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # axis=1 specifies rows (batches) and keepdims keeps the shape structure for matrix operations
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # probability 0-1 of each class
        self.output = probabilities


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


model = [("relu", 2), ("relu", 4), ("sigmoid", 2)]  # will refactor so this is the model input
# X, y = "training data", "training labels"
X, y = spiral_data(100, 3)  # 100 x, y coordinates per class, with 3 classes in a spiral pattern


dense1 = DenseLayer(2, 3)  # define layer with input neurons and output neurons
activation1 = ReLUActivation()  # define activation (ReLU1, ReLU2, ... ReLUn-1, Softmaxn)
dense1.feed_forward(X)  # forward pass
activation1.forward(dense1.output)  # pass through activation, activation1.output is layer output

dense2 = DenseLayer(3, 3)
activation2 = SoftmaxActivation()
dense2.feed_forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])

loss_function = CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)  # calculate network loss
print("Loss:", loss)

predictions = np.argmax(activation2.output, axis=1)
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)
