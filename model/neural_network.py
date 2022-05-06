import json 
import numpy as np

class DenseLayer:  # layer class
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons

    def create(self, n_inputs):
        self.weights = 0.10 * np.random.randn(n_inputs, self.n_neurons)
        self.biases = np.zeros((1, self.n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases  # forward pass multiplying weights, adding biases

    def backward(self, delta_values):
        self.delta_weights = np.dot(self.inputs.transpose(), delta_values)  
        self.delta_biases = np.sum(delta_values, axis=0, keepdims=True)

        # TODO: for regularization penalty, might implement later
        """
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        """
        self.delta_inputs = np.dot(delta_values, self.weights.transpose())

    def json(self):
        return {
            'type': 'Dense',
            'weights': self.weights.tolist(),
            'biases': self.biases.tolist()
        }


# Adam optimizer
class Optimizer_Adam:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.delta_weights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.delta_biases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.delta_weights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.delta_biases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class ReLUActivation:  # ReLU activation
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)  # 0 if x<0, otherwise x

    def backward(self, delta_values):
        self.delta_inputs = delta_values.copy()
        self.delta_inputs[self.output <= 0] = 0  # if x<0, delta will be 0 because curve is flat

    def json(self):
        return 'ReLU'


class SigmoidActivation:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, delta_values):
        self.delta_inputs = delta_values * (self.output * (1 - self.output))

    def json(self):
        return 'Sigmoid'


class SoftmaxActivation:  # Softmax activation
    def forward(self, inputs):
        # e^inputs where max(inputs) = 0 so 0 <= e^inputs <= 1 (normalized outputs preventing overflow)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # axis=1 specifies rows (batches) and keepdims keeps the shape structure for matrix operations
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # probability 0-1 of each class
        self.output = probabilities

    def backward(self, delta_values):
        self.delta_inputs = np.empty_like(delta_values)  # Create uninitialized array
        for i, (output, delta_value) in enumerate(zip(self.output, delta_values)):  # Enumerate outputs and gradients
            output = output.reshape(-1, 1)  # flatten output
            jacobian = np.diagflat(output) - np.dot(output, output.transpose())  # Calculate Jacobian matrix of the output
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.delta_inputs[i] = np.dot(jacobian, delta_value)

    def json(self):
        return 'Softmax'


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
        # Number of samples
        samples = len(delta_values)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.delta_inputs = delta_values.copy()
        # Calculate gradient
        self.delta_inputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.delta_inputs = self.delta_inputs / samples


class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer, activation):
        self.layers.append((layer, activation))
        # self.layers.append(activation)

    def set_loss(self, loss_function):
        self.loss_function = loss_function

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def loss(self, true_y):
        return self.loss_function.calculate(self.layers[-1][1].output, true_y)

    def forward(self, inputs):
        for layer in self.layers:
            layer[0].forward(inputs)  # feed inputs forward through dense layer
            layer[1].forward(layer[0].output)  # feed dense layer outputs forward through activation
            inputs = layer[1].output
        return inputs

    def backward(self, true_y):
        self.loss_function.backward(self.layers[-1][1].output, true_y)
        delta_inputs = self.loss_function.delta_inputs
        for layer in reversed(self.layers):
            layer[1].backward(delta_inputs)  # activation backwards pass
            layer[0].backward(layer[1].delta_inputs)  # layer backwards pass
            delta_inputs = layer[0].delta_inputs
        return delta_inputs

    def optimize(self):
        self.optimizer.pre_update_params()
        for layer in self.layers:
            self.optimizer.update_params(layer[0])
        self.optimizer.post_update_params()

    def train(self, epochs, batch_size, X, y):
        #create weights and biases for each layer
        layer_size = X.shape[1]
        for layer in self.layers:
            layer[0].create(layer_size)
            layer_size = layer[0].n_neurons

        for epoch in range(epochs):
            permutation = np.random.permutation(len(X))  # create random permutation
            X, y = X[permutation], y[permutation]  # shuffle dataset

            if not batch_size:  # if batch_size == None pass entire dataset
                self.forward(X)  # forward pass
                self.backward(y)  # backwards pass
                self.optimize()  # Adam optimization of weights and biases

            else:
                X_batches = [X[k:k + batch_size] for k in range(0, len(X), batch_size)]
                y_batches = [y[k:k + batch_size] for k in range(0, len(y), batch_size)]

                for X_batch, y_batch in zip(X_batches, y_batches):  # update w + b for each batch
                    self.forward(X_batch)  # forward pass
                    self.backward(y_batch)  # backwards pass
                    self.optimize()  # Adam optimization of weights and biases

            if epoch % 100 == 0:
                accuracy_output = self.forward(X)
                loss = self.loss(y)  # calculate loss

                predictions = np.argmax(accuracy_output, axis=1)
                accuracy = np.mean(np.absolute(predictions - y) < np.std(y) / 250)  # calculate accuracy
                print(f'Epoch {epoch} of {epochs} - Loss: {loss}, Accuracy: {accuracy}')

    def json(self):
        # calls the json functions of each layer, which contains weights, biases, and type of activation
        # i don't believe we would need loss function or optimizer if we are only feeding forwards.
        data = [
            {'layer': layer[0].json(), 'activation': layer[1].json()} for layer in self.layers
        ]
        return json.dumps(data)


