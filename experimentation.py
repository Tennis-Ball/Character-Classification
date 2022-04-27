# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt
#
# def backprop(x, y):
#     nabla_b = [np.zeros(b.shape) for b in biases]
#     nabla_w = [np.zeros(w.shape) for w in weights]
#     # feedforward
#     activation = x
#     activations = [x]  # list to store all the activations, layer by layer
#     zs = []  # list to store all the z vectors, layer by layer
#     for b, w, l in zip(biases, weights, layers):
#         z = np.dot(w, activation) + b
#         zs.append(z)
#         activation = activation_func(l[0],z)
#         activations.append(activation)
#     # backward pass
#     delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
#     nabla_b[-1] = delta
#     nabla_w[-1] = np.dot(delta, activations[-2].transpose())
#     for l in range(2, len(layers)):
#         z = zs[-l]
#         sp = activation_func_prime(layers[-l][0],z)
#         delta = np.dot(weights[-l + 1].transpose(), delta) * sp
#         nabla_b[-l] = delta
#         nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
#
#     return (nabla_b, nabla_w)
#
#
# def cost_derivative(output_activations, y):
#     """Return the vector of partial derivatives \partial C_x /
#     \partial a for the output activations."""
#     return (output_activations - y)
#
#
# def sigmoid(z):
#     """The sigmoid function."""
#     return 1.0 / (1.0 + np.exp(-z))
#
#
# def sigmoid_prime(z):
#     """Derivative of the sigmoid function."""
#     return sigmoid(z) * (1 - sigmoid(z))
#
#
# def relu(z):
#     return np.where(z < 0, 0, z)
#
#
# def relu_prime(z):
#     return np.where(z < 0, 0, 1)
#
#
# def activation_func(activation, input):
#     if activation == 'sigmoid':
#         return sigmoid(input)
#     elif activation == 'relu':
#         return relu(input)
#     else:
#         return input
# def activation_func_prime(activation, input):
#     if activation == 'sigmoid':
#         return sigmoid_prime(input)
#     elif activation == 'relu':
#         return relu_prime(input)
#     else:
#         return input
#
#
# def feedforward(a, biases, weights):
#     for b, w in zip(biases, weights):
#         a = sigmoid(np.dot(w, a) + b)
#     return a
#
#
# def get_data():
#     images = []
#     labels = []
#     for apple_image in os.listdir('Apples/'):
#         image = cv2.imread('Apples/' + apple_image)
#         image = cv2.resize(image, (0, 0), None, 0.25, 0.25)  # reduce image dimension by 75% to 25x25
#         # image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
#         image = np.reshape(image, (1875, 1))
#         image = image / 255
#         images.append(image)
#         labels.append(0)
#     for orange_image in os.listdir('Oranges/'):
#         image = cv2.imread('Oranges/' + orange_image)
#         image = cv2.resize(image, (0, 0), None, 0.25, 0.25)  # reduce image dimension by 75% to 25x25
#         # image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
#         image = np.reshape(image, (1875, 1))
#         image = image / 255
#         images.append(image)
#         labels.append(1)
#
#     return images, labels
#
#
# finals = []
# training_images, training_labels = get_data()
# training_data = list(zip(training_images, training_labels))
# np.random.shuffle(training_data)
#
# training_data = training_data[:int(len(training_data) * 0.8)]
# test_data = training_data[int(len(training_data) * 0.8):]
# for i in range(10):
#     np.random.seed(i)
#     # layers in the format of [('activation function', size int)]
#     layers = [("sigmoid", 1875), ("sigmoid", 8), ("sigmoid", 2)]
#     biases = [np.random.randn(y, 1) for (_, y) in layers[1:]]
#     weights = [np.random.randn(y, x) for (_, x), (_, y) in zip(layers[:-1], layers[1:])]
#
#     epochs = 50
#     batch_size = 32
#     learning_rate = 0.1  # learning rate
#
#     for epoch in range(epochs):
#         print('Epoch:', epoch, 'of', epochs)
#         np.random.shuffle(training_data)
#         batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
#         for batch in batches:
#             nabla_b = [np.zeros(b.shape) for b in biases]
#             nabla_w = [np.zeros(w.shape) for w in weights]
#             for x, y in batch:
#                 delta_nabla_b, delta_nabla_w = backprop(x, y)
#                 nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
#                 nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
#
#             weights = [w - (learning_rate / len(batch)) * nw for w, nw in zip(weights, nabla_w)]
#             biases = [b - (learning_rate / len(batch)) * nb for b, nb in zip(biases, nabla_b)]
#
#     test_results = [(np.argmax(feedforward(x, biases, weights)), y) for (x, y) in test_data]
#     accuracy = sum(int(x == y) for (x, y) in test_results) / len(test_results)
#     print(accuracy)
#     finals.append((weights, biases, accuracy))
#
# gweights = []
# gbiases = []
# bweights = []
# bbiases = []
# all_weights = []
# all_biases = []
#
# for i in finals:
#     if i[-1] > 0.95:
#         gweights.append(np.average(i[0][0]))
#         gbiases.append(np.average(i[1][0]))
#     elif i[-1] < 0.05:
#         bweights.append(np.average(i[0][0]))
#         bbiases.append(np.average(i[1][0]))
#     all_weights.append(np.average(i[0][0]))
#     all_biases.append(np.average(i[1][0]))
#
# plt.scatter(range(len(gweights)), gweights, color='blue')
# plt.scatter(range(len(gbiases)), gbiases, color='purple')
# plt.scatter(range(len(bweights)), bweights, color='red')
# plt.scatter(range(len(bbiases)), bbiases, color='orange')
# plt.scatter(range(len(all_weights)), all_weights, color='green')
# plt.scatter(range(len(all_biases)), all_biases, color='black')
# plt.show()
#
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # outcomes = [0.6064516129032258, 1.0, 0.5612903225806452, 0.4967741935483871, 0.2903225806451613, 0.33548387096774196, 0.9419354838709677, 0.896774193548387, 1.0, 0.432258064516129, 0.0064516129032258064, 0.16129032258064516, 0.03225806451612903, 0.5806451612903226, 0.5548387096774193, 0.9612903225806452, 0.22580645161290322, 0.8193548387096774, 0.0, 0.025806451612903226, 1.0, 0.9870967741935484, 0.04516129032258064, 0.6451612903225806, 0.33548387096774196, 0.8516129032258064, 1.0, 0.5548387096774193, 0.6516129032258065, 0.4967741935483871, 0.05161290322580645, 0.9935483870967742, 0.9290322580645162, 0.0064516129032258064, 0.5290322580645161, 0.5741935483870968, 0.08387096774193549, 0.025806451612903226, 0.4645161290322581, 1.0, 0.9870967741935484, 0.01935483870967742, 0.2967741935483871, 0.16774193548387098, 0.9548387096774194, 0.9806451612903225, 0.18064516129032257, 0.8774193548387097, 0.21935483870967742, 0.9870967741935484, 1.0, 0.9354838709677419, 0.9870967741935484, 0.9612903225806452, 0.3419354838709677, 0.025806451612903226, 0.0, 0.7806451612903226, 0.7225806451612903, 0.07741935483870968, 0.09032258064516129, 0.025806451612903226, 0.38064516129032255, 0.6193548387096774, 0.9419354838709677, 0.0, 0.5870967741935483, 0.8709677419354839, 0.9741935483870968, 0.12258064516129032, 0.07741935483870968, 1.0, 0.7612903225806451, 0.36774193548387096, 0.0, 0.06451612903225806, 0.3870967741935484, 0.12258064516129032, 0.45161290322580644, 0.05161290322580645, 0.5548387096774193, 0.012903225806451613, 1.0, 0.4838709677419355, 0.6451612903225806, 0.9870967741935484, 0.2709677419354839, 0.03870967741935484, 0.0, 0.18064516129032257, 0.9225806451612903, 0.8580645161290322, 0.5935483870967742, 0.3096774193548387, 0.012903225806451613, 0.967741935483871, 0.2838709677419355, 0.0, 0.8258064516129032, 0.13548387096774195]
# # layers = [("sigmoid", 1875), ("sigmoid", 8), ("sigmoid", 2)]
# # good0 = []
# # good1 = []
# # bad0 = []
# # bad1 = []
# # all = []
# # for i in range(len(outcomes)):
# #     np.random.seed(i)
# #     biases = [np.random.randn(y, 1) for (_, y) in layers[1:]]
# #     weights = [np.random.randn(y, x) for (_, x), (_, y) in zip(layers[:-1], layers[1:])]
# #     if outcomes[i] > 0.95:
# #         print('Good:', np.average(weights[0]), np.average(weights[1]))
# #         good0.append(np.average(weights[0]))
# #         good1.append(np.average(weights[1]))
# #     elif outcomes[i] < 0.05:
# #         print('Bad:', np.average(weights[0]), np.average(weights[1]))
# #         bad0.append(np.average(weights[0]))
# #         bad1.append(np.average(weights[1]))
# #     all.append(np.average(weights[1]))
# #
# # # plt.scatter(range(len(good0)), good0, color='blue')
# # plt.scatter(range(len(good0)), good1, color='purple')
# # # plt.scatter(range(len(bad0)), bad0, color='red')
# # plt.scatter(range(len(bad0)), bad1, color='orange')
# # # plt.scatter(range(len(all)), all, color='black')
# # plt.show()
import numpy as np
print(np.argmax([0.4, 0.4]))
