# get inputs and expected outputs
# initialize layers with inputs and outputs
# split inputs into batches
# apply relu and softmax activations on neurons
# feed forward to get output layer output
# calculate loss with categorical crossentropy (negative log)
# optimize w & b
from neural_network import *

import os
import numpy as np
import cv2
import os
import nnfs
from nnfs.datasets import spiral_data  # dataset
nnfs.init()  # sets np.random.seed(0) among other variables


def get_data():
    images = []
    labels = []
    data = ['Apples', 'Oranges']
    for i in range(len(data)):
        for image in os.listdir('data/' + data[i]):
            try:
                image = cv2.imread('data/' + data[i] + '/' + image)
                image = cv2.resize(image, (0, 0), None, 0.25, 0.25)  # reduce image dimension
                # image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                image = np.reshape(image, (-1,))
                image = image / 255
                images.append(image)
                labels.append(i)
            except Exception as e:
                print(e)

    return images, labels

# X, y = spiral_data(100, 3)  # 100 x, y coordinates per class, with 3 classes in a spiral pattern
# print(X.shape, y.shape)
X, y = get_data()
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

model = Model()
model.add(DenseLayer(16), ReLUActivation())
model.add(DenseLayer(32), ReLUActivation())
model.add(DenseLayer(8), ReLUActivation())
model.add(DenseLayer(2), SoftmaxActivation())
model.set_loss(CategoricalCrossentropy())
model.set_optimizer(Optimizer_Adam(learning_rate=0.005, decay=1e-3))

model.train(epochs=1000, batch_size=None, X=X, y=y)

#save model.json() to file model.json
with open('model.json', 'w') as f:
    f.write("model = " + model.json())

