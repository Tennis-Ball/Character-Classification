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
import tensorflow as tf
import tensorflow_datasets as tfds

"""
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
"""
def get_data():
    ds = tfds.load('emnist/balanced', split='train', shuffle_files=True)
    images = []
    labels = []
    ds = ds.shuffle(2048).prefetch(tf.data.AUTOTUNE)  # fetch cached dataset + preform shuffle
    # Loading takes a while
    for example in ds:
        image, label = np.array(example["image"]), np.array(example["label"])

        image = np.rot90(image, 3)  # rotate once clockwise to adjust for readability
        image = np.flip(image, 1)  # flip horizontally to adjust for readability
        # change pixels to 0 or 1, depending on threshold
        np.place(image, image < 125, [0])
        np.place(image, image >= 125, [1])
        image = np.reshape(image, (-1,))

        images.append(image)
        labels.append(label)
    return images, labels


X, y = get_data()
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

model = Model()
model.add(DenseLayer(128), ReLUActivation())
model.add(DenseLayer(64), ReLUActivation())
model.add(DenseLayer(64), ReLUActivation())
model.add(DenseLayer(47), SoftmaxActivation())
model.set_loss(CategoricalCrossentropy())
model.set_optimizer(Optimizer_Adam(learning_rate=0.005, decay=1e-3))

model.train(epochs=200, batch_size=128, X=X, y=y, Xv=None, yv=None)

#save model.json() to file model.json
with open('model.json', 'w') as f:
    f.write("model = " + model.json())
