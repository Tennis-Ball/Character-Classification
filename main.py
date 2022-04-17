# let's assume that the goal of this program is to classify oranges and apples into their respective classes using ML
# We can either use an online dataset for websites like Kaggle or create our own through manual classification
# A CNN can be implemented using Google's Tensorflow library, we will create a relatively small model and fit our training data
# The model can be evaluated on a test set separate from the training set, and we will use validation during training for futher insight
# The most common splitting ratio is 60-20-20 among Train-Validation-Testing from the master dataset
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os


dataset_train = []
dataset_labels = []
for i in os.listdir('Apples'):  # append all dataset images
    image = cv2.imread('Apples/' + i, 0)
    image = cv2.resize(image, (0, 0), None, 0.25, 0.25)  # reduce 100x100 images to 25x25
    dataset_train.append(image)
for i in os.listdir('Oranges'):
    image = cv2.imread('Oranges/' + i, 0)
    image = cv2.resize(image, (0, 0), None, 0.25, 0.25)  # reduce 100x100 images to 25x25
    dataset_train.append(image)
for i in range(490):
    dataset_labels.append(0)  # 0 indicates class "apple"
for i in range(479):
    dataset_labels.append(1)  # 1 indicates class "orange"

dataset_size = len(dataset_train)

permutation = np.random.permutation(dataset_size)  # shuffle data to remove any order bias
dataset_train = np.array(dataset_train)
dataset_labels = np.array(dataset_labels)
dataset_train = dataset_train[permutation]
dataset_labels = dataset_labels[permutation]

print(np.shape(dataset_train))
print(np.shape(dataset_labels))
# prepare data
dataset_train = dataset_train.reshape(-1, 25, 25, 1)  # reshape to image dimension size
dataset_labels = to_categorical(dataset_labels)  # convert labels to one-hot vectors (binary)

train_data = dataset_train[:int(dataset_size // (5/3))]  # split into train, validation, and testing sets
val_data = dataset_train[int(dataset_size // (5/3)):int(dataset_size // 1.25)]
test_data = dataset_train[int(dataset_size // 1.25):]
train_labels = dataset_labels[:int(dataset_size // (5/3))]
val_labels = dataset_labels[int(dataset_size // (5/3)):int(dataset_size // 1.25)]
test_labels = dataset_labels[int(dataset_size // 1.25):]

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(25, 25, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())

model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),    # process the binary label inputs
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=10, batch_size=32,
                    validation_data=(val_data, val_labels))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
print(model.summary())
print('Test accuracy:', test_acc)
