import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Construct a tf.data.Dataset
ds, test = tfds.load('emnist/balanced', split=['train', 'test'], shuffle_files=True)
images = []
labels = []
# Build your input pipeline
ds = ds.shuffle(2048).prefetch(tf.data.AUTOTUNE)
for example in ds.take(1):
    print(example["image"].shape)
    image, label = np.array(example["image"]), np.array(example["label"])
    image = np.rot90(image, 3)  # rotate once clockwise to adjust for readability
    image = np.flip(image, 1)  # flip horizontally to adjust for readability
    # change pixels to 0 or 1, depending on threshold
    np.place(image, image<125, [0])
    np.place(image, image>=125, [1])
    # image = np.reshape(image, (-1,))
    images.append(image)
    labels.append(label)

import matplotlib.pyplot as plt
print(len(images))
plt.imshow(images[0])
plt.title(labels[0])
plt.show()
