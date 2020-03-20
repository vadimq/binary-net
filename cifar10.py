# %tensorflow_version 2.x
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# <codecell>

import numpy as np
import tensorflow as tf

# <codecell>

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Convert to [-1, 1].
x_train = (2 * (x_train / 255) - 1).astype(np.float32)
x_test = (2 * (x_test / 255) - 1).astype(np.float32)

# Convert to {-1, 1}.
y_train = (2 * tf.one_hot(y_train, 10) - 1).numpy()
y_test = (2 * tf.one_hot(y_test, 10) - 1).numpy()

x_val, x_train = x_train[45000:], x_train[:45000]
y_val, y_train = y_train[45000:], y_train[:45000]
