# %tensorflow_version 2.x
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# <codecell>

import tensorflow as tf

# <codecell>

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
