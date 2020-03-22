# %tensorflow_version 2.x
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# <codecell>

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import binary_net

batch_size = 50
momentum = .9
epochs = 500
lr_initial = .001
lr_final = .0000003
lr_decay = (lr_final / lr_initial) ** (1 / epochs)

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

# <codecell>

model = tf.keras.Sequential([
    layers.Conv2D(128, (3, 3), padding="same", input_shape=(32, 32, 3)),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),
    layers.Conv2D(128, (3, 3), padding="same"),
    layers.MaxPool2D(),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),

    layers.Conv2D(256, (3, 3), padding="same"),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),
    layers.Conv2D(256, (3, 3), padding="same"),
    layers.MaxPool2D(),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),

    layers.Conv2D(512, (3, 3), padding="same"),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),
    layers.Conv2D(512, (3, 3), padding="same"),
    layers.MaxPool2D(),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),

    layers.Flatten(),
    binary_net.Dense(1024),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),
    binary_net.Dense(1024),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),
    binary_net.Dense(10),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4)])

def schedule(epoch, lr):
    return lr * lr_decay
callback = tf.keras.callbacks.LearningRateScheduler(schedule)
callback.set_model(model)
opt = tf.keras.optimizers.Adam(lr_initial / lr_decay, epsilon=1e-8)

model.compile(optimizer=opt,
              loss=tf.keras.losses.squared_hinge,
              metrics=[tf.keras.losses.squared_hinge,
                       tf.keras.metrics.CategoricalAccuracy()])

# <codecell>

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[callback], validation_data=(x_val, y_val))
