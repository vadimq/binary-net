# %tensorflow_version 2.x
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# <codecell>

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import binary_net

seed = 0
batch_size = 50
momentum = .9
epochs = 5
# w_lr_scale = 1
w_lr_scale = "Glorot"
lr_initial = .001
lr_final = .0000003
lr_decay = (lr_final / lr_initial) ** (1 / epochs)

np.random.seed(seed)
tf.random.set_seed(seed)

# <codecell>

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Convert to [-1, 1].
x_train = (2 * (x_train / 255) - 1).astype(np.float32)
x_test = (2 * (x_test / 255) - 1).astype(np.float32)

# Convert to {-1, 1}.
y_train = (2 * tf.one_hot(np.squeeze(y_train), 10) - 1).numpy()
y_test = (2 * tf.one_hot(np.squeeze(y_test), 10) - 1).numpy()

x_val, x_train = x_train[45000:], x_train[:45000]
y_val, y_train = y_train[45000:], y_train[:45000]

# <codecell>

model = tf.keras.Sequential([
    binary_net.Conv2D(32, (3, 3), w_lr_scale=w_lr_scale, padding="same", use_bias=False, input_shape=(32, 32, 3)),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4, center=False, scale=False),
    layers.Activation(binary_net.sign_d_clipped),
    binary_net.Conv2D(32, (3, 3), w_lr_scale=w_lr_scale, padding="same", use_bias=False),
    layers.MaxPool2D(),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4, center=False, scale=False),
    layers.Activation(binary_net.sign_d_clipped),

    layers.Flatten(),
    binary_net.Dense(10, w_lr_scale=w_lr_scale, use_bias=False),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4, center=False, scale=False)])

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

# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[callback], validation_data=(x_val, y_val))
binary_net.train(model, x_train, y_train, batch_size, epochs, callback, x_val, y_val)
model.evaluate(x_test, y_test, batch_size=batch_size)
