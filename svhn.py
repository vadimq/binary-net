# %tensorflow_version 2.x
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# <codecell>

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import binary_net
import svhn_data

seed = 0
batch_size = 50
momentum = .9
epochs = 500
# w_lr_scale = 1
w_lr_scale = "Glorot"
lr_initial = .001
lr_final = .0000003
lr_decay = (lr_final / lr_initial) ** (1 / epochs)

np.random.seed(seed)
tf.random.set_seed(seed)

# <codecell>

def preprocess(x, y):
    # Convert to [-1, 1].
    x = 2 * (x / 255) - 1

    # Convert to {-1, 1}.
    y = 2 * tf.one_hot(y, 10) - 1
    return x, y

train_ds, val_ds, test_ds = svhn_data.make_data()
train_ds = train_ds.shuffle(1024).batch(batch_size).map(preprocess) \
                   .prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_ds.batch(batch_size).map(preprocess) \
               .prefetch(tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(batch_size).map(preprocess) \
                 .prefetch(tf.data.experimental.AUTOTUNE)

# <codecell>

model = tf.keras.Sequential([
    binary_net.Conv2D(128, (3, 3), w_lr_scale=w_lr_scale, padding="same",
                      input_shape=(32, 32, 3)),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),
    binary_net.Conv2D(128, (3, 3), w_lr_scale=w_lr_scale, padding="same"),
    layers.MaxPool2D(),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),

    binary_net.Conv2D(256, (3, 3), w_lr_scale=w_lr_scale, padding="same"),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),
    binary_net.Conv2D(256, (3, 3), w_lr_scale=w_lr_scale, padding="same"),
    layers.MaxPool2D(),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),

    binary_net.Conv2D(512, (3, 3), w_lr_scale=w_lr_scale, padding="same"),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),
    binary_net.Conv2D(512, (3, 3), w_lr_scale=w_lr_scale, padding="same"),
    layers.MaxPool2D(),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),

    layers.Flatten(),
    binary_net.Dense(1024, w_lr_scale=w_lr_scale),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),
    binary_net.Dense(1024, w_lr_scale=w_lr_scale),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),
    binary_net.Dense(10, w_lr_scale=w_lr_scale),
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

model.fit(train_ds, epochs=epochs, callbacks=[callback], validation_data=val_ds)
# binary_net.train(model, x_train, y_train, batch_size, epochs, callback, x_val, y_val)
model.evaluate(test_ds)
