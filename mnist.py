# %tensorflow_version 2.x
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# <codecell>

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import binary_net

seed = 0
batch_size = 128
momentum = .9
units = 4096
# units = 2048
# units = 128
hidden_layers = 3
epochs = 1000
dropout_in = .2
dropout_hidden = .5
# w_lr_scale = 1
w_lr_scale = "Glorot"
lr_initial = .003
lr_final = .0000003
lr_decay = (lr_final / lr_initial) ** (1 / epochs)
save_path = "checkpoints/mnist_parameters"

np.random.seed(seed)
tf.random.set_seed(seed)  # Doesn't work well with distributed training.

# <codecell>

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
# strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

# <codecell>

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
    path="mnist.npz")

# Convert to [-1, 1].
x_train = (2 * (x_train.reshape(-1, 784) / 255) - 1).astype(np.float32)
x_test = (2 * (x_test.reshape(-1, 784) / 255) - 1).astype(np.float32)

# Convert to {-1, 1}.
y_train = (2 * tf.one_hot(y_train, 10) - 1).numpy()
y_test = (2 * tf.one_hot(y_test, 10) - 1).numpy()

x_val, x_train = x_train[50000:], x_train[:50000]
y_val, y_train = y_train[50000:], y_train[:50000]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).cache() \
                          .shuffle(x_train.shape[0]) \
                          .batch(batch_size, drop_remainder=True) \
                          .prefetch(tf.data.experimental.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).cache() \
                        .batch(batch_size, drop_remainder=True) \
                        .prefetch(tf.data.experimental.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).cache() \
                         .batch(batch_size, drop_remainder=True) \
                         .prefetch(tf.data.experimental.AUTOTUNE)

# <codecell>

with strategy.scope():
    inputs = tf.keras.Input(shape=(784,))
    x = layers.Dropout(dropout_in)(inputs)
    for i in range(hidden_layers):
        x = binary_net.Dense(units, w_lr_scale=w_lr_scale)(x)
        x = layers.BatchNormalization(momentum=momentum, epsilon=1e-4)(x)
        x = layers.Activation(binary_net.sign_d_clipped)(x)
        x = layers.Dropout(dropout_hidden)(x)
    x = binary_net.Dense(10, w_lr_scale=w_lr_scale)(x)
    outputs = layers.BatchNormalization(momentum=momentum, epsilon=1e-4)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

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
# binary_net.train(model, x_train, y_train, batch_size, epochs, callback, x_val, y_val, save_path)
model.evaluate(test_ds)
