# %tensorflow_version 2.x
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# <codecell>

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time
import binary_net

seed = 0
batch_size = 100
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
tf.random.set_seed(seed)

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

# <codecell>

ki = tf.random_uniform_initializer(-1, 1)
inputs = tf.keras.Input(shape=(784,))
x = layers.Dropout(dropout_in)(inputs)
for i in range(hidden_layers):
    x = binary_net.Dense(units, w_lr_scale=w_lr_scale, kernel_initializer=ki)(x)
    x = layers.BatchNormalization(momentum=momentum, epsilon=1e-4)(x)
    x = layers.Activation(binary_net.sign_d_clipped)(x)
    x = layers.Dropout(dropout_hidden)(x)
x = binary_net.Dense(10, w_lr_scale=w_lr_scale, kernel_initializer=ki)(x)
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

# model.fit(x_train, y_train, batch_size=batch_size, epochs=2, callbacks=[callback], validation_data=(x_val, y_val))

# <codecell>

def shuffle(x, y):
    order = np.random.permutation(x.shape[0])
    return x[order], y[order]

@tf.function
def train_batch(x_train_slice, y_train_slice):
    w = [(l, l.kernel, tf.identity(l.kernel)) for l in model.layers
         if isinstance(l, binary_net.Dense)]

    with tf.GradientTape() as tape:
        y_ = model(x_train_slice, training=True)
        loss = tf.reduce_mean(tf.keras.losses.squared_hinge(y_train_slice, y_))
    vars = model.trainable_variables
    grads = tape.gradient(loss, vars)
    opt.apply_gradients(zip(grads, vars))

    for e in w:
        val = e[2] + e[0].w_lr_scale * (e[1] - e[2])
        val = tf.clip_by_value(val, -1, 1)
        e[1].assign(val)
    return loss

def train(num_epochs):
    global x_train, y_train
    best_val_err = 100
    best_epoch = 1
    batches = x_train.shape[0] // batch_size
    for i in range(num_epochs):
        start = time.time()
        callback.on_epoch_begin(i)
        x_train, y_train = shuffle(x_train, y_train)

        loss = 0
        for j in range(batches):
            x_train_slice = x_train[j * batch_size:(j + 1) * batch_size]
            y_train_slice = y_train[j * batch_size:(j + 1) * batch_size]
            loss += train_batch(x_train_slice, y_train_slice)
        loss /= batches

        result = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
        result[2] = (1 - result[2]) * 100
        if result[2] <= best_val_err:
            best_val_err = result[2]
            best_epoch = i + 1

            if save_path is not None:
                model.save_weights(save_path)

        duration = time.time() - start
        lr = opt._decayed_lr(tf.float32).numpy()
        print("Epoch {} of {} took {} s.".format(i + 1, num_epochs, duration))
        print("  LR:                         {}".format(lr))
        print("  training loss:              {}".format(loss))
        print("  validation loss:            {}".format(result[0]))
        print("  validation error rate:      {}%".format(result[2]))
        print("  best epoch:                 {}".format(best_epoch))
        print("  best validation error rate: {}%".format(best_val_err))

# <codecell>

train(epochs)
model.evaluate(x_test, y_test, batch_size=batch_size)
