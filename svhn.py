# %tensorflow_version 2.x
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# <codecell>

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import binary_net
import svhn_data

seed = 0
batch_size = 50
momentum = .9
epochs = 200
# w_lr_scale = 1
w_lr_scale = "Glorot"
lr_initial = .001
lr_final = .000001
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
    binary_net.Conv2D(64, (3, 3), w_lr_scale=w_lr_scale, padding="same",
                      input_shape=(32, 32, 3)),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),
    binary_net.Conv2D(64, (3, 3), w_lr_scale=w_lr_scale, padding="same"),
    layers.MaxPool2D(),
    layers.BatchNormalization(momentum=momentum, epsilon=1e-4),
    layers.Activation(binary_net.sign_d_clipped),

    binary_net.Conv2D(128, (3, 3), w_lr_scale=w_lr_scale, padding="same"),
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

def train(model, dataset, epochs, callback, validation_data, save_path=None):
    @tf.function
    def train_batch(x, y):
        w = [(l, l.kernel, tf.identity(l.kernel)) for l in model.layers
             if isinstance(l, binary_net.Dense) or
                isinstance(l, binary_net.Conv2D)]

        with tf.GradientTape() as tape:
            y_ = model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.squared_hinge(y, y_))
        vars = model.trainable_variables
        grads = tape.gradient(loss, vars)
        model.optimizer.apply_gradients(zip(grads, vars))

        for e in w:
            val = e[2] + e[0].w_lr_scale * (e[1] - e[2])
            val = tf.clip_by_value(val, -1, 1)
            e[1].assign(val)
        return loss

    best_val_err = 100
    best_epoch = 1
    for i in range(epochs):
        start = time.time()
        callback.on_epoch_begin(i)

        batches = 0
        loss = 0
        for x, y in dataset:
            loss += train_batch(x, y)
            batches += 1
            # print(batches, loss / batches)
        loss /= batches

        result = model.evaluate(validation_data, verbose=0)
        val_err = (1 - result[2]) * 100
        if val_err <= best_val_err:
            best_val_err = val_err
            best_epoch = i + 1

            if save_path is not None:
                model.save_weights(save_path)

        duration = time.time() - start
        lr = model.optimizer._decayed_lr(tf.float32).numpy()
        print("Epoch {} of {} took {} s.".format(i + 1, epochs, duration))
        print("  LR:                         {}".format(lr))
        print("  training loss:              {}".format(loss))
        print("  validation loss:            {}".format(result[0]))
        print("  validation error rate:      {}%".format(val_err))
        print("  best epoch:                 {}".format(best_epoch))
        print("  best validation error rate: {}%".format(best_val_err))

# <codecell>

# model.fit(train_ds, epochs=epochs, callbacks=[callback], validation_data=val_ds)
train(model, train_ds, epochs, callback, val_ds)
model.evaluate(test_ds)
