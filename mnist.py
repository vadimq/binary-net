import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import binary_net

batch_size = 100
momentum = .9
# units = 4096
units = 2048
hidden_layers = 3
epochs = 1000
dropout_in = 0
dropout_hidden = 0
w_lr_scale = 'Glorot'
initial_learning_rate = .003
decay_rate = .0000003 / initial_learning_rate

print('Loading data...')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
    path='mnist.npz')

# Convert to [-1, 1].
x_train = 2 * (x_train.reshape(-1, 784) / 255) - 1
x_test = 2 * (x_test.reshape(-1, 784) / 255) - 1

# Convert to {-1, 1}.
y_train = (2 * tf.one_hot(y_train, 10) - 1).numpy()
y_test = (2 * tf.one_hot(y_test, 10) - 1).numpy()

x_val, x_train = x_train[50000:], x_train[:50000]
y_val, y_train = y_train[50000:], y_train[:50000]

print('Building the model...')

inputs = tf.keras.Input(shape=(784,))
x = layers.Dropout(dropout_in)(inputs)
for i in range(hidden_layers):
    x = binary_net.Dense(units, w_lr_scale=w_lr_scale, use_bias=False)(x)
    x = layers.BatchNormalization(momentum=momentum, epsilon=1e-4, center=False, scale=False)(x)
    x = layers.Activation(binary_net.sign_d_clipped)(x)
    x = layers.Dropout(dropout_hidden)(x)
x = binary_net.Dense(10, w_lr_scale=w_lr_scale, use_bias=False)(x)
outputs = layers.BatchNormalization(momentum=momentum, epsilon=1e-4, center=False, scale=False)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, epochs, decay_rate)
opt = tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-8)

model.compile(optimizer=opt,
              loss=tf.keras.losses.squared_hinge,
              metrics=[tf.keras.losses.squared_hinge])

print('Training...')

# model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_val, y_val))

################################################################################

x_train, y_train = x_train[:500], y_train[:500]

def shuffle(x, y):
    order = np.random.permutation(x.shape[0])
    return x[order], y[order]

@tf.function
def train_batch(x_train_slice, y_train_slice):
    w = [(l, l.kernel, tf.identity(l.kernel)) for l in model.layers if hasattr(l, 'kernel')]

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

def train(num_steps):
    global x_train, y_train
    batches = x_train.shape[0] // batch_size
    for _ in range(num_steps):
        x_train, y_train = shuffle(x_train, y_train)
        loss = 0
        for i in range(batches):
            x_train_slice = x_train[i * batch_size:(i + 1) * batch_size]
            y_train_slice = y_train[i * batch_size:(i + 1) * batch_size]
            loss += train_batch(x_train_slice, y_train_slice)
        loss /= batches
        print("Training loss: {}".format(loss))
train(2)
