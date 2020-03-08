import tensorflow as tf
from tensorflow.keras import layers
import binary_net

batch_size = 100
momentum = .9
units = 4096
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
y_train = 2 * tf.one_hot(y_train, 10) - 1
y_test = 2 * tf.one_hot(y_test, 10) - 1

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

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))
