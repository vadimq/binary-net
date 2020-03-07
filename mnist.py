import tensorflow as tf

print('Loading data...')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
    path='mnist.npz')

# Convert to [-1, 1].
x_train = 2 * (x_train.reshape(-1, 1, 28, 28) / 255) - 1
x_test = 2 * (x_test.reshape(-1, 1, 28, 28) / 255) - 1

# Convert to {-1, 1}.
y_train = 2 * tf.one_hot(y_train, 10) - 1
y_test = 2 * tf.one_hot(y_test, 10) - 1

x_valid = x_test
y_valid = y_test

print('Training...')
