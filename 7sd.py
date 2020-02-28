import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import binary_net

x = np.array([[0, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 1, 1],
              [0, 1, 0, 0],
              [0, 1, 0, 1],
              [0, 1, 1, 0],
              [0, 1, 1, 1],
              [1, 0, 0, 0],
              [1, 0, 0, 1]]) * 2 - 1
y = np.array([[1, 1, 1, 1, 1, 1, 0],
              [0, 1, 1, 0, 0, 0, 0],
              [1, 1, 0, 1, 1, 0, 1],
              [1, 1, 1, 1, 0, 0, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [1, 0, 1, 1, 0, 1, 1],
              [1, 0, 1, 1, 1, 1, 1],
              [1, 1, 1, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 0, 0, 1, 1]]) * 2 - 1

model = tf.keras.Sequential([
    binary_net.Dense(100, use_bias=False, kernel_constraint=binary_net.Clip()),
    layers.BatchNormalization(momentum=.9, epsilon=1e-4, center=False, scale=False),
    layers.Activation(binary_net.sign_clipped_d),
    binary_net.Dense(100, use_bias=False, kernel_constraint=binary_net.Clip()),
    layers.BatchNormalization(momentum=.9, epsilon=1e-4, center=False, scale=False),
    layers.Activation(binary_net.sign_clipped_d),
    binary_net.Dense(y.shape[1], use_bias=False, kernel_constraint=binary_net.Clip()),
    layers.BatchNormalization(momentum=.9, epsilon=1e-4, center=False, scale=False)])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=.03),
              loss=tf.keras.losses.squared_hinge,
              metrics=[tf.keras.losses.squared_hinge])

model.fit(x, y, batch_size=10, epochs=500)

y_ = model.predict(x)
print(np.sum(np.sign(y_) == y) / y.size)
