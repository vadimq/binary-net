import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import binary_net

batch_size = 10
epochs = 500
initial_learning_rate = .03
decay_rate = .000003 / initial_learning_rate

ki1 = 'glorot_uniform'
ki2 = 'glorot_uniform'
ki3 = 'glorot_uniform'

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
    binary_net.Dense(100, use_bias=False, kernel_initializer=ki1, kernel_constraint=binary_net.Clip()),
    layers.BatchNormalization(momentum=.9, epsilon=1e-4, center=False, scale=False),
    layers.Activation(binary_net.sign_d_clipped),
    binary_net.Dense(100, use_bias=False, kernel_initializer=ki2, kernel_constraint=binary_net.Clip()),
    layers.BatchNormalization(momentum=.9, epsilon=1e-4, center=False, scale=False),
    layers.Activation(binary_net.sign_d_clipped),
    binary_net.Dense(y.shape[1], use_bias=False, kernel_initializer=ki3, kernel_constraint=binary_net.Clip()),
    layers.BatchNormalization(momentum=.9, epsilon=1e-4, center=False, scale=False)])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, epochs, decay_rate)
# lr_schedule = .03
opt = tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-8)

'''
model.compile(optimizer=opt,
              loss=tf.keras.losses.squared_hinge,
              metrics=[tf.keras.losses.squared_hinge])

# model.fit(x, y, batch_size=batch_size, epochs=epochs)
for i in range(3):
    model.fit(x, y, batch_size=batch_size, epochs=1)
    print(model(x, training=True))

y_ = model.predict(x)
print(np.sum(np.sign(y_) == y) / y.size)
'''

################################################################################

model.build(x.shape)

for i in range(3):
    w = [(l.kernel, tf.identity(l.kernel)) for l in model.layers if hasattr(l, 'kernel')]

    with tf.GradientTape() as tape:
        y_ = model(x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.squared_hinge(y, y_))
    vars = model.trainable_variables
    grads = tape.gradient(loss, vars)
    opt.apply_gradients(zip(grads, vars))

    for e in w:
        e[0].assign(e[1] + (e[0] - e[1]))

    print(model(x, training=True))
