import tensorflow as tf

@tf.custom_gradient
def sign_d_clipped(x):
    def grad(dy):
        return tf.where(tf.abs(x) <= 1, dy, tf.zeros_like(x))
    return tf.where(x < 0, -tf.ones_like(x), tf.ones_like(x)), grad

@tf.custom_gradient
def sign(x):
    def grad(dy):
        return dy
    # return tf.where(x < 0, -tf.ones_like(x), tf.ones_like(x)), grad
    return tf.sign(x), grad

@tf.custom_gradient
def threshold(input, thresh=.1):
    def grad(dy):
        return dy
    return tf.where(tf.abs(input) < thresh, tf.zeros_like(input), input), grad

class Clip(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.clip_by_value(w, -1, 1)

class Dense(tf.keras.layers.Dense):
    def call(self, inputs):
        kernel = self.kernel
        # self.kernel = sign(self.kernel)
        self.kernel = sign(threshold(self.kernel))
        rvalue = super(Dense, self).call(inputs)
        self.kernel = kernel
        return rvalue
