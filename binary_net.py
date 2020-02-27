import tensorflow as tf
# tf.enable_eager_execution()

@tf.custom_gradient
def sign(x, clip=False):
    def grad(dy):
        return tf.where(tf.abs(x) <= 1, dy, tf.zeros_like(x)) if clip else dy
    return tf.where(x < 0, -tf.ones_like(x), tf.ones_like(x)), grad
# print(sign(0))

class Clip(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.clip_by_value(w, -1, 1)

class Dense(tf.keras.layers.Dense):
    def call(self, inputs):
        Wr = self.kernel
        self.kernel = sign(self.kernel)
        rvalue = super(Dense, self).call(inputs)
        self.kernel = Wr
        return rvalue
