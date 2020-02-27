import tensorflow as tf
# tf.enable_eager_execution()

@tf.custom_gradient
def sign(x, clip=False):
    def grad(dy):
        return tf.where(tf.abs(x) <= 1, dy, tf.zeros_like(x)) if clip else dy
    return tf.where(x < 0, -tf.ones_like(x), tf.ones_like(x)), grad
# print(sign(0))
