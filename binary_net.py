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
    return tf.where(x < 0, -tf.ones_like(x), tf.ones_like(x)), grad

def quantization(thresh):
    @tf.custom_gradient
    def q(x):
        def grad(dy):
            return dy
        return tf.sign(tf.where(tf.abs(x) < thresh, tf.zeros_like(x), x)), grad
    return q

class Dense(tf.keras.layers.Dense):
    def __init__(self, units, thresh=0, w_lr_scale="Glorot", **kwargs):
        super(Dense, self).__init__(units, **kwargs)
        self.quantization = quantization(thresh) if thresh else sign
        self.w_lr_scale = w_lr_scale

    def build(self, input_shape):
        super(Dense, self).build(input_shape)
        if self.w_lr_scale == "Glorot":
            init = tf.sqrt(6 / (input_shape[-1] + self.units))
            # The BinaryConnect paper says that such scaling improves the
            # effectiveness, but doesn't say why.
            self.w_lr_scale = tf.cast(2 / init, self.dtype)

    def call(self, inputs):
        kernel = self.kernel
        self.kernel = self.quantization(self.kernel)
        rvalue = super(Dense, self).call(inputs)
        self.kernel = kernel
        return rvalue

class Clip(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.clip_by_value(w, -1, 1)
