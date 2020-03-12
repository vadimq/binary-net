import numpy as np
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
    def __init__(self, units, ternary_thresh=0, w_lr_scale='Glorot', **kwargs):
        super(Dense, self).__init__(units, **kwargs)
        self.quantization = quantization(ternary_thresh) if ternary_thresh else sign
        self.w_lr_scale = w_lr_scale

    def build(self, input_shape):
        super(Dense, self).build(input_shape)
        if self.w_lr_scale == 'Glorot':
            self.w_lr_scale = tf.cast(1 / np.sqrt(1.5 / (input_shape[-1] + self.units)), self.dtype)

    def call(self, inputs):
        kernel = self.kernel
        self.kernel = self.quantization(self.kernel)
        rvalue = super(Dense, self).call(inputs)
        self.kernel = kernel
        return rvalue

class Clip(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.clip_by_value(w, -1, 1)

class ClippingScaling(tf.keras.callbacks.Callback):
    def __init__(self, scale=False):
        super(ClippingScaling, self).__init__()
        self.scale = scale
        self.w = None

    def on_train_batch_begin(self, batch, logs=None):
        if self.w is None:
            self.w = [{"layer": l, "w": l.kernel} for l in self.model.layers if hasattr(l, "kernel")]
        for e in self.w:
            e["val"] = tf.identity(e["layer"].kernel)

    def on_train_batch_end(self, batch, logs=None):
        for e in self.w:
            val = e["val"] + e["layer"].w_lr_scale * (e["w"] - e["val"])
            val = tf.clip_by_value(val, -1, 1)
            e["w"].assign(val)
