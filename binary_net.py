import time
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
    def __init__(self, units, thresh=0, w_lr_scale="Glorot",
                 kernel_initializer=tf.random_uniform_initializer(-1, 1),
                 **kwargs):
        super(Dense, self).__init__(units,
                                    kernel_initializer=kernel_initializer,
                                    **kwargs)
        self.quantization = quantization(thresh) if thresh else sign
        self.w_lr_scale = w_lr_scale

    def build(self, input_shape):
        super(Dense, self).build(input_shape)
        if self.w_lr_scale == "Glorot":
            init = tf.sqrt(6 / (input_shape[-1] + self.units))
            # The BinaryConnect paper says that such scaling improves the
            # effectiveness, but doesn't say why.
            init /= 2
            self.w_lr_scale = tf.cast(1 / init, self.dtype)

    def call(self, inputs):
        kernel = self.kernel
        self.kernel = self.quantization(self.kernel)
        rvalue = super(Dense, self).call(inputs)
        self.kernel = kernel
        return rvalue

class Conv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, thresh=0, w_lr_scale="Glorot",
                 kernel_initializer=tf.random_uniform_initializer(-1, 1),
                 **kwargs):
        super(Conv2D, self).__init__(filters, kernel_size,
                                     kernel_initializer=kernel_initializer,
                                     **kwargs)
        self.quantization = quantization(thresh) if thresh else sign
        self.w_lr_scale = w_lr_scale

    def build(self, input_shape):
        super(Conv2D, self).build(input_shape)
        if self.w_lr_scale == "Glorot":
            ca = self._get_channel_axis()
            init = tf.sqrt(6 / (np.prod(self.kernel_size) * input_shape[ca] +
                                np.prod(self.kernel_size) * self.filters))
            init /= 2
            self.w_lr_scale = tf.cast(1 / init, self.dtype)

    def call(self, inputs):
        kernel = self.kernel
        self.kernel = self.quantization(self.kernel)
        rvalue = super(Conv2D, self).call(inputs)
        self.kernel = kernel
        return rvalue

class Clip(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.clip_by_value(w, -1, 1)

def shuffle(x, y):
    order = np.random.permutation(x.shape[0])
    return x[order], y[order]

def train(model, x, y, batch_size, epochs, callback, x_val, y_val,
          save_path=None):
    @tf.function
    def train_batch(x, y):
        w = [(l, l.kernel, tf.identity(l.kernel)) for l in model.layers
             if isinstance(l, Dense) or isinstance(l, Conv2D)]

        with tf.GradientTape() as tape:
            y_ = model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.squared_hinge(y, y_))
        vars = model.trainable_variables
        grads = tape.gradient(loss, vars)
        model.optimizer.apply_gradients(zip(grads, vars))

        for e in w:
            val = e[2] + e[0].w_lr_scale * (e[1] - e[2])
            val = tf.clip_by_value(val, -1, 1)
            e[1].assign(val)
        return loss

    best_val_err = 100
    best_epoch = 1
    batches = x.shape[0] // batch_size
    for i in range(epochs):
        start = time.time()
        callback.on_epoch_begin(i)
        # x, y = shuffle(x, y)

        loss = 0
        for j in range(batches):
            loss += train_batch(x[j * batch_size:(j + 1) * batch_size],
                                y[j * batch_size:(j + 1) * batch_size])
            # print(j, loss / (j + 1))
        loss /= batches

        result = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
        val_err = (1 - result[2]) * 100
        if val_err <= best_val_err:
            best_val_err = val_err
            best_epoch = i + 1

            if save_path is not None:
                model.save_weights(save_path)

        duration = time.time() - start
        lr = model.optimizer._decayed_lr(tf.float32).numpy()
        print("Epoch {} of {} took {} s.".format(i + 1, epochs, duration))
        print("  LR:                         {}".format(lr))
        print("  training loss:              {}".format(loss))
        print("  validation loss:            {}".format(result[0]))
        print("  validation error rate:      {}%".format(val_err))
        print("  best epoch:                 {}".format(best_epoch))
        print("  best validation error rate: {}%".format(best_val_err))
