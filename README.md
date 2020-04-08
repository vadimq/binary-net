This is a reimplementation of Binarized Neural Network (BNN) from the paper [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](arxiv.org/abs/1602.02830). The code is organized similarly to the original [Theano repository](github.com/MatthieuCourbariaux/BinaryNet). Inference optimizations aren't included.

# Does it work in exactly the same way as the Theano version?

Conceptually, yes, but the results won't be identical.

There are slight differences in how `Adam` and `Conv2D` work in Theano and TensorFlow. These are enough for the results to quickly diverge when training for many steps and epochs. The accumulation of batch normalization statistics is also slightly different.

# MNIST

We have reproduced the final test error of **0.96%** using the same network. It took 5 hours 58 minutes on Tesla P4 GPU on Colab. We trained it once.
