from smash.factory.net._layers import Conv2D
import tensorflow as tf
import numpy as np


# ==================
# Test configuration
# ==================

height, width, depth = 8, 12, 3
filter_shape = (2, 5)
filters = 4
atol = 1e-04
np.random.seed(21)  # random seed

# ==============================================
# Create input, weights and accumulated gradient
# ==============================================

# Inputs for tensor layer
x_tf = np.random.randn(1, height, width, depth).astype(np.float32)
w_tf = np.random.randn(*(filter_shape + (depth, filters))).astype(np.float32)
b_tf = np.random.randn(filters).astype(np.float32)
accum_grad_tf = np.random.randn(1, height, width, filters).astype(np.float32)

# Convert to valid shape taken by smash Net
x_sm = x_tf[0]
w_sm = w_tf.transpose(3, 2, 0, 1).reshape(filters, -1)
b_sm = b_tf[np.newaxis, ...]
accum_grad_sm = accum_grad_tf[0]

# ============
# Forward test
# ============

# Create conv2d layer
layer_tf = tf.keras.layers.Conv2D(
    filters,
    kernel_size=filter_shape,
    padding="same",
    strides=(1, 1),
)
layer_sm = Conv2D(
    filters,
    filter_shape,
    input_shape=x_sm.shape,
    kernel_initializer="zeros",
    bias_initializer="zeros",
)

# Set weight and bias
layer_tf.build(input_shape=(None,) + x_tf.shape[1:])
layer_tf.set_weights([w_tf, b_tf])

layer_sm._initialize(optimizer=None)
layer_sm.weight = w_sm
layer_sm.bias = b_sm

# Forward pass
x_tf = tf.convert_to_tensor(x_tf)
with tf.GradientTape() as tape:
    tape.watch(x_tf)
    y_tf = layer_tf(x_tf)

y_sm = layer_sm._forward_pass(x_sm)

print("Forward test..")
if np.allclose(y_tf, y_sm, atol=atol):
    print("=== pass")
else:
    print("xxx failed")
    print(y_sm)
    print("vs.......")
    print(y_tf.numpy())

# =============
# Gradient test
# =============
grad_tf = tape.gradient(y_tf, x_tf, output_gradients=tf.constant(accum_grad_tf))

layer_sm.trainable = False
grad_sm = layer_sm._backward_pass(accum_grad_sm)

print("Gradient test..")
if np.allclose(grad_tf, grad_sm, atol=atol):
    print("=== pass")
else:
    print("xxx failed")
    print(grad_sm)
    print("vs.......")
    print(grad_tf.numpy())
