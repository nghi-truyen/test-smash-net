from smash.factory.net._layers import Conv2d
import tensorflow as tf
import numpy as np


# ==================
# Test configuration
# ==================

height, width, depth = 4, 6, 3
filter_shape = (2, 2)
filters = 5
atol = 1e-06
# np.random.seed(5)  # random seed

# =============================================
# Create input, weight and accumulated gradient
# =============================================

# Inputs for tensor layer
x_tf = np.random.randn(1, height, width, depth).astype(np.float32)
w_tf = np.random.randn(1, filter_shape[0], filter_shape[0], depth, filters).astype(
    np.float32
)
accum_grad_tf = np.random.randn(1, height, width, filters).astype(np.float32)

# Convert to valid shape taken by smash Net
x_sm = x_tf[0]
w_sm = w_tf[0].transpose(3, 2, 0, 1).reshape(filters, -1)
accum_grad_sm = accum_grad_tf[0]

# ============
# Forward test
# ============

# Create conv2d layer
layer_tf = tf.keras.layers.Conv2D(
    filters=filters,
    kernel_size=filter_shape,
    padding="same",
    strides=(1, 1),
    use_bias=False,
)
layer_sm = Conv2d(filters, filter_shape, input_shape=x_sm.shape)

# Set weights
layer_tf.build(input_shape=(None,) + x_tf.shape[1:])
layer_tf.set_weights(w_tf)

layer_sm._initialize(optimizer=None)
layer_sm.weight = w_sm

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
