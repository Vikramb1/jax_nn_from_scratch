import jax.numpy as jnp

def relu(params, x, **kwargs):
    return jnp.maximum(0, x)

def leaky_relu(params, x, alpha=0.01, **kwargs):
    return jnp.maximum(alpha * x, x)

def sigmoid(params, x, **kwargs):
    return 1 / (1 + jnp.exp(-x))

def tanh(params, x, **kwargs):
    return jnp.tanh(x)

def softmax(params, x, **kwargs):
    # numerically stable softmax (nan's in training original softmax)
    x_max = jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)