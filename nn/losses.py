import jax.numpy as jnp
from nn.activations import Softmax

#convert to functions
def cross_entropy_loss(preds, targets, from_logits: bool = True, one_hot: bool = True):
    if from_logits:
        preds = Softmax(preds)
    if one_hot:
        targets = jnp.argmax(targets, axis=1)
    return -jnp.sum(targets * jnp.log(preds))

def mse_loss(preds, targets):
    return jnp.mean((preds - targets) ** 2)

def binary_cross_entropy_loss(preds, targets):
    return -jnp.sum(targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))
