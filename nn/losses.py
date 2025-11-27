import jax.numpy as jnp
from nn.activations import softmax

def cross_entropy_loss(preds, targets, from_logits: bool = True, one_hot: bool = True):
    if from_logits:
        preds = softmax(None, preds)
    
    if one_hot:
        # targets are already one-hot encoded
        log_probs = jnp.log(preds + 1e-8)  # Add epsilon for numerical stability
        return -jnp.mean(jnp.sum(targets * log_probs, axis=-1))
    else:
        # targets are class indices
        log_probs = jnp.log(preds + 1e-8)
        return -jnp.mean(log_probs[jnp.arange(len(targets)), targets])

def mse_loss(preds, targets):
    return jnp.mean((preds - targets) ** 2)

def binary_cross_entropy_loss(preds, targets):
    return -jnp.sum(targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))
