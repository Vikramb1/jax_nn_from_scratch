import jax.numpy as jnp

class ReLU:
    def __call__(self, x):
        return jnp.maximum(0, x)
    
class LeakyReLU:
    alpha: float = 0.01

    def __call__(self, x):
        return jnp.maximum(self.alpha * x, x)
    
class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + jnp.exp(-x))
    
class Tanh:
    def __call__(self, x):
        return jnp.tanh(x)
    
class Softmax:
    def __call__(self, x):
        return jnp.exp(x) / jnp.sum(jnp.exp(x))