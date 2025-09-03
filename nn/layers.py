from typing import Callable, Optional, Tuple, Any, Dict
import jax
import jax.numpy as jnp
from jax import random, lax
from nn.init import kaiming_init, xavier_init, constant_init, zero_init
PyTree = Any

def dense_init(
        key: jax.Array,
        in_dim: int,
        out_dim: int,
        *,
        w_init=kaiming_init, # (key, shape, dtype) -> jax.Array
        b_init=zero_init,
        dtype=jnp.float32,
) -> Dict[str, jax.Array]:
    w_shape = (out_dim, in_dim)
    b_shape = (out_dim,)
    return {'W': w_init(key, w_shape, dtype), 'B': b_init(key, b_shape, dtype)}
    

def dense_apply(
    params: Dict[str, jax.Array],
    x: jax.Array,
) -> jax.Array:
    
    y = jnp.matmul(x, jnp.transpose(params['W'])) + params['B']
    return y

def dropout_apply(
        x: jax.Array,
        *,
        rate: float,
        key,
        training: bool = True
) -> jax.Array:
    zeroed_matrix = jnp.choice(key, [0, 1], shape=x.shape, p=[rate, 1-rate])
    if training:
        x = x * zeroed_matrix
    return x

