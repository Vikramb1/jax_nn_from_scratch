from typing import Callable, Optional, Tuple, Any
import jax
import jax.numpy as jnp
from jax import random, lax
from nn.init import kaiming_init, xavier_init, constant_init
PyTree = Any

def dense_init(
        key: jax.Array,
        in_dim: int,
        out_dim:int,
        *,
        w_init=kaiming_init, # (key, shape, dtype) -> jax.Array
        b_init: bool = True,
        dtype=jnp.float32,
) -> Dict[str, jax.Array]:

    

def dense_apply(
    params: Dict[str, jax.Array],
    x: jax.Array,
) -> jax.Array:
    
