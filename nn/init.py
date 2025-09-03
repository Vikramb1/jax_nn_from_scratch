import jax.numpy as jnp

def kaiming_init(key, shape, dtype):
    std = jnp.sqrt(2 / shape[0])
    return std * jnp.normal(key, shape, dtype)

def constant_int(key, shape, dtype, c=0.0):
    return c * jnp.ones(shape, dtype=dtype)

def xavier_init(key, shape, dtype):
    abs_bound = jnp.sqrt(6 / (shape[0] + shape[1]))
    return jnp.uniform(key, shape, dtype, minval=-abs_bound, maxval=abs_bound)