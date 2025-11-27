from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, Callable
import jax
import jax.numpy as jnp
from jax import random, tree_leaves, tree_map

PyTree = Any

def apply_updates(params: PyTree, updates: PyTree) -> PyTree:
    return tree_map(lambda l, r: l + r, params, updates)

def tree_global_norm(tree: PyTree) -> jax.Array:
    leaves = tree_leaves(tree)
    global_l2_squared = jnp.sum(jnp.array([jnp.vdot(x, x) for x in leaves]))
    global_norm = jnp.sqrt(global_l2_squared)
    return global_norm

def count_parameters(params: PyTree) -> int:
    leaves = tree_leaves(params)
    return sum(jnp.size(leaf) for leaf in leaves)

def split_key(key: jax.Array, num: int) -> List[jax.Array]:
    return list(jax.random.split(key, num))
