from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Protocol
import jax
import jax.numpy as jnp
from jax import random
from nn.layers import dense_apply

PyTree = Any

class InitFn(Protocol):
    def __call__(self, key: jax.Array, *args, **kwargs) -> Union[Dict[str, jax.Array], Tuple[Dict[str, jax.Array], PyTree]]:
        pass

class ApplyFn(Protocol):
    def __call__(self, params: Dict[str, jax.Array], x: jax.Array, *args, **kwargs) -> Union[jax.Array, Tuple[jax.Array, PyTree]]:
        pass

LayerSpec = Tuple[InitFn, ApplyFn, Dict[str, Any]]

def sequential_init(key: jax.Array, 
                    specs: Sequence[LayerSpec]
                    ) -> Tuple[List[Dict[str, jax.Array]], 
                               List[Optional[PyTree]]]:
    key_arr = jax.random.split(key, len(specs))
    params_list = []
    states_list = []

    for (init_fn, apply_fn, config_kwargs), subkey in zip(specs, key_arr):
        initializer = init_fn(subkey, **config_kwargs)
        if isinstance(initializer, tuple):
            params, state = initializer
            params_list.append(params)
            states_list.append(state)
        if isinstance(initializer, dict):
            params_list.append(params)
            states_list.append(None)
    return params_list, states_list


def sequential_apply(
    params_list: Sequence[Dict[str, jax.Array]],
    x: jax.Array,
    *,
    apply_fns: Sequence[ApplyFn],
    states_list: Optional[Sequence[Optional[PyTree]]] = None,
    training: bool = False,
    rng: Optional[jax.Array] = None,
    runtime_overrides: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[jax.Array, Optional[List[Optional[PyTree]]]]:
    curr_state = x
    new_states = []

    if rng is not None:
        rng_keys = random.split(rng, len(apply_fns))
    else:
        rng_keys = [None] * len(apply_fns)

    for i, (params, apply_fn, rng_key) in enumerate(zip(params_list, apply_fns, rng_keys)):
        kwargs = {'training': training}
        if rng_key is not None:
            kwargs['key'] = rng_key
        if runtime_overrides and i < len(runtime_overrides):
            kwargs.update(runtime_overrides[i])

        result = apply_fn(params, curr_state, **kwargs)

        if isinstance(result, tuple):
            curr_state, state = result
            new_states.append(state)
        else:
            curr_state = result
            new_states.append(None)

    return curr_state, new_states if any(s is not None for s in new_states) else None

def mlp_init(
    key: jax.Array,
    sizes: Sequence[int],
    *,
    dense_init_fn: InitFn,
    dense_init_kwargs: Optional[Dict[str, Any]] = None,
    # use_layernorm: bool = False,
    # layernorm_init_fn: Optional[InitFn] = None,
    # layernorm_init_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, jax.Array]], List[Optional[PyTree]], List[ApplyFn]]:
    params_list, states_list, apply_fns = [], [], []
    rng_keys = random.split(key, len(sizes)-1)
    init_kwargs = dense_init_kwargs or {}

    for i in range(len(sizes) - 1):
        in_dim, out_dim = sizes[i], sizes[i + 1]
        params_list.append(dense_init_fn(rng_keys[i], in_dim, out_dim, **init_kwargs))
        states_list.append(None)
        apply_fns.append(dense_apply)
    
    return params_list, states_list, apply_fns

def mlp_apply(
    params_list: Sequence[Dict[str, jax.Array]],
    x: jax.Array,
    *,
    apply_fns: Sequence[ApplyFn],
    activation: Callable[[jax.Array], jax.Array],
    dropout_rate: float = 0.0,
    dropout_apply_fn: Optional[Callable[..., jax.Array]] = None,
    training: bool = False,
    rng: Optional[jax.Array] = None,
    states_list: Optional[Sequence[Optional[PyTree]]] = None,
) -> Tuple[jax.Array, Optional[List[Optional[PyTree]]]]:
    expanded_params = []
    expanded_apply_fns = []
    expanded_states = []
    expanded_overrides = []
    
    for i in range(len(params_list) - 1):
        expanded_params.append(params_list[i])
        expanded_apply_fns.append(apply_fns[i])
        expanded_states.append(states_list[i] if states_list else None)
        expanded_overrides.append({})
        
        expanded_params.append(None)
        expanded_apply_fns.append(activation)
        expanded_states.append(None)
        expanded_overrides.append({})

        if dropout_rate > 0 and dropout_apply_fn:
            expanded_params.append(None)
            expanded_apply_fns.append(dropout_apply_fn)
            expanded_states.append(None)
            expanded_overrides.append({'rate': dropout_rate})

    expanded_params.append(params_list[-1])
    expanded_apply_fns.append(apply_fns[-1])
    expanded_states.append(states_list[-1] if states_list else None)
    expanded_overrides.append({})
    
    return sequential_apply(
        params_list=expanded_params,
        x=x,
        apply_fns=expanded_apply_fns,
        states_list=expanded_states,
        training=training,
        rng=rng,
        runtime_overrides=expanded_overrides
    )