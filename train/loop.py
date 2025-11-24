from typing import Any, Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
from nn.utils import tree_global_norm, apply_updates

PyTree = Any

def train_step(
    model_apply: Callable[..., Tuple[jax.Array, Optional[PyTree]]],
    loss_fn: Callable[..., jax.Array],
    optimizer: Any,
    *,
    from_logits: bool = True,
) -> Callable[..., Tuple[PyTree, PyTree, PyTree, Dict[str, jax.Array]]]:
    def step(params, state, batch, opt_state, rng):
        x, y = batch
        
        def loss_fn_wrapper(params):
            logits, new_state = model_apply(
                params, x,
                states_list=state,
                training=True,
                rng=rng,

            )
            loss = loss_fn(logits, y, from_logits=from_logits)
            return loss, (logits, new_state)

        (loss_value, (logits, new_state)), grads = jax.value_and_grad(
            loss_fn_wrapper,
            has_aux=True
        )(params)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = apply_updates(params, updates)

        metrics = {
            'loss': loss_value,
            'grad_norm': tree_global_norm(grads)
        }
    
        return new_params, new_state, new_opt_state, metrics

    return step