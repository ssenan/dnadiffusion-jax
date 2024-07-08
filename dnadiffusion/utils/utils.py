from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state


def one_hot_encode(seq, alphabet, max_seq_len):
    """One-hot encode a sequence."""
    seq_len = len(seq)
    seq_array = jnp.zeros((max_seq_len, len(alphabet)))
    for i in range(seq_len):
        seq_array[i, alphabet.index(seq[i])] = 1
    return seq_array


def l2_norm(x):
    return x / jnp.sqrt(jnp.sum(x**2, axis=2, keepdims=True) + 1e-8)


def linear_beta_schedule(beta_start, beta_end, n_timesteps):
    betas = jnp.linspace(beta_start, beta_end, n_timesteps, dtype=jnp.float32)
    return betas


def diffusion_params(timesteps, beta_start, beta_end):
    betas = linear_beta_schedule(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    alphas_cumprod_prev = jnp.pad(alphas_cumprod[:-1], ((1, 0),), constant_values=1.0)
    sqrt_recip_alphas = jnp.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod) / (1.0 - alphas)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_1m_alphas_cumprod": sqrt_1m_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }


def get_init_state(rng, x_shape, t_shape, classes_shape, model, tx):
    x_init = jnp.ones(x_shape, dtype=jnp.float32)
    t_init = jnp.ones(t_shape, dtype=jnp.int32)
    class_init = jnp.ones(classes_shape, dtype=jnp.int32)

    variables = model.init(rng, x_init, t_init, class_init)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )

    return state


def get_abstract_state(x_shape, t_shape, classes_shape, model, tx, rng):
    init_state_fn = partial(get_init_state, x_shape, t_shape, classes_shape, model, tx)

    state_shape = jax.eval_shape(init_state_fn, rng)

    state_spec = nn.get_partition_spec(state_shape)

    return state_shape, state_spec


def create_initial_state(x_shape, t_shape, classes_shape, model, tx, checkpoint_manager, rng, mesh):
    init_state_fn = partial(get_init_state, x_shape, t_shape, classes_shape, model, tx)

    state = jax.jit(init_state_fn)(rng)
    pass


def batch_sharding(batch, data_sharding):
    def per_device_init_fn(index):
        return batch[index]

    global_input = jax.make_array_from_callback(batch.shape, data_sharding, per_device_init_fn)
    return global_input
