import jax
from flax.training.train_state import TrainState
from google.cloud import storage
from jax import numpy as jnp


def create_sample(
    state: TrainState,
    rng: jax.Array,
    cell_types: list,
    number_of_samples: int,
    sample_bs: int,
    sequence_length: int,
    group_number: int,
    cond_weight_to_metric: float,
    diffusion_params: dict | None = None,
):
    @jax.jit
    def sample_batch(rng):
        if group_number is not None:
            sampled = jnp.full((sample_bs,), group_number)
        else:
            rng, cell_rng = jax.random.split(rng)
            sampled = jax.random.choice(cell_rng, jnp.array(cell_types), shape=(sample_bs,))

        classes = sampled.astype(jnp.float32)
        rng, sample_rng = jax.random.split(rng)

        sampled_images = sample_loop_fn(
            state,
            sample_rng,
            classes,
            (sample_bs, 4, 200, 1),
            cond_weight_to_metric,
            diffusion_params,
        )
        sequences = jnp.argmax(sampled_images.reshape(sample_bs, 4, 200), axis=1)
        return sequences, rng

    @jax.jit
    def body_fn(i, carry):
        sequences, rng = carry
        new_sequences, new_rng = sample_batch(rng)
        start_idx = i * sample_bs
        sequences = jax.lax.dynamic_update_slice(sequences, new_sequences, (start_idx, 0))
        return sequences, new_rng
        # return sequences.at[i].set(new_sequences), new_rng

    # initial_sequences = jnp.zeros((number_of_samples, sample_bs, 200), dtype=jnp.int32)
    num_batches = number_of_samples // sample_bs
    initial_sequences = jnp.zeros((number_of_samples, 200), dtype=jnp.int32)
    # final_sequences, _ = jax.lax.fori_loop(
    #     0, number_of_samples, body_fn, (initial_sequences, rng)
    # )
    final_sequences, _ = jax.lax.fori_loop(0, num_batches, body_fn, (initial_sequences, rng))

    return final_sequences


def sample_loop_fn(state, rng, classes, shape, cond_weight, diffusion_params):
    def sample_step_fn(x, t, t_index, sample_rng):
        batch_size = x.shape[0]
        sample_rng, noise_rng = jax.random.split(sample_rng)

        # Double for guidance
        t_double = jnp.repeat(t, 2)
        x_double = jnp.tile(x, (2, 1, 1, 1))

        betas_t = diffusion_params["betas"][t_double][:, None, None, None]
        sqrt_one_minus_alphas_t = diffusion_params["sqrt_1m_alphas_cumprod"][t_double][:, None, None, None]
        sqrt_recip_alphas_t = diffusion_params["sqrt_recip_alphas"][t_double][:, None, None, None]

        # CFG
        classes_masked = (classes * context_mask).astype(jnp.int32)
        preds = state.apply_fn({"params": state.params}, x_double, t_double, classes_masked)
        eps1 = (1 + cond_weight) * preds[:batch_size]
        eps2 = cond_weight * preds[batch_size:]
        x_t = eps1 - eps2

        model_mean = sqrt_recip_alphas_t[:batch_size] * (
            x - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_t[:batch_size]
        )

        def compute_final_mean(model_mean, _):
            return model_mean

        def compute_noisy_mean(model_mean, t):
            posterior_variance_t = diffusion_params["posterior_variance"][t][:, None, None, None]
            noise = jax.random.normal(noise_rng, x.shape)
            return model_mean + jnp.sqrt(posterior_variance_t) * noise

        return jax.lax.cond(
            t_index == 0,
            compute_final_mean,
            compute_noisy_mean,
            model_mean,
            t,
        ), sample_rng

    batch_size = shape[0]
    rng, noise_rng = jax.random.split(rng)
    timesteps = diffusion_params["timesteps"]

    # Start from pure noise
    img = jax.random.normal(noise_rng, shape)

    # Guided sampling
    n_sample = classes.shape[0]
    context_mask = jnp.ones(classes.shape, dtype=jnp.float32)

    # Making 0 index unconditional, doubling the batch size
    classes = jnp.repeat(classes, 2, axis=0)
    context_mask = jnp.repeat(context_mask, 2, axis=0)
    context_mask = context_mask.at[n_sample:].set(0.0)

    def body_fn(i, carry):
        img, sample_rng = carry
        t = jnp.full((batch_size,), i, dtype=jnp.int32)
        new_img, new_sample_rng = sample_step_fn(img, t, i, sample_rng)
        return new_img, new_sample_rng

    # Sampling
    final_img, _ = jax.lax.fori_loop(
        0,
        timesteps,
        lambda i, carry: body_fn(timesteps - 1 - i, carry),
        (img, rng),
    )

    return final_img


def write_gcs(bucket_name: str, file_name: str, sequences: jax.Array) -> None:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    with blob.open("w") as f:
        f.write("\n".join(sequences))
