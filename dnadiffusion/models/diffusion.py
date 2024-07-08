import jax
import jax.numpy as jnp
import optax


def q_sample(x, t, diffusion_params, noise=None):
    sqrt_alphas_cumprod_t = diffusion_params["sqrt_alphas_cumprod"][t, None, None, None]
    sqrt_1m_alphas_cumprod_t = diffusion_params["sqrt_1m_alphas_cumprod"][t, None, None, None]

    return sqrt_alphas_cumprod_t * x + sqrt_1m_alphas_cumprod_t * noise


def p_loss(rng, state, x, classes, timesteps, diffusion_params, p_uncond=0.1):
    b, h, w, c = x.shape
    rng, t_rng = jax.random.split(rng)
    batch_t = jax.random.randint(t_rng, shape=(b,), minval=0, maxval=timesteps, dtype=jnp.int32)

    # Creating noisy sample
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, shape=x.shape)
    x_noisy = q_sample(x, batch_t, diffusion_params, noise)

    # Context mask
    rng, context_rng = jax.random.split(rng)
    context_mask = jax.random.bernoulli(context_rng, p=(1 - p_uncond), shape=(classes.shape[0],))

    # Mask for unconditional guidance
    classes = classes * context_mask

    pred = state.apply_fn({"params": state.params}, x_noisy, batch_t, classes)
    loss = jnp.mean(optax.huber_loss(pred, noise))
    return loss
