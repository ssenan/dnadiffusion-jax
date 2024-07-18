import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


def q_sample(
    x: jax.Array,
    t: jax.Array,
    diffusion_params: dict,
    noise: jax.Array | None = None,
):
    sqrt_alphas_cumprod_t = diffusion_params["sqrt_alphas_cumprod"][t, None, None, None]
    sqrt_1m_alphas_cumprod_t = diffusion_params["sqrt_1m_alphas_cumprod"][t, None, None, None]

    return sqrt_alphas_cumprod_t * x + sqrt_1m_alphas_cumprod_t * noise


def p_loss(
    rng: jax.Array,
    state: TrainState,
    x: jax.Array,
    classes: jax.Array,
    diffusion_params: dict,
    p_uncond: float = 0.1,
):
    b, h, w, c = x.shape
    rng, t_rng = jax.random.split(rng)
    batch_t = jax.random.randint(t_rng, shape=(b,), minval=0, maxval=diffusion_params["timesteps"], dtype=jnp.int32)

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


def linear_beta_schedule(beta_start: float, beta_end: float, n_timesteps: int):
    betas = jnp.linspace(beta_start, beta_end, n_timesteps, dtype=jnp.float32)
    return betas


def diffusion_params(timesteps: int, beta_start: float, beta_end: float):
    betas = linear_beta_schedule(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    alphas_cumprod_prev = jnp.pad(alphas_cumprod[:-1], ((1, 0),), constant_values=1.0)
    sqrt_recip_alphas = jnp.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod) / (1.0 - alphas)

    return {
        "timesteps": timesteps,
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_1m_alphas_cumprod": sqrt_1m_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }
