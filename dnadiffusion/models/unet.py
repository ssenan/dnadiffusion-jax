from typing import Optional

import jax
from flax import linen as nn
from jax import numpy as jnp

from dnadiffusion.models.layers import (
    Attention,
    Downsample,
    LearnedSinusoidalPosEmb,
    LinearAttention,
    MEAttention,
    PreNorm,
    Residual,
    ResNetBlock,
    Upsample,
)


class UNet(nn.Module):
    dim: int
    init_dim: Optional[int] = None
    dim_mults: tuple[int, ...] = (1, 2, 4)
    channels: int = 1
    resnet_block_groups: int = 4
    learned_sinusoidal_dim: int = 18
    num_classes: int = 10
    class_embed_dim: int = 3
    output_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, time: jax.Array, classes: jax.Array):
        init_dim = self.dim if self.init_dim is None else self.init_dim
        x = nn.Conv(init_dim, (7, 7), padding=3)(x)
        r = x

        time_dim = init_dim * 4

        time_emb = nn.Sequential(
            [
                LearnedSinusoidalPosEmb(self.learned_sinusoidal_dim),
                nn.Dense(time_dim),
                nn.gelu,
                nn.Dense(time_dim),
            ]
        )(time)

        if self.num_classes is not None:
            label_embed = nn.Embed(self.num_classes, time_dim)(classes)

        # Time embedding
        if classes is not None:
            time_emb = time_emb + label_embed

        # Layers
        num_resolutions = len(self.dim_mults)
        hs = []
        # Downsampling
        for i in range(num_resolutions):
            is_last = i >= (num_resolutions - 1)
            dim_in = x.shape[-1]

            x = ResNetBlock(dim_in)(x, time_emb)
            hs.append(x)

            x = ResNetBlock(dim_in)(x, time_emb)
            x = Residual(PreNorm(LinearAttention(dim_in)))(x)
            hs.append(x)
            x = (
                Downsample(self.dim * self.dim_mults[i])(x)
                if not is_last
                else nn.Conv(self.dim * self.dim_mults[i], (3, 3), padding=1)(x)
            )
        # Middle
        mid_dim = self.dim * self.dim_mults[-1]
        x = ResNetBlock(mid_dim)(x, time_emb)
        x = Residual(PreNorm(Attention(mid_dim)))(x)
        x = ResNetBlock(mid_dim)(x, time_emb)

        # Upsampling
        for i in reversed(range(num_resolutions)):
            is_last = i <= 0

            dim_in = self.dim * self.dim_mults[i]
            dim_out = self.dim * self.dim_mults[i - 1] if i > 0 else self.init_dim
            x = jnp.concatenate((x, hs.pop()), axis=-1)
            x = ResNetBlock(dim_in)(x, time_emb)

            x = jnp.concatenate((x, hs.pop()), axis=-1)
            x = ResNetBlock(dim_in)(x, time_emb)

            x = Residual(PreNorm(LinearAttention(dim_in)))(x)

            x = Upsample(dim_out)(x) if not is_last else nn.Conv(dim_in, (3, 3), padding=((1, 1), (1, 1)))(x)

        # Output
        x = jnp.concatenate((x, r), axis=-1)
        x = ResNetBlock(self.dim)(x, time_emb)
        x = nn.Conv(1, (1, 1), 1, dtype=self.dtype)(x)

        # Reshaping results
        x_reshaped = jnp.reshape(x, (-1, 4, 200))
        t_cross_reshaped = jnp.reshape(time_emb, (-1, 4, 200))

        cross_attn_out = nn.LayerNorm()(x_reshaped.reshape(-1, 800)).reshape(-1, 4, 200)
        cross_attn_out = MEAttention(
            dim=200,
            dim_head=64,
            heads=1,
            q_bucket_size=1024,
            k_bucket_size=2048,
        )(cross_attn_out, t_cross_reshaped)
        crossattn_out = jnp.reshape(cross_attn_out, (-1, 4, 200, 1))

        x += crossattn_out

        if self.output_attention:
            return x, crossattn_out
        return x
