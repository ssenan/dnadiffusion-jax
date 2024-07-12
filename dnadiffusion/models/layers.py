import math
from functools import partial
from typing import Any, Callable, Optional

import jax
from flax import linen as nn
from jax import lax, numpy as jnp

from dnadiffusion.utils.utils import l2_norm


class ResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: jax.Array):
        residual = x
        x = nn.Conv(self.features, (3, 3), padding=1)(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), padding=1, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        return x + residual


class EmbedFC(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: jax.Array):
        x = nn.Dense(self.features)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.features)(x)
        return x


class Residual(nn.Module):
    fn: Callable[..., Any]

    @nn.compact
    def __call__(self, x: jax.Array, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Upsample(nn.Module):
    dim: Optional[int] = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array):
        b, h, w, c = x.shape
        dim = self.dim if self.dim is not None else c
        x = jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")
        x = nn.Conv(dim, (3, 3), padding=1, dtype=self.dtype)(x)
        assert x.shape == (b, h * 2, w * 2, dim)
        return x


class Downsample(nn.Module):
    dim: Optional[int] = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array):
        b, h, w, c = x.shape
        dim = self.dim if self.dim is not None else c
        x = nn.Conv(dim, (4, 4), strides=(2, 2), padding=1, dtype=self.dtype)(x)
        assert x.shape == (b, h // 2, w // 2, dim)
        return x


class PreNorm(nn.Module):
    fn: Callable[..., Any]

    @nn.compact
    def __call__(self, x: jax.Array):
        x = nn.LayerNorm()(x)
        return self.fn(x)


class LearnedSinusoidalPosEmb(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array):
        x = jnp.reshape(x, (x.shape[0], 1))
        w = self.param("w", nn.initializers.normal(stddev=1), (self.dim,))
        w = jnp.reshape(w, (1, w.shape[0]))
        f = 2 * jnp.pi * x @ w
        fourier = jnp.concatenate([jnp.sin(f), jnp.cos(f)], axis=-1)
        fourier = jnp.concatenate([x, fourier], axis=-1)
        return fourier


class Block(nn.Module):
    dim: int
    groups: int = 8
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, scale_shift: jax.Array | None = None):
        x = nn.Conv(self.dim, (3, 3), padding=1, dtype=self.dtype)(x)
        x = nn.GroupNorm(self.groups, dtype=self.dtype)(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = nn.silu(x)
        return x


class ResNetBlock(nn.Module):
    dim: int
    groups: int = 8
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, time_emb: jax.Array):
        time_emb = nn.silu(time_emb)
        time_emb = nn.Dense(self.dim * 2, dtype=self.dtype)(time_emb)
        time_emb = time_emb[:, jnp.newaxis, jnp.newaxis, :]
        scale_shift = jnp.split(time_emb, 2, axis=-1)

        h = Block(self.dim, self.groups, dtype=self.dtype)(x, scale_shift)
        h = Block(self.dim, self.groups, dtype=self.dtype)(h)

        if x.shape[-1] != self.dim:
            x = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(x)

        return x + h


class LinearAttention(nn.Module):
    dim: int
    heads: int = 4
    dim_head: int = 32

    @nn.compact
    def __call__(self, x: jax.Array):
        b, h, w, c = x.shape
        hidden_dim = self.dim_head * self.heads

        qkv = nn.Conv(hidden_dim * 3, kernel_size=(1, 1), use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(b, -1, h * w, self.heads)
        k = k.reshape(b, -1, h * w, self.heads)
        v = v.reshape(b, -1, h * w, self.heads)

        q = nn.softmax(q, axis=-3)
        k = nn.softmax(k, axis=-2)
        q = q / jnp.sqrt(self.dim_head)
        v = v / (h * w)

        context = jnp.einsum("bdhn,behn->bden", k, v)
        out = jnp.einsum("bdeh,bdnh->benh", context, q)

        out = out.reshape(b, h, w, hidden_dim)
        out = nn.Conv(features=self.dim, kernel_size=(1, 1))(out)
        out = nn.LayerNorm()(out)

        return out


class Attention(nn.Module):
    dim: int
    heads: int = 4
    dim_head: int = 32
    scale: int = 10

    @nn.compact
    def __call__(self, x: jax.Array):
        hidden_dim = self.dim_head * self.heads
        b, h, w, c = x.shape
        qkv = nn.Conv(features=hidden_dim * 3, kernel_size=(1, 1), use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(b, -1, h * w, self.heads)
        k = k.reshape(b, -1, h * w, self.heads)
        v = v.reshape(b, -1, h * w, self.heads)

        q, k = map(l2_norm, (q, k))

        sim = jnp.einsum("bhdn,bhen->bden", k, v) * self.scale
        attn = jax.nn.softmax(sim, axis=-2)
        out = jnp.einsum("bedh,bndh->benh", attn, v)
        out = out.reshape(b, h, w, hidden_dim)
        out = nn.Conv(features=self.dim, kernel_size=(1, 1))(out)
        return out


def _query_chunk_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    key_chunk_size: int = 2048,
    precision: Any = lax.Precision.HIGHEST,
    dtype: jnp.dtype = jnp.float32,
):
    num_kv, num_heads, k_features = key.shape
    v_features = value.shape[-1]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features).astype(dtype)

    @partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query: jax.Array, key: jax.Array, value: jax.Array):
        attn_weights = jnp.einsum("qhd,khd->qhk", query, key, precision=precision).astype(dtype)
        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = jnp.einsum("vhf,qhv->qhf", value, exp_weights, precision=precision).astype(dtype)
        return (
            exp_values,
            exp_weights.sum(axis=-1),
            max_score.reshape((query.shape[0], num_heads)),
        )

    def chunk_scanner(chunk_idx: int):
        key_chunk = lax.dynamic_slice(key, (chunk_idx, 0, 0), slice_sizes=(key_chunk_size, num_heads, k_features))
        value_chunk = lax.dynamic_slice(
            value,
            (chunk_idx, 0, 0),
            slice_sizes=(key_chunk_size, num_heads, v_features),
        )
        return summarize_chunk(query, key_chunk, value_chunk)

    chunk_values, chunk_weights, chunk_max = lax.map(chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size))

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    return all_values / all_weights


def jax_efficient_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 2048,
    precision: Any = jax.lax.Precision.HIGHEST,
):
    num_q, num_heads, q_features = query.shape[-3:]
    num_kv = key.shape[-3]

    def chunk_scanner(chunk_idx: int, _):
        query_chunk = lax.dynamic_slice(
            query,
            tuple([0] * (query.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(query.shape[:-3]) + (min(query_chunk_size, num_q), num_heads, q_features),
        )
        return (
            chunk_idx + query_chunk_size,
            _query_chunk_attention2(chunk_idx, query_chunk, key, value, precision=precision),
        )

    _, res = lax.scan(chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size))
    return jnp.concatenate(res, axis=-3)


class MEAttention(nn.Module):
    dim: int = 200
    heads: int = 1
    dim_head: int = 64
    dtype: jnp.dtype = jnp.float32
    q_bucket_size: int = 1024
    k_bucket_size: int = 2048

    @nn.compact
    def __call__(self, x: jax.Array, emb: jax.Array):
        inner_dim = self.dim_head * self.heads
        q = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype)(x)
        k = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype)(emb)
        v = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype)(emb)

        q = q.reshape(q.shape[0], -1, q.shape[-1], self.heads)
        k = k.reshape(k.shape[0], -1, k.shape[-1], self.heads)
        v = v.reshape(v.shape[0], -1, v.shape[-1], self.heads)

        # out = jax_memory_efficient_attention(
        out = jax_efficient_attention(
            q,
            k,
            v,
            query_chunk_size=self.q_bucket_size,
            key_chunk_size=self.k_bucket_size,
        )

        out = out.reshape(*x.shape[:-1], -1)

        out = nn.Dense(self.dim, dtype=self.dtype)(out)
        return out
