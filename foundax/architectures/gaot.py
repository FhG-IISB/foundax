"""GAOT -- Geometry-Aware Operator Transformer (NeurIPS 2025).

**Paper:** Gao et al., *"GAOT: Geometry-Aware Operator Transformer for
Arbitrary-Geometry PDE Problems"* (2025).
https://arxiv.org/abs/2505.18781

Architecture: MAGNO encoder → UViT transformer → MAGNO decoder.

This is a faithful JAX/Equinox port of the upstream
``camlab-ethz/GAOT`` PyTorch implementation. Forward-pass numerical
equivalence with the reference is verified by ``scripts/compare_gaot.py``.

.. warning::

    The upstream GAOT repository (https://github.com/camlab-ethz/GAOT)
    carries **no code license** (all rights reserved by default).
    No pretrained weights have been released as of 2026-06.

Usage::

    import numpy as np, jax, foundax as fx
    from foundax.architectures.gaot import compute_neighbors_csr

    # Build a 32×32 latent grid
    H = W = 32
    xs = np.linspace(0, 1, H)
    latent_coord = np.stack(np.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)

    model = fx.gaot.S(input_size=2, output_size=1)
    enc_nbrs = compute_neighbors_csr(x_coord, latent_coord, radius=0.033)
    dec_nbrs = compute_neighbors_csr(latent_coord, query_coord, radius=0.033)
    out = model(latent_coord, x_coord, pndata, query_coord, enc_nbrs, dec_nbrs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np


# ---------------------------------------------------------------------------
# Neighbor search (outside JIT — scipy KDTree, returns CSR + helper arrays)
# ---------------------------------------------------------------------------

def compute_neighbors_csr(
    data: np.ndarray,
    queries: np.ndarray,
    radius: float,
) -> dict:
    """Radius-based neighbor search in CSR format (matches upstream).

    Args:
        data: ``[N_d, D]`` source point coordinates.
        queries: ``[N_q, D]`` query point coordinates.
        radius: Neighbourhood search radius.

    Returns:
        Dict with:

        - ``neighbors_index``  : ``int32[E]`` indices into ``data``
        - ``neighbors_row_splits``: ``int32[N_q+1]`` CSR offsets
        - ``seg_ids``         : ``int32[E]`` query-id of every edge
          (= ``repeat_interleave(arange(N_q), counts)`` precomputed
          for JAX-friendly scatter ops)
        - ``counts``          : ``int32[N_q]`` neighbours per query
    """
    from scipy.spatial import cKDTree

    src = np.asarray(data, dtype=np.float64)
    qry = np.asarray(queries, dtype=np.float64)
    tree = cKDTree(src)
    raw = tree.query_ball_point(qry, r=radius, workers=-1)
    counts = np.array([len(n) for n in raw], dtype=np.int64)
    nbr_index = np.concatenate([np.asarray(n, dtype=np.int64) for n in raw]) \
        if counts.sum() > 0 else np.zeros((0,), dtype=np.int64)
    row_splits = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    seg_ids = np.repeat(np.arange(len(qry), dtype=np.int64), counts)
    return {
        "neighbors_index": jnp.asarray(nbr_index),
        "neighbors_row_splits": jnp.asarray(row_splits),
        "seg_ids": jnp.asarray(seg_ids),
        "counts": jnp.asarray(counts),
    }


# Back-compat alias for the older padded-dense API name.
compute_neighbors = compute_neighbors_csr


# ---------------------------------------------------------------------------
# CSR segment reductions (JAX equivalents of torch_scatter)
# ---------------------------------------------------------------------------

def _segment_sum(data, seg_ids, num_segments):
    return jax.ops.segment_sum(data, seg_ids, num_segments=num_segments)


def _segment_max(data, seg_ids, num_segments):
    return jax.ops.segment_max(data, seg_ids, num_segments=num_segments)


def _segment_mean(data, seg_ids, counts, num_segments):
    s = _segment_sum(data, seg_ids, num_segments)
    # Reshape counts for broadcasting against trailing feature dims.
    c = counts.astype(s.dtype)
    while c.ndim < s.ndim:
        c = c[..., None]
    return s / jnp.maximum(c, 1.0)


def _segment_softmax(scores, seg_ids, counts, num_segments):
    """Numerically stable per-segment softmax over a flat edge list."""
    max_vals = _segment_max(scores, seg_ids, num_segments)        # [N_q]
    shifted = scores - max_vals[seg_ids]                            # [E]
    exp = jnp.exp(shifted)
    denom = _segment_sum(exp, seg_ids, num_segments)              # [N_q]
    return exp / denom[seg_ids]


# ---------------------------------------------------------------------------
# Linear / channel MLP building blocks (Conv1d-kernel-1 == Linear)
# ---------------------------------------------------------------------------

class LinearChannelMLP(eqx.Module):
    """Stack of Linear layers with non-linearity between (but not after) layers.

    Matches upstream ``LinearChannelMLP``. Default non-linearity is GELU.
    """

    fcs: list
    n_layers: int = eqx.field(static=True)

    def __init__(self, layers: List[int], *, key):
        keys = jax.random.split(key, len(layers) - 1)
        self.fcs = [
            eqx.nn.Linear(layers[i], layers[i + 1], key=keys[i])
            for i in range(len(layers) - 1)
        ]
        self.n_layers = len(layers) - 1

    def __call__(self, x):
        def apply_one(v):
            for i, fc in enumerate(self.fcs):
                v = fc(v)
                if i < self.n_layers - 1:
                    v = jax.nn.gelu(v)
            return v
        for _ in range(x.ndim - 1):
            apply_one = jax.vmap(apply_one)
        return apply_one(x)


class ChannelMLP(eqx.Module):
    """1×1 conv MLP applied along the channel dim (= Linear, vmap'd over points).

    The upstream ``ChannelMLP`` uses ``nn.Conv1d(C_in, C_out, 1)`` on the
    channel-first layout. We store the same weight matrix and use Linear; for
    weight transfer, the Conv1d ``[out, in, 1]`` tensor is squeezed to ``[out, in]``.
    """

    fcs: list
    n_layers: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        n_layers: int = 2,
        *,
        key,
    ):
        out_channels = in_channels if out_channels is None else out_channels
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        keys = jax.random.split(key, n_layers)
        fcs = []
        for i in range(n_layers):
            if i == 0 and i == n_layers - 1:
                fcs.append(eqx.nn.Linear(in_channels, out_channels, key=keys[i]))
            elif i == 0:
                fcs.append(eqx.nn.Linear(in_channels, hidden_channels, key=keys[i]))
            elif i == n_layers - 1:
                fcs.append(eqx.nn.Linear(hidden_channels, out_channels, key=keys[i]))
            else:
                fcs.append(eqx.nn.Linear(hidden_channels, hidden_channels, key=keys[i]))
        self.fcs = fcs
        self.n_layers = n_layers

    def __call__(self, x):
        # x : [..., C]; we vmap the eqx.nn.Linear over leading dims.
        def apply_one(v):
            for i, fc in enumerate(self.fcs):
                v = fc(v)
                if i < self.n_layers - 1:
                    v = jax.nn.gelu(v)
            return v

        # Broadcast over all leading axes generically:
        for _ in range(x.ndim - 1):
            apply_one = jax.vmap(apply_one)
        return apply_one(x)


# ---------------------------------------------------------------------------
# AGNO
# ---------------------------------------------------------------------------

class AGNO(eqx.Module):
    """Attentional Graph Neural Operator (faithful port of upstream)."""

    channel_mlp: LinearChannelMLP
    query_proj: Optional[eqx.nn.Linear]
    key_proj: Optional[eqx.nn.Linear]

    transform_type: str = eqx.field(static=True)
    use_attn: bool = eqx.field(static=True)
    attention_type: str = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    scaling_factor: float = eqx.field(static=True)

    def __init__(
        self,
        channel_mlp_layers: List[int],
        transform_type: str = "linear",
        use_attn: bool = False,
        attention_type: str = "cosine",
        coord_dim: Optional[int] = None,
        *,
        key,
    ):
        if transform_type not in (
            "linear", "linear_kernelonly", "nonlinear", "nonlinear_kernelonly"
        ):
            raise ValueError(f"Invalid transform_type: {transform_type}")

        self.transform_type = transform_type
        self.use_attn = bool(use_attn)
        self.attention_type = attention_type
        self.coord_dim = coord_dim if coord_dim is not None else 0

        k1, k2, k3 = jax.random.split(key, 3)
        self.channel_mlp = LinearChannelMLP(channel_mlp_layers, key=k1)

        if self.use_attn and attention_type == "dot_product":
            attention_dim = 64
            self.query_proj = eqx.nn.Linear(coord_dim, attention_dim, key=k2)
            self.key_proj = eqx.nn.Linear(coord_dim, attention_dim, key=k3)
            self.scaling_factor = 1.0 / (attention_dim ** 0.5)
        else:
            self.query_proj = None
            self.key_proj = None
            self.scaling_factor = 0.0

    def __call__(
        self,
        y: jnp.ndarray,                  # [N_y, coord_dim] (source)
        neighbors: dict,                 # CSR with seg_ids + counts
        x: Optional[jnp.ndarray] = None,  # [N_x, coord_dim] (query)
        f_y: Optional[jnp.ndarray] = None,  # [N_y, C] or [B, N_y, C]
    ) -> jnp.ndarray:
        if x is None:
            x = y
        nbr_idx = neighbors["neighbors_index"]
        seg_ids = neighbors["seg_ids"]
        counts = neighbors["counts"]
        num_query = counts.shape[0]

        # Gather edge endpoints (source / target coords expanded to per-edge)
        rep_features = y[nbr_idx]              # [E, coord_dim]
        self_features = x[seg_ids]             # [E, coord_dim]

        batched = False
        in_features = None
        if f_y is not None:
            if f_y.ndim == 3:
                batched = True
                in_features = f_y[:, nbr_idx, :]     # [B, E, C_in]
            elif f_y.ndim == 2:
                in_features = f_y[nbr_idx]            # [E, C_in]
            else:
                raise ValueError(f"f_y has unexpected ndim: {f_y.ndim}")

        # Attention weights -----------------------------------------------------
        attention_weights = None
        if self.use_attn:
            qc = self_features[:, : self.coord_dim]
            kc = rep_features[:, : self.coord_dim]
            if self.attention_type == "dot_product":
                q = jax.vmap(self.query_proj)(qc)
                k = jax.vmap(self.key_proj)(kc)
                scores = jnp.sum(q * k, axis=-1) * self.scaling_factor
            else:  # cosine
                q_n = qc / (jnp.linalg.norm(qc, axis=-1, keepdims=True) + 1e-12)
                k_n = kc / (jnp.linalg.norm(kc, axis=-1, keepdims=True) + 1e-12)
                scores = jnp.sum(q_n * k_n, axis=-1)
            attention_weights = _segment_softmax(scores, seg_ids, counts, num_query)

        # Kernel MLP input (order: [rep, self] — y first, x second) -------------
        agg_features = jnp.concatenate([rep_features, self_features], axis=-1)
        if f_y is not None and self.transform_type in (
            "nonlinear", "nonlinear_kernelonly"
        ):
            if batched:
                agg_features = jnp.broadcast_to(
                    agg_features[None, ...],
                    (in_features.shape[0],) + agg_features.shape,
                )
            agg_features = jnp.concatenate([agg_features, in_features], axis=-1)

        # Apply kernel MLP element-wise
        rep = self.channel_mlp(agg_features)  # [E, C_out] or [B, E, C_out]

        # Multiply by f_y for non-kernel-only transforms
        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            rep = rep * in_features  # element-wise

        # Apply attention weights
        if self.use_attn:
            if batched:
                rep = rep * attention_weights[None, :, None]
            else:
                rep = rep * attention_weights[:, None]

        # Aggregate
        if self.use_attn:
            if batched:
                out = jax.vmap(
                    lambda d: _segment_sum(d, seg_ids, num_query)
                )(rep)
            else:
                out = _segment_sum(rep, seg_ids, num_query)
        else:
            if batched:
                out = jax.vmap(
                    lambda d: _segment_mean(d, seg_ids, counts, num_query)
                )(rep)
            else:
                out = _segment_mean(rep, seg_ids, counts, num_query)
        return out


# ---------------------------------------------------------------------------
# Geometric embedding
# ---------------------------------------------------------------------------

def node_pos_encode(x: jnp.ndarray, freq: int = 4) -> jnp.ndarray:
    """Sin/cos positional encoding of 2D coordinates (matches upstream)."""
    freqs = jnp.arange(1, freq + 1, dtype=x.dtype)
    phi = jnp.pi * (x + 1.0)
    out = freqs[None, :, None] * phi[:, None, :]     # [N, freq, D]
    out = jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=2)  # [N, freq, 2D]
    return out.reshape(out.shape[0], -1)


class _StatMLP(eqx.Module):
    """Linear(stat → 64) → ReLU → Linear(64 → out) → ReLU (upstream)."""
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, stat_dim, out_dim, *, key):
        k1, k2 = jax.random.split(key)
        self.l1 = eqx.nn.Linear(stat_dim, 64, key=k1)
        self.l2 = eqx.nn.Linear(64, out_dim, key=k2)

    def __call__(self, x):
        x = jax.nn.relu(self.l1(x))
        x = jax.nn.relu(self.l2(x))
        return x


class _PointNetMLP(eqx.Module):
    """Linear(D → 64) → ReLU → Linear(64 → 64) → ReLU."""
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, in_dim, *, key):
        k1, k2 = jax.random.split(key)
        self.l1 = eqx.nn.Linear(in_dim, 64, key=k1)
        self.l2 = eqx.nn.Linear(64, 64, key=k2)

    def __call__(self, x):
        x = jax.nn.relu(self.l1(x))
        x = jax.nn.relu(self.l2(x))
        return x


class _PointNetFC(eqx.Module):
    """Linear(64 → out) → ReLU."""
    l: eqx.nn.Linear

    def __init__(self, out_dim, *, key):
        self.l = eqx.nn.Linear(64, out_dim, key=key)

    def __call__(self, x):
        return jax.nn.relu(self.l(x))


class GeometricEmbedding(eqx.Module):
    """Per-query geometric features (statistical or pointnet)."""

    mlp: Optional[_StatMLP]
    pointnet_mlp: Optional[_PointNetMLP]
    fc: Optional[_PointNetFC]

    method: str = eqx.field(static=True)
    pooling: str = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        method: str = "statistical",
        pooling: str = "max",
        *,
        key,
    ):
        method = method.lower()
        pooling = pooling.lower()
        if pooling not in ("max", "mean"):
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.method = method
        self.pooling = pooling
        self.input_dim = input_dim
        self.output_dim = output_dim

        k1, k2, k3 = jax.random.split(key, 3)
        if method == "statistical":
            stat_dim = 3 + 2 * input_dim
            self.mlp = _StatMLP(stat_dim, output_dim, key=k1)
            self.pointnet_mlp = None
            self.fc = None
        elif method == "pointnet":
            self.mlp = None
            self.pointnet_mlp = _PointNetMLP(input_dim, key=k2)
            self.fc = _PointNetFC(output_dim, key=k3)
        else:
            raise ValueError(f"Unknown method: {method}")

    # -- statistical -------------------------------------------------------------
    def _statistical(self, input_geom, latent_queries, spatial_nbrs):
        D = latent_queries.shape[1]
        num_queries = latent_queries.shape[0]
        nbr_idx = spatial_nbrs["neighbors_index"]
        seg_ids = spatial_nbrs["seg_ids"]
        counts = spatial_nbrs["counts"]

        nbr_coords = input_geom[nbr_idx]                  # [E, D]
        query_coords_pe = latent_queries[seg_ids]         # [E, D]

        distances = jnp.linalg.norm(nbr_coords - query_coords_pe, axis=1)
        counts_f = counts.astype(jnp.float32)
        has_nbrs = counts_f > 0

        # D_avg = scatter_mean(distances)
        D_sum = _segment_sum(distances, seg_ids, num_queries)
        D_avg = D_sum / jnp.maximum(counts_f, 1.0)

        # D_var = E[X^2] - E[X]^2  (clamped >= 0)
        dist_sq = distances ** 2
        E_X2 = _segment_sum(dist_sq, seg_ids, num_queries) / jnp.maximum(counts_f, 1.0)
        D_var = jnp.clip(E_X2 - D_avg ** 2, min=0.0)

        # centroid offset
        nbr_centroid_sum = _segment_sum(nbr_coords, seg_ids, num_queries)
        nbr_centroid = nbr_centroid_sum / jnp.maximum(counts_f[:, None], 1.0)
        Delta = nbr_centroid - latent_queries

        # covariance + PCA eigenvalues (descending)
        centered = nbr_coords - nbr_centroid[seg_ids]
        cov_components = centered[:, :, None] * centered[:, None, :]   # [E, D, D]
        cov_sum = _segment_sum(cov_components, seg_ids, num_queries)   # [Q, D, D]
        cov_matrix = cov_sum / jnp.maximum(counts_f[:, None, None], 1.0)
        # avoid NaN for empty queries by substituting identity
        safe_cov = jnp.where(
            has_nbrs[:, None, None], cov_matrix, jnp.eye(D)[None, :, :]
        )
        eigenvalues = jnp.linalg.eigvalsh(safe_cov)          # ascending
        PCA = jnp.flip(eigenvalues, axis=1)                  # descending
        PCA = jnp.where(has_nbrs[:, None], PCA, 0.0)

        geo = jnp.concatenate(
            [counts_f[:, None], D_avg[:, None], D_var[:, None], Delta, PCA],
            axis=1,
        )  # [Q, 3 + 2D]
        geo = jnp.where(has_nbrs[:, None], geo, 0.0)

        # Mean/std normalisation across queries (matches upstream).
        # torch.std defaults to unbiased=True (Bessel's correction); jnp matches
        # numpy with ddof=0. Use ddof=1 here for parity.
        feat_mean = jnp.mean(geo, axis=0, keepdims=True)
        feat_std = jnp.std(geo, axis=0, keepdims=True, ddof=1)
        feat_std = jnp.where(feat_std < 1e-6, 1.0, feat_std)
        return (geo - feat_mean) / feat_std

    # -- pointnet ----------------------------------------------------------------
    def _pointnet(self, input_geom, latent_queries, spatial_nbrs):
        num_queries = latent_queries.shape[0]
        nbr_idx = spatial_nbrs["neighbors_index"]
        seg_ids = spatial_nbrs["seg_ids"]
        counts = spatial_nbrs["counts"]
        has_nbrs = counts > 0

        nbr_coords = input_geom[nbr_idx]              # [E, D]
        q_per_e = latent_queries[seg_ids]              # [E, D]
        centered = nbr_coords - q_per_e
        feat = jax.vmap(self.pointnet_mlp)(centered)   # [E, 64]

        if self.pooling == "max":
            very_neg = jnp.full((num_queries, feat.shape[-1]), -jnp.inf, dtype=feat.dtype)
            pooled = _segment_max(feat, seg_ids, num_queries)
            # segment_max returns -inf for empty segments; zero them out
            pooled = jnp.where(has_nbrs[:, None], pooled, 0.0)
            pooled = jnp.where(jnp.isfinite(pooled), pooled, 0.0)
        else:  # mean
            pooled = _segment_sum(feat, seg_ids, num_queries) / jnp.maximum(
                counts.astype(feat.dtype)[:, None], 1.0
            )

        out = jax.vmap(self.fc)(pooled)               # [Q, output_dim]
        out = jnp.where(has_nbrs[:, None], out, 0.0)
        return out

    def __call__(self, input_geom, latent_queries, spatial_nbrs):
        if self.method == "statistical":
            feats = self._statistical(input_geom, latent_queries, spatial_nbrs)
            return jax.vmap(self.mlp)(feats)
        return self._pointnet(input_geom, latent_queries, spatial_nbrs)


# ---------------------------------------------------------------------------
# MAGNO Encoder / Decoder
# ---------------------------------------------------------------------------

@dataclass
class MAGNOConfig:
    coord_dim: int = 2
    radius: float = 0.033
    hidden_size: int = 64
    mlp_layers: int = 3
    lifting_channels: int = 32
    scales: List[float] = field(default_factory=lambda: [1.0])
    use_scale_weights: bool = False
    use_attention: bool = True
    attention_type: str = "cosine"
    use_geoembed: bool = True
    embedding_method: str = "statistical"
    pooling: str = "max"
    transform_type: str = "linear"
    sampling_strategy: Optional[str] = None
    max_neighbors: Optional[int] = None
    sample_ratio: Optional[float] = None
    node_embedding: bool = False
    precompute_edges: bool = True   # always True in our JIT-friendly port


def _kernel_coord_dim(coord_dim: int, node_embedding: bool) -> int:
    return coord_dim * 4 * 2 if node_embedding else coord_dim


class MAGNOEncoder(eqx.Module):
    """MAGNO Encoder: physical mesh → latent grid."""

    agno: AGNO
    lifting: ChannelMLP
    geoembed: Optional[GeometricEmbedding]
    recovery: Optional[ChannelMLP]
    scale_weighting_l1: Optional[eqx.nn.Linear]
    scale_weighting_l2: Optional[eqx.nn.Linear]

    coord_dim: int = eqx.field(static=True)
    scales: Tuple[float, ...] = eqx.field(static=True)
    use_scale_weights: bool = eqx.field(static=True)
    use_geoembed: bool = eqx.field(static=True)
    node_embedding: bool = eqx.field(static=True)
    transform_type: str = eqx.field(static=True)

    def __init__(self, in_channels: int, out_channels: int, config: MAGNOConfig, *, key):
        self.coord_dim = config.coord_dim
        self.scales = tuple(config.scales)
        self.use_scale_weights = config.use_scale_weights
        self.use_geoembed = config.use_geoembed
        self.node_embedding = config.node_embedding
        self.transform_type = config.transform_type

        kdim = _kernel_coord_dim(self.coord_dim, self.node_embedding)
        kernel_input_dim = 2 * kdim
        if config.transform_type in ("nonlinear", "nonlinear_kernelonly"):
            kernel_input_dim += in_channels

        mlp_sizes = (
            [kernel_input_dim]
            + [config.hidden_size] * config.mlp_layers
            + [out_channels]
        )

        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.agno = AGNO(
            channel_mlp_layers=mlp_sizes,
            transform_type=config.transform_type,
            use_attn=config.use_attention,
            attention_type=config.attention_type,
            coord_dim=kdim,
            key=k1,
        )
        self.lifting = ChannelMLP(
            in_channels=in_channels,
            hidden_channels=config.hidden_size,
            out_channels=out_channels,
            n_layers=1,
            key=k2,
        )

        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=out_channels,
                method=config.embedding_method,
                pooling=config.pooling,
                key=k3,
            )
            self.recovery = ChannelMLP(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                n_layers=1,
                key=k4,
            )
        else:
            self.geoembed = None
            self.recovery = None

        if self.use_scale_weights:
            self.scale_weighting_l1 = eqx.nn.Linear(kdim, config.hidden_size // 4, key=k5)
            self.scale_weighting_l2 = eqx.nn.Linear(
                config.hidden_size // 4, len(self.scales), key=k6
            )
        else:
            self.scale_weighting_l1 = None
            self.scale_weighting_l2 = None

    def __call__(
        self,
        x_coord: jnp.ndarray,           # [N, D]
        pndata: jnp.ndarray,            # [B, N, C_in]
        latent_tokens_coord: jnp.ndarray,  # [L, D]
        encoder_nbrs: list,             # list[dict] per scale
    ) -> jnp.ndarray:                   # [B, L, C_out]
        batch_size = pndata.shape[0]
        # Lift features (per-batch, per-point linear)
        pndata = self.lifting(pndata)   # [B, N, C_out]

        if self.use_scale_weights:
            sw = self.scale_weighting_l1(latent_tokens_coord)
            sw = jax.nn.relu(sw)
            sw = self.scale_weighting_l2(sw)
            sw = jax.nn.softmax(sw, axis=-1)
        else:
            sw = None

        encoded_scales = []
        for s, nbrs in enumerate(encoder_nbrs):
            if self.node_embedding:
                y = node_pos_encode(x_coord)
                xq = node_pos_encode(latent_tokens_coord)
            else:
                y = x_coord
                xq = latent_tokens_coord

            enc = self.agno(y=y, neighbors=nbrs, x=xq, f_y=pndata)  # [B, L, C_out]

            if self.use_geoembed:
                ge = self.geoembed(
                    input_geom=x_coord,
                    latent_queries=latent_tokens_coord,
                    spatial_nbrs=nbrs,
                )  # [L, C_out]
                ge = jnp.broadcast_to(ge[None, ...], enc.shape)
                enc = jnp.concatenate([enc, ge], axis=-1)
                enc = self.recovery(enc)
            encoded_scales.append(enc)

        if len(encoded_scales) == 1:
            return encoded_scales[0]
        if self.use_scale_weights:
            stack = jnp.stack(encoded_scales, axis=0)         # [S, B, L, C]
            w = sw.T[:, None, :, None]                         # [S, 1, L, 1]
            return (stack * w).sum(axis=0)
        return jnp.mean(jnp.stack(encoded_scales, axis=0), axis=0)


class MAGNODecoder(eqx.Module):
    """MAGNO Decoder: latent grid → query mesh."""

    agno: AGNO
    projection: ChannelMLP
    geoembed: Optional[GeometricEmbedding]
    recovery: Optional[ChannelMLP]
    scale_weighting_l1: Optional[eqx.nn.Linear]
    scale_weighting_l2: Optional[eqx.nn.Linear]

    coord_dim: int = eqx.field(static=True)
    scales: Tuple[float, ...] = eqx.field(static=True)
    use_scale_weights: bool = eqx.field(static=True)
    use_geoembed: bool = eqx.field(static=True)
    node_embedding: bool = eqx.field(static=True)

    def __init__(self, in_channels: int, out_channels: int, config: MAGNOConfig, *, key):
        self.coord_dim = config.coord_dim
        self.scales = tuple(config.scales)
        self.use_scale_weights = config.use_scale_weights
        self.use_geoembed = config.use_geoembed
        self.node_embedding = config.node_embedding

        kdim = _kernel_coord_dim(self.coord_dim, self.node_embedding)
        kernel_input_dim = 2 * kdim
        if config.transform_type in ("nonlinear", "nonlinear_kernelonly"):
            kernel_input_dim += in_channels

        mlp_sizes = (
            [kernel_input_dim]
            + [config.hidden_size] * config.mlp_layers
            + [in_channels]
        )

        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.agno = AGNO(
            channel_mlp_layers=mlp_sizes,
            transform_type=config.transform_type,
            use_attn=config.use_attention,
            attention_type=config.attention_type,
            coord_dim=kdim,
            key=k1,
        )
        self.projection = ChannelMLP(
            in_channels=in_channels,
            hidden_channels=config.hidden_size,
            out_channels=out_channels,
            n_layers=1,
            key=k2,
        )
        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=in_channels,
                method=config.embedding_method,
                pooling=config.pooling,
                key=k3,
            )
            self.recovery = ChannelMLP(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                n_layers=1,
                key=k4,
            )
        else:
            self.geoembed = None
            self.recovery = None
        if self.use_scale_weights:
            self.scale_weighting_l1 = eqx.nn.Linear(kdim, config.hidden_size // 4, key=k5)
            self.scale_weighting_l2 = eqx.nn.Linear(
                config.hidden_size // 4, len(self.scales), key=k6
            )
        else:
            self.scale_weighting_l1 = None
            self.scale_weighting_l2 = None

    def __call__(
        self,
        latent_tokens_coord: jnp.ndarray,  # [L, D]
        rndata: jnp.ndarray,               # [B, L, C_in]
        query_coord: jnp.ndarray,          # [M, D]
        decoder_nbrs: list,                # list[dict] per scale
    ) -> jnp.ndarray:                       # [B, M, C_out]
        if self.use_scale_weights:
            sw = self.scale_weighting_l1(query_coord)
            sw = jax.nn.relu(sw)
            sw = self.scale_weighting_l2(sw)
            sw = jax.nn.softmax(sw, axis=-1)
        else:
            sw = None

        decoded_scales = []
        for s, nbrs in enumerate(decoder_nbrs):
            if self.node_embedding:
                y = node_pos_encode(latent_tokens_coord)
                xq = node_pos_encode(query_coord)
            else:
                y = latent_tokens_coord
                xq = query_coord
            dec = self.agno(y=y, neighbors=nbrs, x=xq, f_y=rndata)  # [B, M, C_in]
            if self.use_geoembed:
                ge = self.geoembed(
                    input_geom=latent_tokens_coord,
                    latent_queries=query_coord,
                    spatial_nbrs=nbrs,
                )
                ge = jnp.broadcast_to(ge[None, ...], dec.shape)
                dec = jnp.concatenate([dec, ge], axis=-1)
                dec = self.recovery(dec)
            decoded_scales.append(dec)

        if len(decoded_scales) == 1:
            decoded = decoded_scales[0]
        elif self.use_scale_weights:
            stack = jnp.stack(decoded_scales, axis=0)
            w = sw.T[:, None, :, None]
            decoded = (stack * w).sum(axis=0)
        else:
            decoded = jnp.mean(jnp.stack(decoded_scales, axis=0), axis=0)

        return self.projection(decoded)


# ---------------------------------------------------------------------------
# Transformer blocks (UViT-style)
# ---------------------------------------------------------------------------

class RMSNorm(eqx.Module):
    weight: jnp.ndarray
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = jnp.ones(dim)
        self.eps = eps

    def __call__(self, x):
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


class FFN(eqx.Module):
    """SwiGLU: ``w2(silu(w1(x)) * w3(x))`` (no bias)."""
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear

    def __init__(self, input_size: int, ffn_hidden_size: int, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.w1 = eqx.nn.Linear(input_size, ffn_hidden_size, use_bias=False, key=k1)
        self.w2 = eqx.nn.Linear(ffn_hidden_size, input_size, use_bias=False, key=k2)
        self.w3 = eqx.nn.Linear(input_size, ffn_hidden_size, use_bias=False, key=k3)

    def __call__(self, x):
        return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))


def _rope_apply(x: jnp.ndarray) -> jnp.ndarray:
    """Apply ``rotary_embedding_torch``-style RoPE over the last dim.

    Matches ``rotary_embedding_torch.RotaryEmbedding(dim=D).rotate_queries_or_keys(x)``
    invoked on tensor ``x[..., seq, D]`` with default position = arange(seq).
    """
    *lead, seq, D = x.shape
    half = D // 2
    freqs = 1.0 / (10000.0 ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    t = jnp.arange(seq, dtype=jnp.float32)
    angles = t[:, None] * freqs[None, :]                         # [seq, half]
    # The rotary_embedding_torch convention is repeat-then-interleave on the half axis
    sin = jnp.repeat(jnp.sin(angles), 2, axis=-1)                # [seq, D]
    cos = jnp.repeat(jnp.cos(angles), 2, axis=-1)                # [seq, D]

    def rotate_half(t):
        t1 = t[..., 0::2]
        t2 = t[..., 1::2]
        out = jnp.stack([-t2, t1], axis=-1)
        return out.reshape(t.shape)

    # Broadcast sin/cos over leading dims
    while sin.ndim < x.ndim:
        sin = sin[None, ...]
        cos = cos[None, ...]
    return (x * cos) + (rotate_half(x) * sin)


class GroupQueryAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear

    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    use_rope: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 8,
        num_kv_heads: int = 8,
        positional_embedding: str = "absolute",
        *,
        key,
    ):
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = positional_embedding == "rope"

        kv_hidden_size = self.head_dim * self.num_kv_heads
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(input_size, hidden_size, use_bias=False, key=k1)
        self.k_proj = eqx.nn.Linear(input_size, kv_hidden_size, use_bias=False, key=k2)
        self.v_proj = eqx.nn.Linear(input_size, kv_hidden_size, use_bias=False, key=k3)
        self.o_proj = eqx.nn.Linear(hidden_size, input_size, use_bias=False, key=k4)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, S, C]
        q = jax.vmap(jax.vmap(self.q_proj))(x)
        k = jax.vmap(jax.vmap(self.k_proj))(x)
        v = jax.vmap(jax.vmap(self.v_proj))(x)
        B, S, _ = x.shape
        q = q.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.num_kv_heads != self.num_heads:
            r = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, r, axis=1)
            v = jnp.repeat(v, r, axis=1)

        if self.use_rope:
            q = _rope_apply(q)
            k = _rope_apply(k)

        scale = self.head_dim ** -0.5
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)             # [B, H, S, Dh]
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return jax.vmap(jax.vmap(self.o_proj))(out)


class TransformerBlock(eqx.Module):
    attn: GroupQueryAttention
    ffn: FFN
    attn_norm: Optional[RMSNorm]
    ffn_norm: Optional[RMSNorm]
    skip_proj: Optional[eqx.nn.Linear]

    skip_connection: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_hidden_size: int,
        positional_embedding: str = "absolute",
        use_attn_norm: bool = True,
        use_ffn_norm: bool = True,
        norm_eps: float = 1e-6,
        skip_connection: bool = False,
        *,
        key,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.attn = GroupQueryAttention(
            input_size=input_size, hidden_size=hidden_size,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            positional_embedding=positional_embedding, key=k1,
        )
        self.ffn = FFN(input_size, ffn_hidden_size, key=k2)
        self.attn_norm = RMSNorm(input_size, eps=norm_eps) if use_attn_norm else None
        self.ffn_norm = RMSNorm(input_size, eps=norm_eps) if use_ffn_norm else None
        self.skip_connection = skip_connection
        self.skip_proj = (
            eqx.nn.Linear(2 * input_size, input_size, key=k3)
            if skip_connection else None
        )

    def __call__(self, x, skip=None):
        if self.skip_connection and skip is not None:
            x = jnp.concatenate([x, skip], axis=-1)
            x = jax.vmap(jax.vmap(self.skip_proj))(x)
        h = x if self.attn_norm is None else self.attn_norm(x)
        h = x + self.attn(h)
        h = h if self.ffn_norm is None else self.ffn_norm(h)
        out = h + jax.vmap(jax.vmap(self.ffn))(h)
        return out


@dataclass
class AttentionConfig:
    num_heads: int = 8
    num_kv_heads: int = 8
    use_conditional_norm: bool = False  # unsupported in this port (parity sets False)
    cond_norm_hidden_size: int = 4
    atten_dropout: float = 0.0


@dataclass
class TransformerConfig:
    patch_size: int = 8
    hidden_size: int = 256
    use_attn_norm: bool = True
    use_ffn_norm: bool = True
    norm_eps: float = 1e-6
    num_layers: int = 3
    positional_embedding: str = "absolute"
    use_long_range_skip: bool = True
    ffn_multiplier: int = 4
    attn_config: AttentionConfig = field(default_factory=AttentionConfig)


class _Transformer(eqx.Module):
    input_proj: Optional[eqx.nn.Linear]
    output_proj: Optional[eqx.nn.Linear]
    encoder_layers: list
    middle_layer: Optional[TransformerBlock]
    decoder_layers: list

    use_long_range_skip: bool = eqx.field(static=True)

    def __init__(self, input_size: int, output_size: int, config: TransformerConfig, *, key):
        self.use_long_range_skip = config.use_long_range_skip
        hidden_size = config.hidden_size
        ffn_hidden_size = hidden_size * config.ffn_multiplier
        ac = config.attn_config

        if input_size != hidden_size:
            k_in, key = jax.random.split(key)
            self.input_proj = eqx.nn.Linear(input_size, hidden_size, key=k_in)
            working_size = hidden_size
        else:
            self.input_proj = None
            working_size = input_size

        if working_size != output_size:
            k_out, key = jax.random.split(key)
            self.output_proj = eqx.nn.Linear(working_size, output_size, key=k_out)
        else:
            self.output_proj = None

        n_enc = config.num_layers // 2
        n_dec = config.num_layers // 2
        n_mid = 1 if (config.num_layers % 2 == 1) else 0

        n_total = n_enc + n_mid + n_dec
        sub_keys = jax.random.split(key, max(n_total, 1))
        idx = 0

        def _make_block(sk, skip):
            return TransformerBlock(
                input_size=working_size,
                hidden_size=hidden_size,
                num_heads=ac.num_heads,
                num_kv_heads=ac.num_kv_heads,
                ffn_hidden_size=ffn_hidden_size,
                positional_embedding=config.positional_embedding,
                use_attn_norm=config.use_attn_norm,
                use_ffn_norm=config.use_ffn_norm,
                norm_eps=config.norm_eps,
                skip_connection=skip,
                key=sk,
            )

        self.encoder_layers = [_make_block(sub_keys[idx + i], False) for i in range(n_enc)]
        idx += n_enc
        if n_mid:
            self.middle_layer = _make_block(sub_keys[idx], False)
            idx += 1
        else:
            self.middle_layer = None
        self.decoder_layers = [_make_block(sub_keys[idx + i], True) for i in range(n_dec)]

    def __call__(self, x):
        # x : [B, S, C]
        if self.input_proj is not None:
            x = jax.vmap(jax.vmap(self.input_proj))(x)
        skips = []
        for layer in self.encoder_layers:
            x = layer(x)
            skips.append(x)
        if self.middle_layer is not None:
            x = self.middle_layer(x)
        for layer in self.decoder_layers:
            skip = skips.pop() if self.use_long_range_skip else None
            x = layer(x, skip=skip)
        if self.output_proj is not None:
            x = jax.vmap(jax.vmap(self.output_proj))(x)
        return x


# ---------------------------------------------------------------------------
# GAOT model (top level)
# ---------------------------------------------------------------------------

def _compute_absolute_embeddings(positions: jnp.ndarray, embed_dim: int) -> jnp.ndarray:
    """Match upstream ``_compute_absolute_embeddings``."""
    num_pos_dims = positions.shape[1]
    dim_touse = embed_dim // (2 * num_pos_dims)
    freq_seq = jnp.arange(dim_touse, dtype=positions.dtype)
    inv_freq = 1.0 / (10000.0 ** (freq_seq / dim_touse))
    sinusoid_inp = positions[:, :, None] * inv_freq[None, None, :]
    pe = jnp.concatenate([jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)
    return pe.reshape(positions.shape[0], -1)


class GAOT(eqx.Module):
    """Top-level GAOT model: MAGNO encoder → UViT transformer → MAGNO decoder."""

    encoder: MAGNOEncoder
    decoder: MAGNODecoder
    patch_linear: eqx.nn.Linear
    processor: _Transformer
    positions: jnp.ndarray

    input_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    node_latent_size: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    H: int = eqx.field(static=True)
    W: int = eqx.field(static=True)
    D: Optional[int] = eqx.field(static=True)
    positional_embedding_name: str = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        output_size: int,
        magno_config: MAGNOConfig,
        transformer_config: TransformerConfig,
        latent_tokens_size: Tuple[int, ...],
        *,
        key,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.coord_dim = magno_config.coord_dim
        self.node_latent_size = magno_config.lifting_channels
        self.patch_size = transformer_config.patch_size
        self.positional_embedding_name = transformer_config.positional_embedding

        if self.coord_dim == 2:
            assert len(latent_tokens_size) == 2
            self.H, self.W = latent_tokens_size
            self.D = None
            patch_volume = self.patch_size * self.patch_size
            nph = self.H // self.patch_size
            npw = self.W // self.patch_size
            positions = jnp.stack(
                jnp.meshgrid(
                    jnp.arange(nph, dtype=jnp.float32),
                    jnp.arange(npw, dtype=jnp.float32),
                    indexing="ij",
                ),
                axis=-1,
            ).reshape(-1, 2)
        else:
            assert len(latent_tokens_size) == 3
            self.H, self.W, self.D = latent_tokens_size
            patch_volume = self.patch_size ** 3
            nph = self.H // self.patch_size
            npw = self.W // self.patch_size
            npd = self.D // self.patch_size
            positions = jnp.stack(
                jnp.meshgrid(
                    jnp.arange(nph, dtype=jnp.float32),
                    jnp.arange(npw, dtype=jnp.float32),
                    jnp.arange(npd, dtype=jnp.float32),
                    indexing="ij",
                ),
                axis=-1,
            ).reshape(-1, 3)
        self.positions = positions

        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.encoder = MAGNOEncoder(input_size, self.node_latent_size, magno_config, key=k1)
        self.decoder = MAGNODecoder(self.node_latent_size, output_size, magno_config, key=k2)
        self.patch_linear = eqx.nn.Linear(
            patch_volume * self.node_latent_size,
            patch_volume * self.node_latent_size,
            key=k3,
        )
        self.processor = _Transformer(
            input_size=self.node_latent_size * patch_volume,
            output_size=self.node_latent_size * patch_volume,
            config=transformer_config,
            key=k4,
        )

    # -- sub-stages --------------------------------------------------------------
    def encode(self, x_coord, pndata, latent_tokens_coord, encoder_nbrs):
        return self.encoder(
            x_coord=x_coord, pndata=pndata,
            latent_tokens_coord=latent_tokens_coord, encoder_nbrs=encoder_nbrs,
        )

    def process(self, rndata: jnp.ndarray) -> jnp.ndarray:
        # rndata: [B, L, C]
        B, n_regional, C = rndata.shape
        P = self.patch_size
        if self.coord_dim == 2:
            H, W = self.H, self.W
            assert n_regional == H * W, (n_regional, H * W)
            assert H % P == 0 and W % P == 0
            nph, npw = H // P, W // P
            rndata = rndata.reshape(B, H, W, C)
            rndata = rndata.reshape(B, nph, P, npw, P, C)
            rndata = rndata.transpose(0, 1, 3, 2, 4, 5).reshape(B, nph * npw, P * P * C)
        else:
            H, W, D = self.H, self.W, self.D
            assert n_regional == H * W * D
            assert H % P == 0 and W % P == 0 and D % P == 0
            nph, npw, npd = H // P, W // P, D // P
            rndata = rndata.reshape(B, H, W, D, C)
            rndata = rndata.reshape(B, nph, P, npw, P, npd, P, C)
            rndata = rndata.transpose(0, 1, 3, 5, 2, 4, 6, 7).reshape(
                B, nph * npw * npd, P * P * P * C
            )

        rndata = jax.vmap(jax.vmap(self.patch_linear))(rndata)

        if self.positional_embedding_name == "absolute":
            patch_volume = P ** self.coord_dim
            pos_emb = _compute_absolute_embeddings(
                self.positions, patch_volume * self.node_latent_size
            )
            # Truncate or pad to match if integer division left a remainder
            d_have = pos_emb.shape[-1]
            d_want = patch_volume * self.node_latent_size
            if d_have < d_want:
                pos_emb = jnp.pad(pos_emb, ((0, 0), (0, d_want - d_have)))
            elif d_have > d_want:
                pos_emb = pos_emb[:, :d_want]
            rndata = rndata + pos_emb[None, :, :]

        rndata = self.processor(rndata)

        # Unpatchify
        if self.coord_dim == 2:
            rndata = rndata.reshape(B, nph, npw, P, P, C)
            rndata = rndata.transpose(0, 1, 3, 2, 4, 5).reshape(B, H * W, C)
        else:
            rndata = rndata.reshape(B, nph, npw, npd, P, P, P, C)
            rndata = rndata.transpose(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, H * W * D, C)
        return rndata

    def decode(self, latent_tokens_coord, rndata, query_coord, decoder_nbrs):
        return self.decoder(
            latent_tokens_coord=latent_tokens_coord, rndata=rndata,
            query_coord=query_coord, decoder_nbrs=decoder_nbrs,
        )

    def __call__(
        self,
        latent_tokens_coord: jnp.ndarray,
        xcoord: jnp.ndarray,
        pndata: jnp.ndarray,
        query_coord: Optional[jnp.ndarray] = None,
        encoder_nbrs: Optional[list] = None,
        decoder_nbrs: Optional[list] = None,
    ) -> jnp.ndarray:
        """Forward pass.

        ``encoder_nbrs`` / ``decoder_nbrs`` must be lists of CSR-dict objects,
        one per scale (length == len(magno_config.scales)). For the default
        single-scale config they are length-1 lists.
        """
        rndata = self.encode(xcoord, pndata, latent_tokens_coord, encoder_nbrs)
        rndata = self.process(rndata)
        if query_coord is None:
            query_coord = xcoord
        return self.decode(latent_tokens_coord, rndata, query_coord, decoder_nbrs)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def gaot(
    input_size: int = 2,
    output_size: int = 1,
    magno_config: Optional[MAGNOConfig] = None,
    transformer_config: Optional[TransformerConfig] = None,
    latent_tokens_size: Tuple[int, ...] = (32, 32),
    *,
    key=None,
) -> GAOT:
    if key is None:
        key = jax.random.PRNGKey(0)
    if magno_config is None:
        magno_config = MAGNOConfig()
    if transformer_config is None:
        transformer_config = TransformerConfig()
    return GAOT(
        input_size=input_size,
        output_size=output_size,
        magno_config=magno_config,
        transformer_config=transformer_config,
        latent_tokens_size=latent_tokens_size,
        key=key,
    )


def _variant(input_size, output_size, lifting_channels, hidden_size,
             num_layers, patch_size, latent_tokens_size, key, **overrides):
    mc = MAGNOConfig(lifting_channels=lifting_channels,
                     **{k: v for k, v in overrides.items() if k in MAGNOConfig.__dataclass_fields__})
    tc = TransformerConfig(
        hidden_size=hidden_size, num_layers=num_layers, patch_size=patch_size,
        **{k: v for k, v in overrides.items() if k in TransformerConfig.__dataclass_fields__},
    )
    return gaot(input_size=input_size, output_size=output_size,
                magno_config=mc, transformer_config=tc,
                latent_tokens_size=latent_tokens_size, key=key)


def S(input_size=2, output_size=1, *, latent_tokens_size=(32, 32), key=None) -> GAOT:
    """GAOT-S (Small)."""
    return _variant(input_size, output_size, 32, 256, 3, 2, latent_tokens_size, key)


def M(input_size=2, output_size=1, *, latent_tokens_size=(32, 32), key=None) -> GAOT:
    """GAOT-M (Medium)."""
    return _variant(input_size, output_size, 64, 512, 6, 2, latent_tokens_size, key)


def L(input_size=2, output_size=1, *, latent_tokens_size=(32, 32), key=None) -> GAOT:
    """GAOT-L (Large)."""
    return _variant(input_size, output_size, 96, 768, 12, 2, latent_tokens_size, key)


s, m, l = S, M, L  # noqa: E741

__all__ = [
    "compute_neighbors_csr", "compute_neighbors",
    "MAGNOConfig", "TransformerConfig", "AttentionConfig",
    "AGNO", "GeometricEmbedding", "MAGNOEncoder", "MAGNODecoder",
    "RMSNorm", "FFN", "GroupQueryAttention", "TransformerBlock",
    "GAOT", "gaot", "S", "M", "L", "s", "m", "l",
]
