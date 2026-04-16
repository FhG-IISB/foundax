from __future__ import annotations

from dataclasses import dataclass

from .model import DPOTNet


@dataclass(frozen=True)
class DPOTConfig:
    img_size: int
    patch_size: int
    mixing_type: str
    in_channels: int
    out_channels: int
    in_timesteps: int
    out_timesteps: int
    n_blocks: int
    embed_dim: int
    out_layer_dim: int
    depth: int
    modes: int
    mlp_ratio: float
    n_cls: int
    normalize: bool
    act: str
    time_agg: str


DPOT_CONFIGS = {
    "Ti": DPOTConfig(
        img_size=128,
        patch_size=8,
        mixing_type="afno",
        in_channels=4,
        out_channels=4,
        in_timesteps=10,
        out_timesteps=1,
        n_blocks=4,
        embed_dim=512,
        out_layer_dim=32,
        depth=4,
        modes=32,
        mlp_ratio=1.0,
        n_cls=12,
        normalize=False,
        act="gelu",
        time_agg="exp_mlp",
    ),
    "S": DPOTConfig(
        img_size=128,
        patch_size=8,
        mixing_type="afno",
        in_channels=4,
        out_channels=4,
        in_timesteps=10,
        out_timesteps=1,
        n_blocks=8,
        embed_dim=1024,
        out_layer_dim=32,
        depth=6,
        modes=32,
        mlp_ratio=1.0,
        n_cls=12,
        normalize=False,
        act="gelu",
        time_agg="exp_mlp",
    ),
    "M": DPOTConfig(
        img_size=128,
        patch_size=8,
        mixing_type="afno",
        in_channels=4,
        out_channels=4,
        in_timesteps=10,
        out_timesteps=1,
        n_blocks=8,
        embed_dim=1024,
        out_layer_dim=32,
        depth=12,
        modes=32,
        mlp_ratio=4.0,
        n_cls=12,
        normalize=False,
        act="gelu",
        time_agg="exp_mlp",
    ),
    "L": DPOTConfig(
        img_size=128,
        patch_size=8,
        mixing_type="afno",
        in_channels=4,
        out_channels=4,
        in_timesteps=10,
        out_timesteps=1,
        n_blocks=16,
        embed_dim=1536,
        out_layer_dim=128,
        depth=24,
        modes=32,
        mlp_ratio=4.0,
        n_cls=12,
        normalize=False,
        act="gelu",
        time_agg="exp_mlp",
    ),
    "H": DPOTConfig(
        img_size=128,
        patch_size=8,
        mixing_type="afno",
        in_channels=4,
        out_channels=4,
        in_timesteps=10,
        out_timesteps=1,
        n_blocks=8,
        embed_dim=2048,
        out_layer_dim=128,
        depth=27,
        modes=32,
        mlp_ratio=4.0,
        n_cls=12,
        normalize=False,
        act="gelu",
        time_agg="exp_mlp",
    ),
}


def _make_dpot(name: str) -> DPOTNet:
    cfg = DPOT_CONFIGS[name]
    return DPOTNet(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        mixing_type=cfg.mixing_type,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        in_timesteps=cfg.in_timesteps,
        out_timesteps=cfg.out_timesteps,
        n_blocks=cfg.n_blocks,
        embed_dim=cfg.embed_dim,
        out_layer_dim=cfg.out_layer_dim,
        depth=cfg.depth,
        modes=cfg.modes,
        mlp_ratio=cfg.mlp_ratio,
        n_cls=cfg.n_cls,
        normalize=cfg.normalize,
        act=cfg.act,
        time_agg=cfg.time_agg,
    )


def dpot_ti() -> DPOTNet:
    return _make_dpot("Ti")


def dpot_s() -> DPOTNet:
    return _make_dpot("S")


def dpot_m() -> DPOTNet:
    return _make_dpot("M")


def dpot_l() -> DPOTNet:
    return _make_dpot("L")


def dpot_h() -> DPOTNet:
    return _make_dpot("H")
