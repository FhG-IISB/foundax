from __future__ import annotations

from dataclasses import dataclass

from .model import BCAT


@dataclass(frozen=True)
class BCATConfig:
    n_layer: int
    dim_emb: int
    dim_ffn: int
    n_head: int
    norm_first: bool
    norm_type: str
    activation: str
    qk_norm: bool
    x_num: int
    max_output_dim: int
    patch_num: int
    patch_num_output: int
    conv_dim: int
    time_embed: str
    max_time_len: int
    max_data_len: int
    deep: bool


BCAT_CONFIGS = {
    "default": BCATConfig(
        n_layer=12,
        dim_emb=1024,
        dim_ffn=2752,
        n_head=8,
        norm_first=True,
        norm_type="rms",
        activation="swiglu",
        qk_norm=True,
        x_num=128,
        max_output_dim=4,
        patch_num=16,
        patch_num_output=16,
        conv_dim=32,
        time_embed="learnable",
        max_time_len=20,
        max_data_len=20,
        deep=False,
    ),
}


def bcat_default() -> BCAT:
    cfg = BCAT_CONFIGS["default"]
    return BCAT(
        n_layer=cfg.n_layer,
        dim_emb=cfg.dim_emb,
        dim_ffn=cfg.dim_ffn,
        n_head=cfg.n_head,
        norm_first=cfg.norm_first,
        norm_type=cfg.norm_type,
        activation=cfg.activation,
        qk_norm=cfg.qk_norm,
        x_num=cfg.x_num,
        max_output_dim=cfg.max_output_dim,
        patch_num=cfg.patch_num,
        patch_num_output=cfg.patch_num_output,
        conv_dim=cfg.conv_dim,
        time_embed=cfg.time_embed,
        max_time_len=cfg.max_time_len,
        max_data_len=cfg.max_data_len,
        deep=cfg.deep,
    )
