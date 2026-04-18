from __future__ import annotations

from dataclasses import dataclass


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
