from dataclasses import dataclass, field
from typing import Literal, Optional


NormType = Literal["layer", "rms"]
TimeEmbedType = Literal["continuous", "learnable"]
EmbedderType = Literal["linear", "conv"]


@dataclass
class EmbedderConfig:
    type: EmbedderType = "conv"
    dim: int = 1024
    patch_num: int = 8
    patch_num_output: int = 16
    time_embed: TimeEmbedType = "learnable"
    max_time_len: int = 32
    conv_dim: int = 32
    deep: bool = False


@dataclass
class DataEncoderConfig:
    n_layer: int = 7
    dim_emb: int = 1024
    dim_ffn: int = 2048
    n_head: int = 8
    dropout: float = 0.0
    norm_first: bool = True
    positional_embedding: Optional[Literal["sinusoidal", "learnable"]] = None
    norm: NormType = "rms"


@dataclass
class DataDecoderConfig:
    n_layer: int = 12
    query_dim: int = 1
    self_attn: int = 0
    dim_emb: int = 1024
    dim_ffn: int = 2048
    n_head: int = 8
    dropout: float = 0.0
    norm_first: bool = True
    patch_num_output: int = 16
    time_embed: TimeEmbedType = "learnable"
    max_time_len: int = 32
    final_ln: bool = True
    norm: NormType = "rms"


@dataclass
class PROSE1to1Config:
    dim_emb: int = 1024
    dim_ffn: int = 2048
    n_head: int = 8
    dropout: float = 0.0
    norm_first: bool = True
    patch_num: int = 8
    patch_num_output: int = 16
    carry_last_frame: bool = False
    time_embed: TimeEmbedType = "learnable"
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    data_encoder: DataEncoderConfig = field(default_factory=DataEncoderConfig)
    data_decoder: DataDecoderConfig = field(default_factory=DataDecoderConfig)


def prose_fd_1to1_default_config() -> PROSE1to1Config:
    cfg = PROSE1to1Config()
    cfg.embedder = EmbedderConfig(
        type="conv",
        dim=cfg.dim_emb,
        patch_num=cfg.patch_num,
        patch_num_output=cfg.patch_num_output,
        time_embed=cfg.time_embed,
        conv_dim=32,
        deep=False,
    )
    cfg.data_encoder = DataEncoderConfig(
        n_layer=7,
        dim_emb=cfg.dim_emb,
        dim_ffn=cfg.dim_ffn,
        n_head=cfg.n_head,
        dropout=cfg.dropout,
        norm_first=cfg.norm_first,
        positional_embedding=None,
        norm="rms",
    )
    cfg.data_decoder = DataDecoderConfig(
        n_layer=12,
        query_dim=1,
        self_attn=0,
        dim_emb=cfg.dim_emb,
        dim_ffn=cfg.dim_ffn,
        n_head=cfg.n_head,
        dropout=cfg.dropout,
        norm_first=cfg.norm_first,
        patch_num_output=cfg.patch_num_output,
        time_embed=cfg.time_embed,
        final_ln=True,
        norm="rms",
    )
    return cfg
