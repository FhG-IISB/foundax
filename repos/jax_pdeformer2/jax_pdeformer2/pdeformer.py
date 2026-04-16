"""PDEformer model implemented in JAX/Flax."""

from typing import Dict, Optional
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .graphormer import GraphormerEncoder
from .function_encoder import Conv2dFuncEncoderV3
from .inr_with_hypernet import PolyINRWithHypernet
from .basic_block import MLP

# Constants
SPACE_DIM = 3


class PDEEncoder(nn.Module):
    """
    PDEEncoder encodes the input graph and function into a fixed-size representation.
    It consists of a GraphormerEncoder, a scalar encoder, and a function encoder.
    """

    # Graphormer config
    num_node_type: int = 128
    num_in_degree: int = 32
    num_out_degree: int = 32
    num_spatial: int = 16
    num_encoder_layers: int = 12
    embed_dim: int = 768
    ffn_embed_dim: int = 1536
    num_heads: int = 32
    pre_layernorm: bool = True

    # Scalar encoder config
    scalar_dim_hidden: int = 256
    scalar_num_layers: int = 3

    # Function encoder config
    func_enc_type: str = "cnn2dv3"
    func_enc_num_branches: int = 4
    func_enc_resolution: int = 128
    func_enc_input_txyz: bool = False
    func_enc_keep_nchw: bool = True

    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        node_type,
        node_scalar,
        node_function,
        in_degree,
        out_degree,
        attn_bias,
        spatial_pos,
        deterministic=True,
    ):
        """
        Args:
            node_type: [n_graph, n_node, 1]
            node_scalar: [n_graph, num_scalar, 1]
            node_function: [n_graph, num_function, num_points_function, 5]
            in_degree: [n_graph, n_node]
            out_degree: [n_graph, n_node]
            attn_bias: [n_graph, n_node, n_node]
            spatial_pos: [n_graph, n_node, n_node]
        Returns:
            [n_node, n_graph, embed_dim]
        """
        # Scalar encoder
        scalar_encoder = MLP(
            dim_out=self.embed_dim,
            dim_hidden=self.scalar_dim_hidden,
            num_layers=self.scalar_num_layers,
            dtype=self.dtype,
            name="scalar_encoder",
        )
        node_scalar_feature = scalar_encoder(
            node_scalar
        )  # [n_graph, num_scalar, embed_dim]

        # Function encoder
        n_graph, num_function, num_points_function, _ = node_function.shape
        node_function_flat = node_function.reshape(
            n_graph * num_function, num_points_function, 1 + SPACE_DIM + 1
        )

        # Create function encoder
        function_encoder = Conv2dFuncEncoderV3(
            in_dim=1 + SPACE_DIM + 1,
            out_dim=self.embed_dim,
            resolution=self.func_enc_resolution,
            input_txyz=self.func_enc_input_txyz,
            keep_nchw=self.func_enc_keep_nchw,
            dtype=self.dtype,
            name="function_encoder",
        )

        node_function_feature = function_encoder(node_function_flat)

        # Reshape function features
        # Output shape from CNN: [n_graph*num_function, out_dim, 2, 2] (NCHW, keep_nchw=True)
        # Flatten and reshape: [n_graph, num_function*num_branches, embed_dim]
        node_function_feature = node_function_feature.reshape(
            n_graph, -1, self.embed_dim
        )

        # Concatenate scalar and function features
        node_input_feature = jnp.concatenate(
            [node_scalar_feature, node_function_feature], axis=1
        )  # [n_graph, num_scalar+num_function*num_branches, embed_dim]

        # Graphormer
        graphormer = GraphormerEncoder(
            num_node_type=self.num_node_type,
            num_in_degree=self.num_in_degree,
            num_out_degree=self.num_out_degree,
            num_spatial=self.num_spatial,
            num_encoder_layers=self.num_encoder_layers,
            embed_dim=self.embed_dim,
            ffn_embed_dim=self.ffn_embed_dim,
            num_heads=self.num_heads,
            pre_layernorm=self.pre_layernorm,
            dtype=self.dtype,
            name="graphormer",
        )

        out = graphormer(
            node_type,
            node_input_feature,
            in_degree,
            out_degree,
            attn_bias,
            spatial_pos,
            deterministic=deterministic,
        )  # [n_node, n_graph, embed_dim]

        return out


class PDEformer(nn.Module):
    """
    PDEformer consists of a PDEEncoder and an INR (with hypernet).
    The PDEEncoder encodes the PDE into a fixed-size representation, and the INR
    represents the solution of the PDE at each point.
    """

    # Graphormer config
    num_node_type: int = 128
    num_in_degree: int = 32
    num_out_degree: int = 32
    num_spatial: int = 16
    num_encoder_layers: int = 12
    embed_dim: int = 768
    ffn_embed_dim: int = 1536
    num_heads: int = 32
    pre_layernorm: bool = True

    # Scalar encoder config
    scalar_dim_hidden: int = 256
    scalar_num_layers: int = 3

    # Function encoder config
    func_enc_type: str = "cnn2dv3"
    func_enc_num_branches: int = 4
    func_enc_resolution: int = 128
    func_enc_input_txyz: bool = False
    func_enc_keep_nchw: bool = True

    # INR config
    inr_dim_hidden: int = 768
    inr_num_layers: int = 12
    enable_affine: bool = False
    enable_shift: bool = True
    enable_scale: bool = True
    activation_fn: str = "sin"
    affine_act_fn: str = "identity"

    # Hypernet config
    hyper_dim_hidden: int = 512
    hyper_num_layers: int = 2
    share_hypernet: bool = False

    # Multi-INR config
    multi_inr: bool = False
    separate_latent: bool = False

    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.n_inr_nodes = self.inr_num_layers - 1

    @nn.compact
    def __call__(
        self,
        node_type,
        node_scalar,
        node_function,
        in_degree,
        out_degree,
        attn_bias,
        spatial_pos,
        coordinate,
        deterministic=True,
    ):
        """
        Args:
            node_type: [n_graph, n_node, 1]
            node_scalar: [n_graph, num_scalar, 1]
            node_function: [n_graph, num_function, num_points_function, 5]
            in_degree: [n_graph, n_node]
            out_degree: [n_graph, n_node]
            attn_bias: [n_graph, n_node, n_node]
            spatial_pos: [n_graph, n_node, n_node]
            coordinate: [n_graph, num_points, 4]
        Returns:
            [n_graph, num_points, dim_out]
        """
        # PDE Encoder
        pde_encoder = PDEEncoder(
            num_node_type=self.num_node_type,
            num_in_degree=self.num_in_degree,
            num_out_degree=self.num_out_degree,
            num_spatial=self.num_spatial,
            num_encoder_layers=self.num_encoder_layers,
            embed_dim=self.embed_dim,
            ffn_embed_dim=self.ffn_embed_dim,
            num_heads=self.num_heads,
            pre_layernorm=self.pre_layernorm,
            scalar_dim_hidden=self.scalar_dim_hidden,
            scalar_num_layers=self.scalar_num_layers,
            func_enc_type=self.func_enc_type,
            func_enc_num_branches=self.func_enc_num_branches,
            func_enc_resolution=self.func_enc_resolution,
            func_enc_input_txyz=self.func_enc_input_txyz,
            func_enc_keep_nchw=self.func_enc_keep_nchw,
            dtype=self.dtype,
            name="pde_encoder",
        )

        pde_feature = pde_encoder(
            node_type,
            node_scalar,
            node_function,
            in_degree,
            out_degree,
            attn_bias,
            spatial_pos,
            deterministic=deterministic,
        )  # [n_node, n_graph, embed_dim]

        # INR with hypernet
        inr = PolyINRWithHypernet(
            inr_dim_in=1 + SPACE_DIM,  # [t, x, y, z]
            inr_dim_out=1,
            inr_dim_hidden=self.inr_dim_hidden,
            inr_num_layers=self.inr_num_layers,
            hyper_dim_in=self.embed_dim,
            hyper_dim_hidden=self.hyper_dim_hidden,
            hyper_num_layers=self.hyper_num_layers,
            share_hypernet=self.share_hypernet,
            enable_affine=self.enable_affine,
            enable_shift=self.enable_shift,
            enable_scale=self.enable_scale,
            activation_fn=self.activation_fn,
            affine_act_fn=self.affine_act_fn,
            dtype=self.dtype,
            name="inr",
        )

        out = inr(coordinate, pde_feature[: self.n_inr_nodes])

        if self.multi_inr:
            inr2 = PolyINRWithHypernet(
                inr_dim_in=1 + SPACE_DIM,
                inr_dim_out=1,
                inr_dim_hidden=self.inr_dim_hidden,
                inr_num_layers=self.inr_num_layers,
                hyper_dim_in=self.embed_dim,
                hyper_dim_hidden=self.hyper_dim_hidden,
                hyper_num_layers=self.hyper_num_layers,
                share_hypernet=self.share_hypernet,
                enable_affine=self.enable_affine,
                enable_shift=self.enable_shift,
                enable_scale=self.enable_scale,
                activation_fn=self.activation_fn,
                affine_act_fn=self.affine_act_fn,
                dtype=self.dtype,
                name="inr2",
            )

            if self.separate_latent:
                out2 = inr2(coordinate, pde_feature[self.n_inr_nodes :])
            else:
                out2 = inr2(coordinate, pde_feature[: self.n_inr_nodes])
            return out + out2

        return out


def create_pdeformer_from_config(config: dict, dtype=jnp.float32) -> PDEformer:
    """Create a PDEformer model from a configuration dictionary."""
    model_config = config.get("model", config)
    graphormer_config = model_config.get("graphormer", {})
    scalar_enc_config = model_config.get("scalar_encoder", {})
    func_enc_config = model_config.get("function_encoder", {})
    inr_config = model_config.get("inr", {})
    hypernet_config = model_config.get("hypernet", {})
    multi_inr_config = model_config.get("multi_inr", {})
    poly_inr_config = inr_config.get("poly_inr", {})

    return PDEformer(
        # Graphormer
        num_node_type=graphormer_config.get("num_node_type", 128),
        num_in_degree=graphormer_config.get("num_in_degree", 32),
        num_out_degree=graphormer_config.get("num_out_degree", 32),
        num_spatial=graphormer_config.get("num_spatial", 16),
        num_encoder_layers=graphormer_config.get("num_encoder_layers", 12),
        embed_dim=graphormer_config.get("embed_dim", 768),
        ffn_embed_dim=graphormer_config.get("ffn_embed_dim", 1536),
        num_heads=graphormer_config.get("num_heads", 32),
        pre_layernorm=graphormer_config.get("pre_layernorm", True),
        # Scalar encoder
        scalar_dim_hidden=scalar_enc_config.get("dim_hidden", 256),
        scalar_num_layers=scalar_enc_config.get("num_layers", 3),
        # Function encoder
        func_enc_type=func_enc_config.get("type", "cnn2dv3"),
        func_enc_num_branches=func_enc_config.get("num_branches", 4),
        func_enc_resolution=func_enc_config.get("resolution", 128),
        func_enc_input_txyz=func_enc_config.get("conv2d_input_txyz", False),
        func_enc_keep_nchw=func_enc_config.get("cnn_keep_nchw", True),
        # INR
        inr_dim_hidden=inr_config.get("dim_hidden", 768),
        inr_num_layers=inr_config.get("num_layers", 12),
        enable_affine=poly_inr_config.get("enable_affine", False),
        enable_shift=poly_inr_config.get("enable_shift", True),
        enable_scale=poly_inr_config.get("enable_scale", True),
        activation_fn=poly_inr_config.get("activation_fn", "sin"),
        affine_act_fn=poly_inr_config.get("affine_act_fn", "identity"),
        # Hypernet
        hyper_dim_hidden=hypernet_config.get("dim_hidden", 512),
        hyper_num_layers=hypernet_config.get("num_layers", 2),
        share_hypernet=hypernet_config.get("shared", False),
        # Multi-INR
        multi_inr=multi_inr_config.get("enable", False),
        separate_latent=multi_inr_config.get("separate_latent", False),
        dtype=dtype,
    )


# Configuration for PDEformer-2-Small
PDEFORMER_SMALL_CONFIG = {
    "graphormer": {
        "num_node_type": 128,
        "num_in_degree": 32,
        "num_out_degree": 32,
        "num_spatial": 16,
        "num_encoder_layers": 9,
        "embed_dim": 512,
        "ffn_embed_dim": 1024,
        "num_heads": 32,
        "pre_layernorm": True,
    },
    "scalar_encoder": {
        "dim_hidden": 256,
        "num_layers": 3,
    },
    "function_encoder": {
        "type": "cnn2dv3",
        "num_branches": 4,
        "resolution": 128,
        "conv2d_input_txyz": False,
        "cnn_keep_nchw": True,
    },
    "inr": {
        "type": "poly_inr",
        "num_layers": 12,
        "dim_hidden": 128,
        "poly_inr": {
            "enable_affine": False,
            "enable_shift": True,
            "enable_scale": True,
            "modify_he_init": False,
            "affine_act_fn": "identity",
            "activation_fn": "sin",
        },
    },
    "hypernet": {
        "dim_hidden": 512,
        "num_layers": 2,
        "shared": False,
    },
    "multi_inr": {
        "enable": False,
    },
}

# Configuration for PDEformer-2-Base
PDEFORMER_BASE_CONFIG = {
    "graphormer": {
        "num_encoder_layers": 12,
        "embed_dim": 768,
        "ffn_embed_dim": 1536,
        "num_heads": 32,
    },
    "inr": {
        "dim_hidden": 768,
        "num_layers": 12,
        "poly_inr": {
            "enable_shift": True,
            "enable_scale": True,
        },
    },
    "hypernet": {
        "dim_hidden": 512,
    },
}

# Configuration for PDEformer-2-Fast
# PDEFORMER_FAST_CONFIG = {
#    "graphormer": {
#        "num_encoder_layers": 12,
#        "embed_dim": 768,
#        "ffn_embed_dim": 1536,
#        "num_heads": 32,
#    },
#    "inr": {
#        "dim_hidden": 256,
#        "num_layers": 12,
#        "poly_inr": {
#            "enable_shift": False,
#            "enable_scale": False,
#        }
#    },
# }

# Configuration for PDEformer-2-Base
PDEFORMER_FAST_CONFIG = {
    "graphormer": {
        "num_encoder_layers": 12,
        "embed_dim": 768,
        "ffn_embed_dim": 1536,
        "num_heads": 32,
    },
    "inr": {
        "dim_hidden": 256,
        "num_layers": 12,
        "poly_inr": {
            "enable_shift": True,
            "enable_scale": True,
        },
    },
    "hypernet": {
        "dim_hidden": 512,
    },
}
