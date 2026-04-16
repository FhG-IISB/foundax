"""Function encoder implementations in JAX/Flax."""
from typing import Sequence, Optional
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .basic_block import MLP


class Conv2dFuncEncoderV3(nn.Module):
    """CNN Encoder for functions defined on two-dimensional uniform grids (V3 architecture)."""
    in_dim: int = 5
    out_dim: int = 256
    resolution: int = 128
    input_txyz: bool = True
    keep_nchw: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: [bsz, num_points_ic, dim_in], dim_in=5
        Returns:
            [bsz, out_dim, H/64, W/64] if keep_nchw else [bsz, H/64, W/64, out_dim]
        """
        bsz, _, dim_in = x.shape
        
        # Reshape: [bsz, num_points, dim_in] -> [bsz, H, W, dim_in]
        x = x.reshape(bsz, self.resolution, self.resolution, dim_in)
        
        # NHWC -> NCHW for convolution
        x = jnp.transpose(x, (0, 3, 1, 2))  # [bsz, dim_in, H, W]
        
        f_values = x[:, -1:, :, :]  # [bsz, 1, H, W]
        
        if self.input_txyz:
            net_in = x
        else:
            net_in = f_values
        
        in_channels = net_in.shape[1]
        
        # Conv2d with kernel_size=4, stride=4
        # Layer 1: [bsz, in_channels, H, W] -> [bsz, 32, H/4, W/4]
        x = nn.Conv(
            features=32,
            kernel_size=(4, 4),
            strides=(4, 4),
            padding='VALID',
            kernel_init=nn.initializers.he_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            name="conv1"
        )(jnp.transpose(net_in, (0, 2, 3, 1)))  # NCHW -> NHWC for JAX Conv
        x = nn.relu(x)
        
        # Layer 2: [bsz, H/4, W/4, 32] -> [bsz, H/16, W/16, 128]
        x = nn.Conv(
            features=128,
            kernel_size=(4, 4),
            strides=(4, 4),
            padding='VALID',
            kernel_init=nn.initializers.he_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            name="conv2"
        )(x)
        x = nn.relu(x)
        
        # Layer 3: [bsz, H/16, W/16, 128] -> [bsz, H/64, W/64, out_dim]
        x = nn.Conv(
            features=self.out_dim,
            kernel_size=(4, 4),
            strides=(4, 4),
            padding='VALID',
            kernel_init=nn.initializers.he_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            name="conv3"
        )(x)
        
        if self.keep_nchw:
            # NHWC -> NCHW
            x = jnp.transpose(x, (0, 3, 1, 2))
        
        return x  # [bsz, out_dim, H/64, W/64] or [bsz, H/64, W/64, out_dim]


def get_function_encoder(config_fenc: dict, dim_in: int, dim_out: int, dtype=jnp.float32):
    """Get the function encoder network based on config."""
    function_encoder_type = config_fenc.get("type", "cnn2dv3").lower()
    
    if function_encoder_type == "cnn2dv3":
        resolution = config_fenc.get("resolution", 128)
        input_txyz = config_fenc.get("conv2d_input_txyz", False)
        keep_nchw = config_fenc.get("cnn_keep_nchw", True)
        
        return Conv2dFuncEncoderV3(
            in_dim=dim_in,
            out_dim=dim_out,
            resolution=resolution,
            input_txyz=input_txyz,
            keep_nchw=keep_nchw,
            dtype=dtype,
        )
    else:
        raise NotImplementedError(f"Function encoder type '{function_encoder_type}' not implemented in JAX.")
