"""PolyINR with hypernet implementation in JAX/Flax."""
from typing import Sequence, Optional, Tuple
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .basic_block import MLP, Clamp


class PolyINR(nn.Module):
    """
    PolyINR is an implicit neural representation (INR) architecture.
    Based on paper: https://arxiv.org/abs/2303.11424
    """
    dim_in: int
    dim_out: int
    dim_hidden: int
    num_layers: int
    activation_fn: str = "lrelu"
    affine_act_fn: str = "identity"
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, affine_modulations=None, scale_modulations=None, shift_modulations=None):
        """
        Args:
            x: [bsz, num_points, dim_in]
            affine_modulations: [num_layers-1, bsz, dim_in+1, dim_hidden]
            scale_modulations: [num_layers-1, bsz, dim_hidden]
            shift_modulations: [num_layers-1, bsz, dim_hidden]
        Returns:
            [bsz, num_points, dim_out]
        """
        # Pad coordinates with 1
        x_pad = jnp.concatenate([x, jnp.ones((*x.shape[:-1], 1), dtype=x.dtype)], axis=-1)
        
        hidden_state = 1.0
        
        for layer_idx in range(self.num_layers - 1):
            # Scale modulation
            if scale_modulations is None:
                scale = 1.0
            else:
                scale = 1.0 + jnp.expand_dims(scale_modulations[layer_idx], axis=1)  # [bsz, 1, dim_hidden]
            
            # Shift modulation
            if shift_modulations is None:
                shift = 0.0
            else:
                shift = jnp.expand_dims(shift_modulations[layer_idx], axis=1)  # [bsz, 1, dim_hidden]
            
            # Affine layer
            affine = nn.Dense(
                self.dim_hidden,
                dtype=self.dtype,
                name=f"affines_{layer_idx}"
            )(x)
            
            # Affine activation
            if self.affine_act_fn.lower() in ["none", "identity"]:
                tmp = affine
            elif self.affine_act_fn.lower() in ["lrelu", "leakyrelu"]:
                tmp = nn.leaky_relu(affine, negative_slope=0.2)
                tmp = jnp.clip(tmp, -256.0, 256.0)
            elif self.affine_act_fn.lower() in ["sin", "sine"]:
                tmp = jnp.sin(affine) * math.sqrt(2.0)
            else:
                tmp = affine
            
            # Add affine modulations
            if affine_modulations is not None:
                tmp2 = jnp.matmul(x_pad, affine_modulations[layer_idx])  # [bsz, n_pts, dim_hidden]
                tmp = tmp + tmp2
            
            # Multiplicative interaction
            hidden_state = hidden_state * tmp  # [bsz, n_pts, dim_hidden]
            
            # Dense layer
            hidden_state = nn.Dense(
                self.dim_hidden,
                dtype=self.dtype,
                name=f"dense_layers_{layer_idx}"
            )(hidden_state)
            
            # Apply scale and shift
            hidden_state = scale * hidden_state + shift
            
            # Activation
            if self.activation_fn.lower() in ["lrelu", "leakyrelu"]:
                hidden_state = nn.leaky_relu(hidden_state, negative_slope=0.2)
                hidden_state = jnp.clip(hidden_state, -256.0, 256.0)
            elif self.activation_fn.lower() in ["sin", "sine"]:
                hidden_state = jnp.sin(hidden_state)
        
        # Final layer
        out = nn.Dense(
            self.dim_out,
            dtype=self.dtype,
            name="last_layer"
        )(hidden_state)
        
        return out


class PolyINRWithHypernet(nn.Module):
    """PolyINR model with hypernets for modulation generation."""
    inr_dim_in: int
    inr_dim_out: int
    inr_dim_hidden: int
    inr_num_layers: int
    hyper_dim_in: int
    hyper_dim_hidden: int
    hyper_num_layers: int
    share_hypernet: bool = False
    enable_affine: bool = False
    enable_shift: bool = True
    enable_scale: bool = True
    activation_fn: str = "lrelu"
    affine_act_fn: str = "identity"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.affine_modulations_shape = (
            self.inr_num_layers - 1, -1, self.inr_dim_in + 1, self.inr_dim_hidden
        )

    @nn.compact
    def __call__(self, coordinate, hyper_in):
        """
        Args:
            coordinate: [n_graph, num_points, dim_in]
            hyper_in: [n_inr_node, n_graph, embed_dim]
        Returns:
            [n_graph, num_points, dim_out]
        """
        # Get modulations from hypernets
        affine_modulations, scale_modulations, shift_modulations = self._get_modulations(hyper_in)
        
        # Create the INR
        inr = PolyINR(
            dim_in=self.inr_dim_in,
            dim_out=self.inr_dim_out,
            dim_hidden=self.inr_dim_hidden,
            num_layers=self.inr_num_layers,
            activation_fn=self.activation_fn,
            affine_act_fn=self.affine_act_fn,
            dtype=self.dtype,
            name="inr"
        )
        
        out = inr(coordinate, affine_modulations, scale_modulations, shift_modulations)
        return out
    
    def _get_modulations(self, hyper_in):
        """Generate modulations from hypernets."""
        num_hypernet = self.inr_num_layers - 1
        
        # Affine modulations
        if self.enable_affine:
            hyper_dim_out = (self.inr_dim_in + 1) * self.inr_dim_hidden
            affine_modulations = []
            for idx in range(num_hypernet):
                encoder_in = hyper_in[idx]  # [n_graph, embed_dim]
                if self.share_hypernet:
                    encoder_out = MLP(
                        dim_out=hyper_dim_out,
                        dim_hidden=self.hyper_dim_hidden,
                        num_layers=self.hyper_num_layers,
                        dtype=self.dtype,
                        name="affine_hypernet"
                    )(encoder_in)
                else:
                    encoder_out = MLP(
                        dim_out=hyper_dim_out,
                        dim_hidden=self.hyper_dim_hidden,
                        num_layers=self.hyper_num_layers,
                        dtype=self.dtype,
                        name=f"affine_hypernets_{idx}"
                    )(encoder_in)
                affine_modulations.append(encoder_out)
            
            affine_modulations = jnp.stack(affine_modulations, axis=0)
            affine_modulations = affine_modulations.reshape(self.affine_modulations_shape)
        else:
            affine_modulations = None
        
        # Shift modulations
        if self.enable_shift:
            shift_modulations = []
            for idx in range(num_hypernet):
                encoder_in = hyper_in[idx]
                if self.share_hypernet:
                    encoder_out = MLP(
                        dim_out=self.inr_dim_hidden,
                        dim_hidden=self.hyper_dim_hidden,
                        num_layers=self.hyper_num_layers,
                        dtype=self.dtype,
                        name="shift_hypernet"
                    )(encoder_in)
                else:
                    encoder_out = MLP(
                        dim_out=self.inr_dim_hidden,
                        dim_hidden=self.hyper_dim_hidden,
                        num_layers=self.hyper_num_layers,
                        dtype=self.dtype,
                        name=f"shift_hypernets_{idx}"
                    )(encoder_in)
                shift_modulations.append(encoder_out)
            shift_modulations = jnp.stack(shift_modulations, axis=0)
        else:
            shift_modulations = None
        
        # Scale modulations
        if self.enable_scale:
            scale_modulations = []
            for idx in range(num_hypernet):
                encoder_in = hyper_in[idx]
                if self.share_hypernet:
                    encoder_out = MLP(
                        dim_out=self.inr_dim_hidden,
                        dim_hidden=self.hyper_dim_hidden,
                        num_layers=self.hyper_num_layers,
                        dtype=self.dtype,
                        name="scale_hypernet"
                    )(encoder_in)
                else:
                    encoder_out = MLP(
                        dim_out=self.inr_dim_hidden,
                        dim_hidden=self.hyper_dim_hidden,
                        num_layers=self.hyper_num_layers,
                        dtype=self.dtype,
                        name=f"scale_hypernets_{idx}"
                    )(encoder_in)
                scale_modulations.append(encoder_out)
            scale_modulations = jnp.stack(scale_modulations, axis=0)
        else:
            scale_modulations = None
        
        return affine_modulations, scale_modulations, shift_modulations


def get_inr_with_hypernet(config_model: dict, dim_in: int = 1, dim_out: int = 1, 
                          inr_base: bool = True, dtype=jnp.float32):
    """
    Create an INR with hypernet based on configuration.
    """
    if inr_base:
        config_inr = config_model["inr"]
    else:
        config_inr = config_model.get("inr2", config_model["inr"])
    
    inr_type = config_inr.get("type", "poly_inr").lower()
    
    if inr_type == "poly_inr":
        poly_inr_config = config_inr.get("poly_inr", {})
        return PolyINRWithHypernet(
            inr_dim_in=dim_in,
            inr_dim_out=dim_out,
            inr_dim_hidden=config_inr["dim_hidden"],
            inr_num_layers=config_inr["num_layers"],
            hyper_dim_in=config_model["graphormer"]["embed_dim"],
            hyper_dim_hidden=config_model["hypernet"]["dim_hidden"],
            hyper_num_layers=config_model["hypernet"]["num_layers"],
            share_hypernet=config_model["hypernet"].get("shared", False),
            enable_affine=poly_inr_config.get("enable_affine", False),
            enable_shift=poly_inr_config.get("enable_shift", True),
            enable_scale=poly_inr_config.get("enable_scale", True),
            activation_fn=poly_inr_config.get("activation_fn", "sin"),
            affine_act_fn=poly_inr_config.get("affine_act_fn", "identity"),
            dtype=dtype,
        )
    else:
        raise NotImplementedError(f"INR type '{inr_type}' not implemented in JAX.")
