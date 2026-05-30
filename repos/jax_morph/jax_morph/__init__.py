"""
jax_morph: JAX/Equinox implementation of the MORPH PDE foundation model.

Reference:
    Rautela et al., "MORPH: PDE Foundation Models with Arbitrary Data Modality" (2025)
    https://arxiv.org/abs/2509.21670
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("jax_morph")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from jax_morph.model_eqx import ViT3DRegression
from jax_morph.configs import (
    MORPH_CONFIGS,
    CHECKPOINT_NAMES,
    HF_REPO_ID,
)
from jax_morph.convert_weights import convert_pytorch_to_jax_params


def load_pytorch_state_dict(path, **kwargs):
    """Load a PyTorch MORPH checkpoint, returning the state_dict."""
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


__all__ = [
    "__version__",
    "ViT3DRegression",
    "MORPH_CONFIGS",
    "CHECKPOINT_NAMES",
    "HF_REPO_ID",
    "convert_pytorch_to_jax_params",
    "load_pytorch_state_dict",
]
