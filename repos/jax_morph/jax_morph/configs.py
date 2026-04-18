"""
MORPH model configurations and convenience constructors.

Each config maps to the official MORPH model variants from
``config/argument_parser.py`` in the MORPH repository.

Format: ``{conv_filter, dim, depth, heads, mlp_dim, max_ar, model_size}``.
"""

from typing import Any, Dict

#: Model configurations keyed by variant name.
MORPH_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Ti": dict(
        conv_filter=8,
        dim=256,
        depth=4,
        heads=4,
        mlp_dim=1024,
        max_ar=1,
        model_size="Ti",
    ),
    "S": dict(
        conv_filter=8, dim=512, depth=4, heads=8, mlp_dim=2048, max_ar=1, model_size="S"
    ),
    "M": dict(
        conv_filter=8,
        dim=768,
        depth=8,
        heads=12,
        mlp_dim=3072,
        max_ar=1,
        model_size="M",
    ),
    "L": dict(
        conv_filter=8,
        dim=1024,
        depth=16,
        heads=16,
        mlp_dim=4096,
        max_ar=16,
        model_size="L",
    ),
}

#: HuggingFace checkpoint filenames.
CHECKPOINT_NAMES: Dict[str, str] = {
    "Ti": "morph-Ti-FM-max_ar1_ep225.pth",
    "S": "morph-S-FM-max_ar1_ep225.pth",
    "M": "morph-M-FM-max_ar1_ep290_latestbatch.pth",
    "L": "morph-L-FM-max_ar16_ep189_latestbatch.pth",
}

#: HuggingFace repo ID for downloading pretrained checkpoints.
HF_REPO_ID = "mahindrautela/MORPH"
