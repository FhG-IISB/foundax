"""
Model variant configurations for AViT.

Defines the four standard AViT variants (Ti / S / B / L) from the
`Multiple Physics Pretraining <https://openreview.net/forum?id=DKSI3bULiZ>`_ paper.
"""

# ---- Variant specs --------------------------------------------------------
# embed_dim / num_heads / processor_blocks
AVIT_CONFIGS = {
    "Ti": {"embed_dim": 192, "num_heads": 3, "processor_blocks": 12, "n_states": 12},
    "S": {"embed_dim": 384, "num_heads": 6, "processor_blocks": 12, "n_states": 12},
    "B": {"embed_dim": 768, "num_heads": 12, "processor_blocks": 12, "n_states": 12},
    "L": {"embed_dim": 1024, "num_heads": 16, "processor_blocks": 24, "n_states": 12},
}
