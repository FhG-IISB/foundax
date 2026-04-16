"""Inspect the fast checkpoint to determine correct config."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jax_pdeformer2.utils import load_mindspore_checkpoint

ckpt_path = "../pdeformer2-fast.ckpt"

print("Loading checkpoint...")
weights = load_mindspore_checkpoint(ckpt_path)

print(f"\nTotal parameters: {len(weights)}")
print("\nSample parameter names and shapes:")
for i, (name, param) in enumerate(list(weights.items())[:30]):
    print(f"{name}: {param.shape}")

# Look for key dimensions
print("\n\nKey dimensions:")
for name, param in weights.items():
    if "graphormer" in name and "layers.0" in name and "fc1.weight" in name:
        print(f"FFN dimension: {name}: {param.shape}")  # Should be [ffn_dim, embed_dim]
    if "graphormer" in name and "layers.0" in name and "in_proj.weight" in name:
        print(
            f"Attention dimension: {name}: {param.shape}"
        )  # Should be [3*embed_dim, embed_dim]
    if "inr" in name and "affines.0.weight" in name:
        print(
            f"INR hidden dimension: {name}: {param.shape}"
        )  # Should be [inr_hidden_dim, inr_in_dim]
    if "inr" in name and "dense_layers.0.weight" in name:
        print(f"INR hidden layer: {name}: {param.shape}")

# Count layers
graphormer_layers = set()
inr_layers = set()
for name in weights.keys():
    if "graphormer.layers." in name:
        layer_num = name.split("graphormer.layers.")[1].split(".")[0]
        graphormer_layers.add(int(layer_num))
    if "inr.dense_layers." in name:
        layer_num = name.split("inr.dense_layers.")[1].split(".")[0]
        inr_layers.add(int(layer_num))

print(f"\nGraphormer layers: {sorted(graphormer_layers)}")
print(f"INR dense layers: {sorted(inr_layers)}")

print("\n\nShift hypernet weights:")
for name, param in weights.items():
    if "shift_hypernets_0" in name:
        print(f"{name}: {param.shape}")

print("\n\nScale hypernet weights:")
for name, param in weights.items():
    if "scale_hypernets_0" in name:
        print(f"{name}: {param.shape}")
