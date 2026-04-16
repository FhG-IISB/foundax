# jax_prose

JAX/Flax translation-in-progress of PROSE-FD model architectures.

This repository currently includes the first translated path:
- PROSE-FD `prose_1to1` forward architecture (embedder + data encoder + operator decoder)

## Status

Implemented in this initial pass:
- Model config dataclasses for PROSE-FD 1to1 defaults
- `LinearEmbedder` and `ConvEmbedder` core logic
- Data encoder transformer stack (self-attention + FFN)
- Data operator decoder (cross-attention + FFN, with time query embeddings)
- Top-level `PROSE1to1` model
- Basic shape/forward test

Pending for full 1:1 parity:
- Full `prose_2to1` path (symbol encoder + fusion)
- Optional custom attention and rotary behavior parity
- Weight conversion scripts from PyTorch checkpoints
- jNO factory integration entry in `jno/architectures/models.py`

## Quick start

```bash
cd /home/users/armbrust/projects/jax_prose
pytest -q
```

## Notes

The implementation follows channels-last tensors (`B, H, W, C`) in Flax/JAX.
Input and output tensors for PROSE remain in sequence format:
- Input: `(B, T_in, H, W, C)`
- Output: `(B, T_out, H, W, C)`
