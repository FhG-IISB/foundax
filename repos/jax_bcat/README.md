# jax_bcat

JAX/Flax translation of [BCAT: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics](https://arxiv.org/abs/2501.18972).

## Installation

```bash
pip install -e .
```

## Usage

```python
import jax
import jax.numpy as jnp
from jax_bcat import bcat_default

model = bcat_default()

# Dummy input: (batch, timesteps, x, y, channels)
data = jnp.ones((1, 20, 128, 128, 4))
times = jnp.arange(20, dtype=jnp.float32).reshape(1, 20, 1)

variables = model.init(jax.random.PRNGKey(0), data, times, input_len=10)
output = model.apply(variables, data, times, input_len=10)
# output shape: (1, 10, 128, 128, 4)
```
