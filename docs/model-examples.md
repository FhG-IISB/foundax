# Model Examples

This page provides short, practical examples for the Equinox-facing API.

## Core model examples

```python
import jax
import jax.numpy as jnp
import foundax as fx

# MLP
mlp = fx.mlp(in_features=2, output_dim=1, hidden_dims=64, num_layers=3)
x = jnp.ones((16, 2))
y = mlp(x)

# FNO2D
fno = fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)
grid = jnp.ones((64, 64, 1))
out = fno(grid)

# UNet2D
unet = fx.unet2d(in_channels=1, out_channels=1)
img = jnp.ones((128, 128, 1))
out = unet(img)
```

## Foundation wrapper examples

### Poseidon

```python
import foundax as fx
import jax.numpy as jnp

model = fx.poseidon.T()
x = jnp.ones((1, 128, 128, 4))
t = jnp.array([0.5])
out = model(pixel_values=x, time=t).output
```

### MORPH

```python
import foundax as fx
import jax.numpy as jnp

model = fx.morph.S()
# Example shape: (batch, time, fields, channels, depth, height, width)
x = jnp.ones((1, 2, 1, 1, 16, 16, 16))
out = model(x)
```

### MPP

```python
import foundax as fx
import jax.numpy as jnp

model = fx.mpp.Ti(n_states=3)
# (time, batch, channels, height, width)
x = jnp.ones((2, 1, 3, 64, 64))
state_labels = jnp.array([0, 1, 2])
bcs = jnp.zeros((1, 2), dtype=jnp.int32)
out = model(x, state_labels, bcs)
```

### Walrus

```python
import foundax as fx
import jax.numpy as jnp

model = fx.walrus.base()
# (batch, time, height, width, channels)
x = jnp.ones((1, 2, 64, 64, 4))
state_labels = jnp.array([0, 1, 2, 3])
bcs = [[0, 0], [0, 0]]
out = model(x, state_labels, bcs)
```

### BCAT

```python
import foundax as fx
import jax.numpy as jnp

model = fx.bcat.base()
# (batch, t_in + t_out, 128, 128, channels)
x = jnp.ones((1, 6, 128, 128, 2))
t = jnp.linspace(0.0, 1.0, 6).reshape(1, 6, 1)
out = model(x, t, input_len=4)
```

### PDEformer-2

```python
import foundax as fx
import jax
import jax.numpy as jnp

model = fx.pdeformer2.small()
key = jax.random.PRNGKey(0)

# Minimal synthetic graph-like inputs
n_graph, n_node, n_points = 1, 6, 16
node_type = jnp.ones((n_graph, n_node, 1), dtype=jnp.int32)
node_scalar = jnp.ones((n_graph, 4, 1))
node_function = jnp.ones((n_graph, 2, 128 * 128, 5))
in_degree = jnp.zeros((n_graph, n_node), dtype=jnp.int32)
out_degree = jnp.zeros((n_graph, n_node), dtype=jnp.int32)
attn_bias = jnp.zeros((n_graph, n_node, n_node))
spatial_pos = jnp.zeros((n_graph, n_node, n_node), dtype=jnp.int32)
coordinate = jax.random.uniform(key, (n_graph, n_points, 4))

out = model(
    node_type,
    node_scalar,
    node_function,
    in_degree,
    out_degree,
    attn_bias,
    spatial_pos,
    coordinate,
)
```

### DPOT

```python
import foundax as fx
import jax.numpy as jnp

model = fx.dpot.Ti()
# (batch, height, width, time, channels)
x = jnp.ones((1, 128, 128, 3, 2))
pred, cls = model(x)
```

### PROSE

```python
import foundax as fx
import jax.numpy as jnp

# fd_1to1 returns (model, variables)
model, variables = fx.prose.fd_1to1()

x = jnp.ones((1, 2, 128, 128, 2))
t_in = jnp.zeros((1, 2, 1))
t_out = jnp.ones((1, 2, 1))

pred = model.apply(variables, x, t_in, t_out, deterministic=True)
```

## Notes

- Input signatures differ between families; check the constructor docstring in the wrapper module when integrating a model.
- For production training code, pair these constructors with your optimizer and training loop directly in JAX/Equinox.
