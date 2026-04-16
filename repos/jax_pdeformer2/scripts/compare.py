#!/usr/bin/env python3
"""
Combined script to test MindSpore vs JAX predictions for PDEFormer2 checkpoints.
Tests: pdeformer2-small, pdeformer2-base, pdeformer2-fast
PDEs:
  1. Advection-Burgers: u_t + (u^2)_x + (-0.3*u)_y = 0
  2. Heat Equation: u_t - 0.1*(u_xx + u_yy) = 0

"""

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================

# Configuration

# ============================================================================

PDEFORMER2_ROOT = "/home/b8cl/pdeformer-2"
DATA_DIR = "/home/b8cl/projects/DATA/pdeformer"
OUTPUT_DIR = Path("./")

# Checkpoint configurations: (name, config_file, uf_num_mod)

CHECKPOINTS = [
    ("small", "configs/inference/model-S.yaml", 11),
    ("base", "configs/inference/model-L.yaml", 11),
    ("fast", "configs/inference/model-M.yaml", 11),
]

# PDE configurations: (name, description, create_func_jax, create_func_ms)

PDES = []  # Will be populated after PDENodesCollector is defined

# ============================================================================

# DAG Building

# ============================================================================

VAR_NODE_TYPES = ["uf"]
FUNCTION_NODE_TYPES = ["ic", "cf", "bv", "sdf", "eval"]
OPERATOR_NODE_TYPES = [
    "add",
    "mul",
    "eq0",
    "dt",
    "dx",
    "dy",
    "dz",
    "dn",
    "avg_int",
    "neg",
    "square",
    "exp10",
    "log10",
    "sin",
    "cos",
]
RESERVED_NODE_TYPES = ["vc", "at"] + [f"Reserved{i}" for i in range(14)]
FUNCTION_BRANCH_NODE_TYPES = [f"Branch{i}" for i in range(16)]
INR_NODE_TYPES = [f"Mod{i}" for i in range(32)]
DAG_NODE_TYPES = (
    ["pad"]
    + VAR_NODE_TYPES
    + ["coef"]
    + FUNCTION_NODE_TYPES
    + OPERATOR_NODE_TYPES
    + RESERVED_NODE_TYPES
    + FUNCTION_BRANCH_NODE_TYPES
    + INR_NODE_TYPES
)
NODE_TYPE_DICT = {t: i for i, t in enumerate(DAG_NODE_TYPES)}

MAX_N_SCALAR_NODES = 80
MAX_N_FUNCTION_NODES = 6
FUNCTION_NUM_BRANCHES = 4
NUM_SPATIAL = 16

# Fixed coords for function encoder (128x128)

x_fenc, y_fenc = np.meshgrid(
    np.linspace(0, 1, 129)[:-1],
    np.linspace(0, 1, 129)[:-1],
    indexing="ij",
)


def build_dag_inputs(node_list, node_scalar_list, node_function_list, uf_num_mod):
    """Build all DAG input arrays."""
    import networkx as nx

    mod_node_list, function_branch_node_list, edge_list = [], [], []
    n_vars = n_functions = 0

    for node, type_, *preds in node_list:
        edge_list.extend([(p, node) for p in preds])
        if type_ in VAR_NODE_TYPES:
            n_vars += 1
            for j in range(uf_num_mod):
                mn = f"{node}:Mod{j}"
                mod_node_list.append((mn, f"Mod{j}"))
                edge_list.append((mn, node))
        elif type_ in FUNCTION_NODE_TYPES + ["vc"]:
            n_functions += 1
            for j in range(FUNCTION_NUM_BRANCHES):
                bn = f"{node}:Branch{j}"
                function_branch_node_list.append((bn, f"Branch{j}"))
                if type_ in ["ic", "bv"]:
                    edge_list.append((node, bn))
                else:
                    edge_list.append((bn, node))

    n_func_pad = MAX_N_FUNCTION_NODES - n_functions
    pad_len = n_func_pad * FUNCTION_NUM_BRANCHES

    dag_nodes = mod_node_list + node_list
    n_scalar_pad = MAX_N_SCALAR_NODES - len(dag_nodes)
    dag_nodes += [(f"pad{j}", "pad") for j in range(n_scalar_pad)]
    dag_nodes += function_branch_node_list

    dag = nx.DiGraph()
    dag.add_nodes_from([(n, {"typeid": NODE_TYPE_DICT[t]}) for n, t, *_ in dag_nodes])
    dag.add_edges_from(edge_list)

    nt = np.array([a["typeid"] for _, a in dag.nodes.data()], dtype=np.int32)
    pmask = nt == 0
    nt = np.pad(nt, (0, pad_len))[:, None]

    ns = np.pad(
        np.array(node_scalar_list, dtype=np.float32),
        (n_vars * uf_num_mod, n_scalar_pad),
    )[:MAX_N_SCALAR_NODES, None]

    nf = np.pad(
        np.array(node_function_list, dtype=np.float32),
        ((0, n_func_pad), (0, 0), (0, 0)),
    )

    ind = 1 + np.array([d for _, d in dag.in_degree()], np.int32)
    ind[pmask] = 0
    ind = np.pad(ind, (0, pad_len))
    outd = 1 + np.array([d for _, d in dag.out_degree()], np.int32)
    outd[pmask] = 0
    outd = np.pad(outd, (0, pad_len))

    sp = 1 + nx.floyd_warshall_numpy(dag).clip(0, NUM_SPATIAL - 2).astype(np.int32)
    sp[pmask] = 0
    sp[:, pmask] = 0
    sp = np.pad(sp, ((0, pad_len), (0, pad_len)))

    n = nt.shape[0]
    ab = np.zeros((n, n), np.float32)
    cm = sp == sp.max()
    cm = cm & cm.T
    ab[cm] = -np.inf
    ab[:, nt[:, 0] == 0] = -np.inf

    return dict(
        node_type=nt,
        node_scalar=ns,
        node_function=nf,
        in_degree=ind,
        out_degree=outd,
        attn_bias=ab,
        spatial_pos=sp,
    )


class PDENode:
    def __init__(self, name, c):
        self.name, self._c, self._d = name, c, {}

    def __neg__(self):
        return self._c.neg(self)

    def __add__(self, o):
        return self if o == 0 else self._c.sum(o, self)

    __radd__ = __add__

    def __mul__(self, o):
        if o == 0:
            return 0
        if o == 1:
            return self
        if o == -1:
            return self._c.neg(self)
        return self._c.prod(o, self)

    __rmul__ = __mul__

    def __sub__(self, o):
        return self.__add__(-o)


class PDENodesCollector:
    def __init__(self):
        self.node_list, self.node_scalar, self.node_function = [], [], []

    def _add_node(self, t, preds=None, s=0.0):
        nm = f"{t}{len(self.node_list)}"
        self.node_list.append((nm, t, *(n.name for n in preds)) if preds else (nm, t))
        self.node_scalar.append(float(s))
        return PDENode(nm, self)

    def _add_func(self, fv, *, t=-1.0, x=-1.0, y=-1.0, z=-1.0):
        self.node_function.append(
            np.stack(np.broadcast_arrays(t, x, y, z, fv), -1)
            .reshape(-1, 5)
            .astype(np.float32)
        )

    def new_uf(self):
        return self._add_node("uf")

    def new_coef(self, v):
        return self._add_node("coef", s=v)

    def set_ic(self, src, fv, *, x=-1.0, y=-1.0, z=-1.0):
        self._add_func(fv, t=0.0, x=x, y=y, z=z)
        self._add_node("ic", [src])

    def dt(self, s):
        return self._add_node("dt", [s])

    def dx(self, s):
        return self._add_node("dx", [s])

    def dy(self, s):
        return self._add_node("dy", [s])

    def neg(self, s):
        return self._add_node("neg", [s])

    def square(self, s):
        return self._add_node("square", [s])

    def _filter(self, nodes, ig):
        if len(nodes) == 1 and isinstance(nodes[0], (list, tuple)):
            nodes = nodes[0]
        out = [n for n in nodes if n is not None and n != ig]
        for i, n in enumerate(out):
            if np.isscalar(n):
                out[i] = self.new_coef(n)
        return out

    def sum(self, *ns):
        ns = self._filter(ns, 0.0)
        return ns[0] if len(ns) == 1 else self._add_node("add", ns)

    def prod(self, *ns):
        if 0 in ns:
            return 0
        ns = self._filter(ns, 1.0)
        return ns[0] if len(ns) == 1 else self._add_node("mul", ns)

    def sum_eq0(self, *ns):
        return self._add_node("eq0", [self.sum(*ns)])

    def gen_dag(self, uf_num_mod=11):
        return build_dag_inputs(
            self.node_list, self.node_scalar, self.node_function, uf_num_mod
        )


# ============================================================================

# PDE Definitions

# ============================================================================


def create_pde1_advection_burgers(uf_num_mod):
    """
    PDE 1: Advection-Burgers equation
    u_t + (u^2)_x + (-0.3*u)_y = 0
    IC: sin(2πx) * cos(4πy)
    """
    pde = PDENodesCollector()
    u = pde.new_uf()
    u_ic = np.sin(2 * np.pi * x_fenc) * np.cos(4 * np.pi * y_fenc)
    pde.set_ic(u, u_ic, x=x_fenc, y=y_fenc)
    pde.sum_eq0(pde.dt(u), pde.dx(pde.square(u)), pde.dy(-0.3 * u))
    return pde.gen_dag(uf_num_mod=uf_num_mod)


def create_pde1_advection_burgers_ms(MSCollector, config):
    """MindSpore version of PDE 1"""
    pde = MSCollector()
    u = pde.new_uf()
    u_ic = np.sin(2 * np.pi * x_fenc) * np.cos(4 * np.pi * y_fenc)
    pde.set_ic(u, u_ic, x=x_fenc, y=y_fenc)
    pde.sum_eq0(pde.dt(u), pde.dx(pde.square(u)), pde.dy(-0.3 * u))
    return pde.gen_dag(config)


def create_pde2_heat_equation(uf_num_mod):
    """
    PDE 2: 2D Heat/Diffusion equation
    u_t - 0.05*(u_xx + u_yy) = 0
    which is: u_t - 0.05*u_xx - 0.05*u_yy = 0
    IC: exp(-((x-0.5)^2 + (y-0.5)^2) / 0.02) - Gaussian bump centered at (0.5, 0.5)
    """
    pde = PDENodesCollector()
    u = pde.new_uf()
    # Gaussian bump initial condition
    u_ic = np.exp(-((x_fenc - 0.5) ** 2 + (y_fenc - 0.5) ** 2) / 0.02)
    pde.set_ic(u, u_ic, x=x_fenc, y=y_fenc)
    # u_t - 0.05*u_xx - 0.05*u_yy = 0
    # Using: d/dx(d/dx(u)) for u_xx
    diffusion_coef = 0.05
    u_xx = pde.dx(pde.dx(u))
    u_yy = pde.dy(pde.dy(u))
    pde.sum_eq0(pde.dt(u), -diffusion_coef * u_xx, -diffusion_coef * u_yy)
    return pde.gen_dag(uf_num_mod=uf_num_mod)


def create_pde2_heat_equation_ms(MSCollector, config):
    """MindSpore version of PDE 2"""
    pde = MSCollector()
    u = pde.new_uf()
    u_ic = np.exp(-((x_fenc - 0.5) ** 2 + (y_fenc - 0.5) ** 2) / 0.02)
    pde.set_ic(u, u_ic, x=x_fenc, y=y_fenc)
    diffusion_coef = 0.05
    u_xx = pde.dx(pde.dx(u))
    u_yy = pde.dy(pde.dy(u))
    pde.sum_eq0(pde.dt(u), -diffusion_coef * u_xx, -diffusion_coef * u_yy)
    return pde.gen_dag(config)


def create_pde3_poisson(uf_num_mod):
    """
    PDE 3: 2D Poisson equation with zero Dirichlet boundary conditions
    -u_xx - u_yy = f  (equivalently: u_xx + u_yy + f = 0)
    BC: u = 0 on boundary
    Source: Gaussian bump f = exp(-((x-0.5)^2 + (y-0.5)^2) / 0.05)
    """
    pde = PDENodesCollector()
    u = pde.new_uf()
    # Source term f as a coefficient function node
    f_values = np.exp(-((x_fenc - 0.5) ** 2 + (y_fenc - 0.5) ** 2) / 0.05).astype(
        np.float32
    )
    pde._add_func(f_values, x=x_fenc, y=y_fenc)
    f_node = pde._add_node("cf")
    # Zero Dirichlet BC: u = 0 on boundary
    pde._add_func(np.zeros_like(x_fenc), x=x_fenc, y=y_fenc)
    pde._add_node("bv", [u])
    # Equation: u_xx + u_yy + f = 0
    u_xx = pde.dx(pde.dx(u))
    u_yy = pde.dy(pde.dy(u))
    pde.sum_eq0(u_xx, u_yy, f_node)
    return pde.gen_dag(uf_num_mod=uf_num_mod)


def create_pde3_poisson_ms(MSCollector, config):
    """MindSpore version of PDE 3: 2D Poisson equation."""
    pde = MSCollector()
    u = pde.new_uf()
    f_values = np.exp(-((x_fenc - 0.5) ** 2 + (y_fenc - 0.5) ** 2) / 0.05).astype(
        np.float32
    )
    pde._add_func(f_values, x=x_fenc, y=y_fenc)
    f_node = pde._add_node("cf")
    pde._add_func(np.zeros_like(x_fenc), x=x_fenc, y=y_fenc)
    pde._add_node("bv", [u])
    u_xx = pde.dx(pde.dx(u))
    u_yy = pde.dy(pde.dy(u))
    pde.sum_eq0(u_xx, u_yy, f_node)
    return pde.gen_dag(config)


def get_poisson_query_coords():
    """Query coordinates for Poisson: single t=0 snapshot on 32x32 spatial grid."""
    snap_t = np.array([0.0])
    x_plot, y_plot = np.meshgrid(
        np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij"
    )
    coord = np.stack(
        np.broadcast_arrays(
            snap_t.reshape(-1, 1, 1),
            x_plot[None],
            y_plot[None],
            np.zeros_like(x_plot)[None],
        ),
        axis=-1,
    ).astype(np.float32)
    return snap_t, x_plot, y_plot, coord


# PDE registry

PDES = [
    {
        "name": "advection_burgers",
        "latex": r"$u_t + (u^2)_x - 0.3 u_y = 0$",
        "description": "Advection-Burgers",
        "create_jax": create_pde1_advection_burgers,
        "create_ms": create_pde1_advection_burgers_ms,
    },
    {
        "name": "heat_equation",
        "latex": r"$u_t - 0.05(u_{xx} + u_{yy}) = 0$",
        "description": "2D Heat Equation",
        "create_jax": create_pde2_heat_equation,
        "create_ms": create_pde2_heat_equation_ms,
    },
    {
        "name": "poisson",
        "latex": r"$u_{xx} + u_{yy} + f = 0,\; u|_{\partial\Omega}=0$",
        "description": "2D Poisson (zero Dirichlet BC)",
        "create_jax": create_pde3_poisson,
        "create_ms": create_pde3_poisson_ms,
        "get_coords": get_poisson_query_coords,
    },
]


def get_query_coords():
    """Get query coordinates: 32x32 spatial, 5 time snapshots."""
    snap_t = np.array([0, 0.25, 0.5, 0.75, 1.0])
    x_plot, y_plot = np.meshgrid(
        np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij"
    )
    coord = np.stack(
        np.broadcast_arrays(
            snap_t.reshape(-1, 1, 1),
            x_plot[None],
            y_plot[None],
            0,
        ),
        axis=-1,
    ).astype(np.float32)
    return snap_t, x_plot, y_plot, coord


# ============================================================================

# MindSpore Inference

# ============================================================================


def run_mindspore_inference(ckpt_name, config_file, uf_num_mod, pde_config):
    """Run MindSpore inference for a given checkpoint and PDE."""
    sys.path.insert(0, PDEFORMER2_ROOT)
    original_dir = os.getcwd()
    os.chdir(PDEFORMER2_ROOT)

    try:
        from mindspore import context

        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

        from src import load_config, get_model, PDENodesCollector as MSCollector
        from src.inference import inference_pde

        config = load_config(config_file)
        config.model.load_ckpt = os.path.join(DATA_DIR, f"pdeformer2-{ckpt_name}.ckpt")

        model = get_model(config)

        # Build PDE using MindSpore's collector
        pde_dag = pde_config["create_ms"](MSCollector, config)

        _get_coords = pde_config.get("get_coords", get_query_coords)
        snap_t, x_plot, y_plot, coord = _get_coords()

        u_pred = inference_pde(model, pde_dag, coord)
        u_pred = u_pred[..., 0]

        return np.array(u_pred).astype(np.float32)
    finally:
        os.chdir(original_dir)


# ============================================================================

# JAX Inference

# ============================================================================


def run_jax_inference(ckpt_name, uf_num_mod, pde_config):
    """Run JAX inference for a given checkpoint and PDE."""
    import jax
    import jax.numpy as jnp
    from flax.core import freeze

    from jax_pdeformer2 import create_pdeformer_from_config
    from jax_pdeformer2 import (
        PDEFORMER_SMALL_CONFIG,
        PDEFORMER_BASE_CONFIG,
        PDEFORMER_FAST_CONFIG,
    )
    from jax_pdeformer2.utils import load_mindspore_checkpoint, convert_mindspore_to_jax

    config_map = {
        "small": PDEFORMER_SMALL_CONFIG,
        "base": PDEFORMER_BASE_CONFIG,
        "fast": PDEFORMER_FAST_CONFIG,
    }

    ckpt_path = os.path.join(DATA_DIR, f"pdeformer2-{ckpt_name}.ckpt")
    model_config = config_map[ckpt_name]

    model = create_pdeformer_from_config({"model": model_config})
    ms_weights = load_mindspore_checkpoint(ckpt_path)
    jax_params = convert_mindspore_to_jax(ms_weights)

    def to_jax(d):
        if isinstance(d, dict):
            return {k: to_jax(v) for k, v in d.items()}
        if isinstance(d, np.ndarray):
            return jnp.array(d)
        return d

    params = freeze(to_jax(jax_params))

    pde_dag = pde_config["create_jax"](uf_num_mod)
    # Use PDE-specific coordinate function if provided (e.g. stationary problems)
    _get_coords = pde_config.get("get_coords", get_query_coords)
    snap_t, x_plot, y_plot, coord = _get_coords()
    coordinate = coord.reshape(-1, 4)

    jax_inputs = {k: jnp.array(v)[None] for k, v in pde_dag.items()}
    jax_inputs["coordinate"] = jnp.array(coordinate)[None]

    output = model.apply(params, **jax_inputs)
    u_pred = np.array(output)[0, :, 0].reshape(coord.shape[:-1])

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    return u_pred.astype(np.float32), n_params


# ============================================================================

# Plotting

# ============================================================================


def plot_comparison(
    ms_pred, jax_pred, abs_err, ckpt_name, pde_name, pde_latex, snap_t, x_plot, y_plot
):
    """Create comparison plots for a single checkpoint and PDE."""
    n_times = len(snap_t)

    # Determine common color scale for predictions
    vmin = min(ms_pred.min(), jax_pred.min())
    vmax = max(ms_pred.max(), jax_pred.max())

    # Create figure: 3 rows (MindSpore, JAX, Error) x n_times columns
    fig, axes = plt.subplots(3, n_times, figsize=(4 * n_times, 10))
    fig.suptitle(
        f"PDEFormer2-{ckpt_name}: MindSpore vs JAX Comparison\n" f"PDE: {pde_latex}",
        fontsize=14,
    )

    for i, t in enumerate(snap_t):
        # MindSpore prediction
        im0 = axes[0, i].pcolormesh(
            x_plot,
            y_plot,
            ms_pred[i],
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
            shading="auto",
        )
        axes[0, i].set_title(f"t = {t:.2f}")
        axes[0, i].set_aspect("equal")
        if i == 0:
            axes[0, i].set_ylabel("MindSpore")

        # JAX prediction
        im1 = axes[1, i].pcolormesh(
            x_plot,
            y_plot,
            jax_pred[i],
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
            shading="auto",
        )
        axes[1, i].set_aspect("equal")
        if i == 0:
            axes[1, i].set_ylabel("JAX")

        # Absolute error
        im2 = axes[2, i].pcolormesh(
            x_plot, y_plot, abs_err[i], cmap="hot_r", shading="auto"
        )
        axes[2, i].set_aspect("equal")
        if i == 0:
            axes[2, i].set_ylabel("Abs Error")
        axes[2, i].set_xlabel("x")

    # Add colorbars
    fig.colorbar(im0, ax=axes[0, :], shrink=0.8, label="u (MindSpore)")
    fig.colorbar(im1, ax=axes[1, :], shrink=0.8, label="u (JAX)")
    fig.colorbar(im2, ax=axes[2, :], shrink=0.8, label="|MS - JAX|")

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / f"comparison_{ckpt_name}_{pde_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Saved: {output_path}")
    return output_path


def plot_summary(results, snap_t, x_plot, y_plot):
    """Create a summary plot comparing all checkpoints for each PDE.

    When MindSpore predictions are present the layout is:
        rows = checkpoints × [MindSpore | JAX | Abs Error],  cols = time steps
    When JAX-only the layout is:
        rows = checkpoints,  cols = time steps
    """
    if not results:
        return

    for pde_name, pde_results in results.items():
        if not pde_results:
            continue

        n_ckpts = len(pde_results)
        n_times = len(snap_t)
        has_ms = any("ms_pred" in d for d in pde_results.values())

        # Get PDE latex from first result
        pde_latex = list(pde_results.values())[0].get("pde_latex", pde_name)

        if has_ms:
            # 3 sub-rows per checkpoint: MS | JAX | Error
            n_rows = n_ckpts * 3
            fig, axes = plt.subplots(
                n_rows,
                n_times,
                figsize=(4 * n_times, 3.5 * n_ckpts * 3),
                squeeze=False,
            )
            fig.suptitle(
                f"PDEFormer2 MindSpore vs JAX Across Checkpoints\nPDE: {pde_latex}",
                fontsize=14,
            )

            # Common prediction colour scale
            all_preds = [d["jax_pred"] for d in pde_results.values()]
            all_preds += [d["ms_pred"] for d in pde_results.values() if "ms_pred" in d]
            vmin = min(p.min() for p in all_preds)
            vmax = max(p.max() for p in all_preds)

            last_err_im = None
            for ckpt_idx, (ckpt_name, data) in enumerate(pde_results.items()):
                row_ms = ckpt_idx * 3
                row_jax = ckpt_idx * 3 + 1
                row_err = ckpt_idx * 3 + 2
                label = f"{ckpt_name}\n({data['n_params']:,})"

                for col, t in enumerate(snap_t):
                    # MindSpore row
                    if "ms_pred" in data:
                        axes[row_ms, col].pcolormesh(
                            x_plot,
                            y_plot,
                            data["ms_pred"][col],
                            vmin=vmin,
                            vmax=vmax,
                            cmap="RdBu_r",
                            shading="auto",
                        )
                    axes[row_ms, col].set_aspect("equal")
                    if ckpt_idx == 0:
                        axes[row_ms, col].set_title(f"t = {t:.2f}")
                    if col == 0:
                        axes[row_ms, col].set_ylabel(f"{label}\nMindSpore")

                    # JAX row
                    im_jax = axes[row_jax, col].pcolormesh(
                        x_plot,
                        y_plot,
                        data["jax_pred"][col],
                        vmin=vmin,
                        vmax=vmax,
                        cmap="RdBu_r",
                        shading="auto",
                    )
                    axes[row_jax, col].set_aspect("equal")
                    if col == 0:
                        axes[row_jax, col].set_ylabel(f"{label}\nJAX")

                    # Error row
                    if "abs_err" in data:
                        last_err_im = axes[row_err, col].pcolormesh(
                            x_plot,
                            y_plot,
                            data["abs_err"][col],
                            cmap="hot_r",
                            shading="auto",
                        )
                    axes[row_err, col].set_aspect("equal")
                    if col == 0:
                        axes[row_err, col].set_ylabel(f"{label}\nAbs Error")
                    if ckpt_idx == n_ckpts - 1:
                        axes[row_err, col].set_xlabel("x")

            fig.colorbar(im_jax, ax=axes[:, :], shrink=0.4, label="u", pad=0.01)
            if last_err_im is not None:
                fig.colorbar(
                    last_err_im, ax=axes[:, :], shrink=0.4, label="|MS − JAX|", pad=0.05
                )

        else:
            # JAX-only: rows = checkpoints, cols = time steps
            fig, axes = plt.subplots(
                n_ckpts,
                n_times,
                figsize=(4 * n_times, 3.5 * n_ckpts),
                squeeze=False,
            )
            fig.suptitle(
                f"PDEFormer2 JAX Predictions Across Checkpoints\nPDE: {pde_latex}",
                fontsize=14,
            )
            all_preds = [data["jax_pred"] for data in pde_results.values()]
            vmin = min(p.min() for p in all_preds)
            vmax = max(p.max() for p in all_preds)

            for row, (ckpt_name, data) in enumerate(pde_results.items()):
                jax_pred = data["jax_pred"]
                for col, t in enumerate(snap_t):
                    im = axes[row, col].pcolormesh(
                        x_plot,
                        y_plot,
                        jax_pred[col],
                        vmin=vmin,
                        vmax=vmax,
                        cmap="RdBu_r",
                        shading="auto",
                    )
                    axes[row, col].set_aspect("equal")
                    if row == 0:
                        axes[row, col].set_title(f"t = {t:.2f}")
                    if col == 0:
                        axes[row, col].set_ylabel(
                            f"{ckpt_name}\n({data['n_params']:,} params)"
                        )
                    if row == n_ckpts - 1:
                        axes[row, col].set_xlabel("x")

            fig.colorbar(im, ax=axes, shrink=0.6, label="u")

        plt.tight_layout()
        output_path = OUTPUT_DIR / f"comparison_summary_{pde_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved summary: {output_path}")


def plot_error_summary(results, snap_t):
    """Create error comparison plot across checkpoints for each PDE."""
    if not results:
        return

    n_pdes = len(results)
    fig, axes = plt.subplots(n_pdes, 2, figsize=(12, 5 * n_pdes))

    if n_pdes == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("MindSpore vs JAX Error Comparison", fontsize=14)

    for pde_idx, (pde_name, pde_results) in enumerate(results.items()):
        if not pde_results:
            continue

        pde_latex = list(pde_results.values())[0].get("pde_latex", pde_name)

        ckpt_names = list(pde_results.keys())
        x_pos = np.arange(len(snap_t))
        width = 0.25

        colors = plt.cm.tab10(np.linspace(0, 1, len(ckpt_names)))

        # Plot max and mean absolute error per timestep
        for i, (ckpt_name, data) in enumerate(pde_results.items()):
            max_errs = [data["abs_err"][j].max() for j in range(len(snap_t))]
            mean_errs = [data["abs_err"][j].mean() for j in range(len(snap_t))]

            offset = (i - len(ckpt_names) / 2 + 0.5) * width
            axes[pde_idx, 0].bar(
                x_pos + offset, max_errs, width, label=ckpt_name, color=colors[i]
            )
            axes[pde_idx, 1].bar(
                x_pos + offset, mean_errs, width, label=ckpt_name, color=colors[i]
            )

        axes[pde_idx, 0].set_xlabel("Time")
        axes[pde_idx, 0].set_ylabel("Max Absolute Error")
        axes[pde_idx, 0].set_title(f"Maximum Error - {pde_latex}")
        axes[pde_idx, 0].set_xticks(x_pos)
        axes[pde_idx, 0].set_xticklabels([f"t={t:.2f}" for t in snap_t])
        axes[pde_idx, 0].legend()
        axes[pde_idx, 0].set_yscale("log")

        axes[pde_idx, 1].set_xlabel("Time")
        axes[pde_idx, 1].set_ylabel("Mean Absolute Error")
        axes[pde_idx, 1].set_title(f"Mean Error - {pde_latex}")
        axes[pde_idx, 1].set_xticks(x_pos)
        axes[pde_idx, 1].set_xticklabels([f"t={t:.2f}" for t in snap_t])
        axes[pde_idx, 1].legend()
        axes[pde_idx, 1].set_yscale("log")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "error_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Saved error summary: {output_path}")


# ============================================================================

# Comparison

# ============================================================================


def compare_predictions(ms_pred, jax_pred, ckpt_name, pde_name, snap_t):
    """Compare MindSpore and JAX predictions."""
    abs_err = np.abs(ms_pred - jax_pred)
    rel_err = abs_err / (np.abs(ms_pred) + 1e-8)

    print(f"\n{'='*60}")
    print(f"Results for: pdeformer2-{ckpt_name} | PDE: {pde_name}")
    print(f"{'='*60}")
    print(f"MindSpore prediction shape: {ms_pred.shape}")
    print(f"JAX prediction shape:       {jax_pred.shape}")
    print(f"MindSpore range:  [{ms_pred.min():.6f}, {ms_pred.max():.6f}]")
    print(f"JAX range:        [{jax_pred.min():.6f}, {jax_pred.max():.6f}]")
    print(
        f"\nOverall absolute error:  max={abs_err.max():.2e}  mean={abs_err.mean():.2e}"
    )
    print(
        f"Overall relative error:  max={rel_err.max():.2e}  mean={rel_err.mean():.2e}"
    )
    print(f"\nPer-timestep errors:")
    for i, t in enumerate(snap_t):
        ae = abs_err[i]
        print(f"  t={t:.2f}:  max_abs_err={ae.max():.2e}  mean_abs_err={ae.mean():.2e}")

    return abs_err, rel_err


# ============================================================================

# Main

# ============================================================================


def main():
    print("=" * 60)
    print("PDEFormer2: MindSpore vs JAX Comparison")
    print("=" * 60)
    print("\nPDEs to test:")
    for pde in PDES:
        print(f"  - {pde['description']}: {pde['latex']}")
    print()

    # Results organized by PDE -> checkpoint
    all_results = {pde["name"]: {} for pde in PDES}
    _coords_cache = {}  # cache per-PDE coords for plotting

    for pde_config in PDES:
        pde_name = pde_config["name"]
        pde_latex = pde_config["latex"]
        pde_desc = pde_config["description"]

        print("\n" + "#" * 60)
        print(f"# PDE: {pde_desc}")
        print(f"# {pde_latex}")
        print("#" * 60)

        _get_coords = pde_config.get("get_coords", get_query_coords)
        pde_snap_t, pde_x_plot, pde_y_plot, _ = _get_coords()
        _coords_cache[pde_name] = (pde_snap_t, pde_x_plot, pde_y_plot)

        for ckpt_name, config_file, uf_num_mod in CHECKPOINTS:
            ckpt_path = os.path.join(DATA_DIR, f"pdeformer2-{ckpt_name}.ckpt")

            if not os.path.exists(ckpt_path):
                print(f"\n[SKIP] Checkpoint not found: {ckpt_path}")
                continue

            print(f"\n{'='*60}")
            print(f"Testing: pdeformer2-{ckpt_name} | {pde_desc}")
            print(f"{'='*60}")

            # Skip PDEs with no MindSpore equivalent in MS-only mode
            if pde_config.get("create_ms") is None:
                print(
                    f"[INFO] Skipping MindSpore inference for {pde_desc} (not supported)"
                )

            # JAX inference
            print(f"\n[JAX] Running inference...")
            try:
                jax_pred, n_params = run_jax_inference(
                    ckpt_name, uf_num_mod, pde_config
                )
                print(f"[JAX] Model parameters: {n_params:,}")
                print(f"[JAX] Prediction shape: {jax_pred.shape}")
                print(
                    f"[JAX] Value range: [{jax_pred.min():.6f}, {jax_pred.max():.6f}]"
                )
            except Exception as e:
                print(f"[JAX] Error: {e}")
                import traceback

                traceback.print_exc()
                continue

            if pde_config.get("create_ms") is not None:
                # MindSpore inference
                print(f"\n[MindSpore] Running inference...")
                try:
                    ms_pred = run_mindspore_inference(
                        ckpt_name, config_file, uf_num_mod, pde_config
                    )
                    print(f"[MindSpore] Prediction shape: {ms_pred.shape}")
                    print(
                        f"[MindSpore] Value range: [{ms_pred.min():.6f}, {ms_pred.max():.6f}]"
                    )
                except Exception as e:
                    print(f"[MindSpore] Error: {e}")
                    import traceback

                    traceback.print_exc()
                    ms_pred = None
            else:
                ms_pred = None

            if ms_pred is not None:
                # Compare
                abs_err, rel_err = compare_predictions(
                    ms_pred, jax_pred, ckpt_name, pde_desc, pde_snap_t
                )
                all_results[pde_name][ckpt_name] = {
                    "ms_pred": ms_pred,
                    "jax_pred": jax_pred,
                    "abs_err": abs_err,
                    "rel_err": rel_err,
                    "n_params": n_params,
                    "pde_latex": pde_latex,
                }
                # Generate individual comparison plot
                plot_comparison(
                    ms_pred,
                    jax_pred,
                    abs_err,
                    ckpt_name,
                    pde_name,
                    pde_latex,
                    pde_snap_t,
                    pde_x_plot,
                    pde_y_plot,
                )
            else:
                # JAX-only result
                all_results[pde_name][ckpt_name] = {
                    "jax_pred": jax_pred,
                    "n_params": n_params,
                    "pde_latex": pde_latex,
                }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for pde_config in PDES:
        pde_name = pde_config["name"]
        pde_results = all_results[pde_name]

        if not pde_results:
            continue

        print(f"\n{pde_config['description']}: {pde_config['latex']}")
        print("-" * 72)
        if any("abs_err" in d for d in pde_results.values()):
            print(
                f"{'Checkpoint':<12} {'Params':<15} {'Max Abs Err':<15} {'Mean Abs Err':<15} {'Max Rel Err':<15}"
            )
            print("-" * 72)
            for name, data in pde_results.items():
                if "abs_err" in data:
                    print(
                        f"{name:<12} {data['n_params']:<15,} {data['abs_err'].max():<15.2e} "
                        f"{data['abs_err'].mean():<15.2e} {data['rel_err'].max():<15.2e}"
                    )
        else:
            print(f"{'Checkpoint':<12} {'Params':<15} {'Output range'}")
            print("-" * 40)
            for name, data in pde_results.items():
                pred = data["jax_pred"]
                print(
                    f"{name:<12} {data['n_params']:<15,} [{pred.min():.4f}, {pred.max():.4f}]"
                )

    # Generate summary plots
    print("\n[Plotting] Generating summary plots...")
    for pde_config in PDES:
        pde_name = pde_config["name"]
        if pde_name not in _coords_cache:
            continue
        pde_snap_t, pde_x_plot, pde_y_plot = _coords_cache[pde_name]
        plot_summary(
            {pde_name: all_results[pde_name]}, pde_snap_t, pde_x_plot, pde_y_plot
        )
        # Only plot error comparison when MindSpore results are available
        if any("abs_err" in d for d in all_results[pde_name].values()):
            plot_error_summary({pde_name: all_results[pde_name]}, pde_snap_t)

    print("\nDone.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
