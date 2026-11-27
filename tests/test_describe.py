"""fx.describe: the module tree of any model and the dataflow graph of a composed one. Pure introspection."""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foundax as fx

K = jax.random.PRNGKey(0)


def lin(i, o, k=0):
    return eqx.nn.Linear(i, o, key=jax.random.PRNGKey(k))


def n_params(m):
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array)))


def n_bytes(m):
    return sum(
        x.size * x.dtype.itemsize
        for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))
    )


def find(node, path):
    if node["path"] == path:
        return node
    for c in node["children"]:
        r = find(c, path)
        if r:
            return r


# ---- the tree -------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        lin(3, 5),
        fx.mlp(2, output_dim=1, hidden_dims=8, num_layers=2, key=K),
        eqx.nn.MLP(4, 2, 6, 3, key=K),
    ],
    ids=["linear", "foundax-mlp", "eqx-mlp"],
)
def test_the_summary_matches_the_arrays_of_the_model(model):
    d = fx.describe(model)
    assert d["summary"]["total_parameters"] == n_params(model) and d["summary"][
        "total_bytes"
    ] == n_bytes(model)
    assert d["summary"]["leaf_count"] == len(
        jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    assert (
        d["tree"]["param_count"] == n_params(model)
        and d["type"] == type(model).__name__
    )


def test_every_leaf_has_its_path_shape_dtype_count_and_bytes_and_parents_add_up():
    d = fx.describe(fx.mlp(2, output_dim=1, hidden_dims=8, num_layers=2, key=K))
    out = find(d["tree"], "output_layer/weight")
    assert out == {
        "name": "weight",
        "path": "output_layer/weight",
        "kind": "array",
        "type": out["type"],
        "shape": [1, 8],
        "dtype": "float32",
        "param_count": 8,
        "bytes": 32,
        "children": [],
    }
    assert find(d["tree"], "hidden_layers/0/weight")["shape"] == [8, 2]
    hidden = find(d["tree"], "hidden_layers")
    assert hidden["kind"] == "list" and hidden["param_count"] == (8 * 2 + 8) + (
        8 * 8 + 8
    )

    def sums(n):
        for c in n["children"]:
            sums(c)
        if n["children"]:
            assert n["param_count"] == sum(
                c["param_count"] for c in n["children"]
            ) and n["bytes"] == sum(c["bytes"] for c in n["children"])

    sums(d["tree"])


def test_plain_static_fields_are_kept_and_functions_are_not():
    d = fx.describe(lin(3, 5))
    assert d["tree"]["static"] == {
        "in_features": 3,
        "out_features": 5,
        "use_bias": True,
    }
    m = fx.mlp(2, output_dim=1, hidden_dims=4, num_layers=1, key=K)
    assert "activation" not in [c["name"] for c in fx.describe(m)["tree"]["children"]]


def test_the_result_is_plain_json_and_the_model_is_not_changed():
    m = fx.mlp(2, hidden_dims=4, num_layers=1, key=K)
    before = [
        np.asarray(x).copy()
        for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))
    ]
    d = fx.describe(m)
    assert json.loads(json.dumps(d)) == d
    assert all(
        np.array_equal(a, np.asarray(b))
        for a, b in zip(before, jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array)))
    )


def test_a_list_or_dict_of_modules_is_described_and_something_else_is_refused():
    assert fx.describe([lin(2, 3), lin(3, 1)])["summary"]["total_parameters"] == (
        2 * 3 + 3
    ) + (3 + 1)
    assert fx.describe({"a": lin(2, 3)})["tree"]["children"][0]["path"] == "a"
    with pytest.raises(TypeError, match="needs an equinox module"):
        fx.describe(3.0)


# ---- the graph ------------------------------------------------------------------------------------------------------------------------


def edges(d):
    labels = {n["id"]: n["label"] for n in d["graph"]["nodes"]}
    return sorted(
        (labels[e["source"]], labels[e["target"]]) for e in d["graph"]["edges"]
    )


def test_a_plain_module_has_no_graph_only_a_tree():
    assert (
        fx.describe(lin(2, 3))["graph"] is None
        and fx.describe(fx.mlp(2, key=K))["graph"] is None
    )


def test_a_single_block_is_input_block_output_with_its_channels():
    d = fx.describe(fx.block(lin(3, 8), name="enc"))
    kinds = [n["kind"] for n in d["graph"]["nodes"]]
    assert kinds == ["block", "input", "output"]
    blk = d["graph"]["nodes"][0]
    assert (blk["label"], blk["in_channels"], blk["out_channels"], blk["type"]) == (
        "enc",
        3,
        8,
        "Linear",
    )


def test_a_pipe_is_a_chain_and_the_edges_carry_the_channels():
    p = (
        fx.block(lin(3, 8), name="a")
        | fx.block(lin(8, 16), name="b")
        | fx.block(lin(16, 1), name="c")
    )
    d = fx.describe(p)
    assert edges(d) == sorted([("a", "b"), ("b", "c"), ("c", "output"), ("input", "a")])
    by = {(e["source"], e["target"]): e["label"] for e in d["graph"]["edges"]}
    assert [e["label"] for e in d["graph"]["edges"] if e["label"]] == [
        "8",
        "16",
        "1",
    ] and len(by) == 4
    assert d["summary"]["total_parameters"] == n_params(p)


def test_add_and_cat_branch_from_one_input_and_join_in_one_node():
    a, b = fx.block(lin(4, 4, 1), name="a"), fx.block(lin(4, 4, 2), name="b")
    for make, kind in ((fx.add, "add"), (fx.cat, "cat")):
        d = fx.describe(make(a, b))
        assert edges(d) == sorted(
            [("a", kind), ("b", kind), (kind, "output"), ("input", "a"), ("input", "b")]
        )
        assert sum(n["kind"] == "input" for n in d["graph"]["nodes"]) == 1


def test_a_dot_has_two_inputs_and_a_pipe_for_a_branch_is_expanded():
    branch = fx.block(lin(3, 8), name="b1") | fx.block(lin(8, 4), name="b2")
    d = fx.describe(fx.dot(branch, fx.block(lin(2, 4), name="t1")))
    assert edges(d) == sorted(
        [
            ("b1", "b2"),
            ("b2", "dot"),
            ("dot", "output"),
            ("t1", "dot"),
            ("u", "b1"),
            ("y", "t1"),
        ]
    )


def test_nested_compositions_and_a_block_around_a_pipe():
    inner = fx.block(lin(4, 4, 1), name="x") | fx.block(lin(4, 4, 2), name="y")
    d = fx.describe(
        fx.add(
            inner,
            fx.block(
                fx.block(lin(4, 4, 3), name="z") | fx.block(lin(4, 4, 4), name="w"),
                name="wrapped",
            ),
        )
    )
    assert edges(d) == sorted(
        [
            ("input", "x"),
            ("input", "z"),
            ("w", "add"),
            ("y", "add"),
            ("add", "output"),
            ("x", "y"),
            ("z", "w"),
        ]
    )
    labels = sorted(n["label"] for n in d["graph"]["nodes"])
    assert labels == [
        "add",
        "input",
        "output",
        "w",
        "x",
        "y",
        "z",
    ]  # the wrapper is the composition it holds
    assert (
        ("input", "x") in edges(d)
        and ("input", "z") in edges(d)
        and ("x", "y") in edges(d)
        and ("z", "w") in edges(d)
    )


def test_the_graph_json_roundtrips_and_node_ids_are_unique():
    d = fx.describe(
        fx.add(fx.block(lin(4, 4, 1)) | fx.block(lin(4, 4, 2)), fx.block(lin(4, 4, 3)))
    )
    assert json.loads(json.dumps(d)) == d
    ids = [n["id"] for n in d["graph"]["nodes"]]
    assert len(ids) == len(set(ids)) and all(
        e["source"] in ids and e["target"] in ids for e in d["graph"]["edges"]
    )


def test_the_described_structure_is_the_one_the_model_computes():
    """The graph is read from the composition; check it against a real forward pass: a Pipe applies its blocks in order, add sums, cat concatenates."""
    a, b = fx.block(lin(4, 3, 1), name="a"), fx.block(lin(4, 3, 2), name="b")
    x = jnp.ones(4)
    assert fx.add(a, b)(x).shape == (3,) and fx.cat(a, b)(x).shape == (6,)
    d = fx.describe(fx.cat(a, b))
    assert [n["out_channels"] for n in d["graph"]["nodes"] if n["kind"] == "block"] == [
        3,
        3,
    ]
