import numpy as np
import torch
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, Any


def generate_smooth_input(
    H: int, W: int, num_channels: int, seed: int, num_modes: int = 8
) -> np.ndarray:
    """
    Generate smooth input using sum of sinusoidal functions.
    Creates physically plausible smooth fields.
    """
    np.random.seed(seed)

    x = np.linspace(0, 2 * np.pi, W)
    y = np.linspace(0, 2 * np.pi, H)
    X, Y = np.meshgrid(x, y)

    result = np.zeros((1, H, W, num_channels), dtype=np.float32)

    for c in range(num_channels):
        field = np.zeros((H, W), dtype=np.float32)
        for _ in range(num_modes):
            kx = np.random.randint(1, 5)
            ky = np.random.randint(1, 5)
            amplitude = np.random.uniform(0.5, 2.0)
            phase_x = np.random.uniform(0, 2 * np.pi)
            phase_y = np.random.uniform(0, 2 * np.pi)
            field += amplitude * np.sin(kx * X + phase_x) * np.cos(ky * Y + phase_y)

        field = (field - field.mean()) / (field.std() + 1e-8)
        result[0, :, :, c] = field

    return result


def compute_l2_difference(pt_output: np.ndarray, jax_output: np.ndarray) -> float:
    """Compute the L2 (Euclidean) norm of the difference between outputs."""
    diff = pt_output.flatten() - jax_output.flatten()
    return float(np.sqrt(np.sum(diff**2)))


def compute_relative_l2_difference(
    pt_output: np.ndarray, jax_output: np.ndarray
) -> float:
    """Compute the relative L2 difference (normalized by PT output norm)."""
    diff = pt_output.flatten() - jax_output.flatten()
    l2_diff = np.sqrt(np.sum(diff**2))
    l2_pt = np.sqrt(np.sum(pt_output.flatten() ** 2))
    return float(l2_diff / (l2_pt + 1e-8))


def run_single_comparison(
    pt_model,
    jax_model,
    jax_params: Dict[str, Any],
    input_np: np.ndarray,
    time_np: np.ndarray,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run a single forward pass comparison and return metrics."""
    # PyTorch forward pass
    dummy_input_pt = torch.from_numpy(input_np).permute(0, 3, 1, 2).to(device)
    time_pt = torch.from_numpy(time_np).to(device)

    with torch.no_grad():
        pt_output = pt_model(pixel_values=dummy_input_pt, time=time_pt)
        pt_output_tensor = pt_output.output

    pt_output_np = pt_output_tensor.cpu().numpy().transpose(0, 2, 3, 1)

    # JAX forward pass
    dummy_input_jax = jnp.array(input_np)
    time_jax = jnp.array(time_np)

    jax_out = jax_model.apply(
        jax_params,
        pixel_values=dummy_input_jax,
        time=time_jax,
        deterministic=True,
    )
    jax_output = np.array(jax_out.output)

    if jax_output.ndim == 3:
        jax_output_np = jax_output[None, ...]
    else:
        jax_output_np = jax_output

    # Compute metrics
    abs_diff = np.abs(pt_output_np - jax_output_np)
    l2_diff = compute_l2_difference(pt_output_np, jax_output_np)
    rel_l2_diff = compute_relative_l2_difference(pt_output_np, jax_output_np)

    return {
        "pt_output": pt_output_np,
        "jax_output": jax_output_np,
        "l2_diff": l2_diff,
        "rel_l2_diff": rel_l2_diff,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "time": float(time_np[0]),
        "pt_min": float(pt_output_np.min()),
        "pt_max": float(pt_output_np.max()),
        "pt_mean": float(pt_output_np.mean()),
        "jax_min": float(jax_output_np.min()),
        "jax_max": float(jax_output_np.max()),
        "jax_mean": float(jax_output_np.mean()),
    }


def _plot_comparison(
    pt_output_np: np.ndarray,
    jax_output_np: np.ndarray,
    input_np: np.ndarray,
    filename: str,
    time_val: float = None,
    sample_idx: int = None,
):
    """Generate comparison plot with colorbars including input."""
    pt0 = pt_output_np[0]
    jax0 = jax_output_np[0]
    inp0 = input_np[0]
    diff = np.abs(jax0 - pt0)

    vmin_out = min(pt0.min(), jax0.min())
    vmax_out = max(pt0.max(), jax0.max())
    vmin_in = inp0.min()
    vmax_in = inp0.max()
    dmin = 0.0
    dmax = diff.max()

    fig, ax = plt.subplots(4, 4, figsize=(16, 14), constrained_layout=True)

    cmap_main = "viridis"
    cmap_diff = "magma"

    # Row 0: Input
    for ch in range(4):
        im_in = ax[0, ch].imshow(
            inp0[:, :, ch], cmap=cmap_main, vmin=vmin_in, vmax=vmax_in
        )
        ax[0, ch].set_title(f"Input – ch {ch}")

    # Row 1: PyTorch output
    for ch in range(4):
        im_pt = ax[1, ch].imshow(
            pt0[:, :, ch], cmap=cmap_main, vmin=vmin_out, vmax=vmax_out
        )
        ax[1, ch].set_title(f"PyTorch – ch {ch}")

    # Row 2: JAX output
    for ch in range(4):
        im_jax = ax[2, ch].imshow(
            jax0[:, :, ch], cmap=cmap_main, vmin=vmin_out, vmax=vmax_out
        )
        ax[2, ch].set_title(f"JAX – ch {ch}")

    # Row 3: Absolute difference
    for ch in range(4):
        im_diff = ax[3, ch].imshow(diff[:, :, ch], cmap=cmap_diff, vmin=dmin, vmax=dmax)
        ax[3, ch].set_title(f"|JAX − PT| – ch {ch}")

    # Hide ticks
    for row in range(4):
        for col in range(4):
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])

    # Colorbars
    cbar_in = fig.colorbar(im_in, ax=ax[0, :], fraction=0.02, pad=0.02)
    cbar_in.set_label("Input value")

    cbar_out = fig.colorbar(im_pt, ax=ax[1:3, :], fraction=0.02, pad=0.02)
    cbar_out.set_label("Output value (PT/JAX)")

    cbar_diff = fig.colorbar(im_diff, ax=ax[3, :], fraction=0.02, pad=0.02)
    cbar_diff.set_label("Absolute difference")

    title = "PyTorch vs JAX Output Comparison"
    if sample_idx is not None:
        title += f" (Sample {sample_idx})"
    if time_val is not None:
        title += f" (t={time_val:.2f})"
    fig.suptitle(title, fontsize=16)

    plt.savefig(filename, dpi=150)
    plt.close(fig)


def _plot_summary(results: list, filename: str):
    """Generate summary plots for all comparison results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    l2_diffs = [r["l2_diff"] for r in results]
    rel_l2_diffs = [r["rel_l2_diff"] for r in results]
    times = [r["time"] for r in results]
    sample_indices = [r["sample_idx"] for r in results]

    # L2 difference histogram
    ax = axes[0, 0]
    ax.hist(l2_diffs, bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel("L2 Difference")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of L2 Differences")
    ax.axvline(
        np.mean(l2_diffs),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(l2_diffs):.2e}",
    )
    ax.legend()

    # Relative L2 difference histogram
    ax = axes[0, 1]
    ax.hist(rel_l2_diffs, bins=20, edgecolor="black", alpha=0.7, color="orange")
    ax.set_xlabel("Relative L2 Difference")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Relative L2 Differences")
    ax.axvline(
        np.mean(rel_l2_diffs),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(rel_l2_diffs):.2e}",
    )
    ax.axvline(1e-3, color="g", linestyle=":", label="Tolerance (1e-3)")
    ax.legend()

    # L2 vs time
    ax = axes[1, 0]
    ax.scatter(times, l2_diffs, alpha=0.6)
    ax.set_xlabel("Time Value")
    ax.set_ylabel("L2 Difference")
    ax.set_title("L2 Difference vs Time")

    # L2 vs sample index
    ax = axes[1, 1]
    ax.plot(sample_indices, l2_diffs, "o-", alpha=0.6, markersize=3)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("L2 Difference")
    ax.set_title("L2 Difference Across Samples")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"\n  Summary plot saved to: {filename}")


def verify_model_outputs(
    pt_model, jax_model, jax_params: Dict[str, Any], device: str = "cpu"
):
    """
    Verify that PyTorch and JAX models produce the same output with a single example.
    """
    print("\n" + "=" * 80)
    print("MODEL OUTPUT VERIFICATION (Single Example)")
    print("=" * 80)

    dummy_input_np = generate_smooth_input(128, 128, 4, seed=42)
    time_np = np.array([0.05], dtype=np.float32)

    result = run_single_comparison(
        pt_model, jax_model, jax_params, dummy_input_np, time_np, device
    )

    pt_output_np = result["pt_output"]
    jax_output_np = result["jax_output"]
    correlation = np.corrcoef(pt_output_np.flatten(), jax_output_np.flatten())[0, 1]

    print(f"\n--- Metrics ---")
    print(f"  Relative L2 Difference:   {result['rel_l2_diff']:.6e}")
    print(f"  Max Absolute Diff:        {result['max_abs_diff']:.6e}")
    print(f"  Mean Absolute Diff:       {result['mean_abs_diff']:.6e}")
    print(f"  Correlation coefficient:  {correlation:.6f}")

    _plot_comparison(
        pt_output_np,
        jax_output_np,
        dummy_input_np,
        "pt_jax_poseidonT_comparison.png",
        time_val=time_np[0],
    )

    return result


def run_comprehensive_comparison(
    pt_model,
    jax_model,
    jax_params: Dict[str, Any],
    num_samples: int = 20,
    num_plots: int = 5,
    device: str = "cpu",
):
    """
    Run comprehensive comparison with multiple samples.

    Args:
        pt_model: PyTorch model
        jax_model: JAX model
        jax_params: JAX parameters
        num_samples: Total number of samples to test (default: 20)
        num_plots: Number of samples to generate plots for (default: 5)
        device: Device for PyTorch model

    Returns:
        Dictionary with aggregated results
    """
    print("\n" + "=" * 80)
    print(f"COMPREHENSIVE COMPARISON ({num_samples} samples)")
    print("=" * 80)

    H, W = 128, 128
    MODEL_C = 4

    time_values = np.linspace(0.0, 0.1, 11)

    results = []
    plot_indices = set(np.linspace(0, num_samples - 1, num_plots, dtype=int).tolist())

    print(f"\nRunning {num_samples} comparisons...")
    print(f"  Grid size: {H}x{W}")
    print(f"  Time values: {len(time_values)} values from 0.0 to 1.0")
    print(f"  Plots will be generated for samples: {sorted(plot_indices)}")
    print()

    print(f"{'Idx':>4} | {'Time':>5} -> {'Rel L2':>12}")
    print("-" * 40)

    for sample_idx in range(num_samples):
        seed = 42 + sample_idx

        time_val = time_values[sample_idx % len(time_values)]

        if sample_idx >= len(time_values):
            np.random.seed(seed + 1000)
            time_val = np.random.uniform(0.0, 1.0)

        dummy_input_np = generate_smooth_input(H, W, MODEL_C, seed=seed)
        time_np = np.array([time_val], dtype=np.float32)

        try:
            result = run_single_comparison(
                pt_model, jax_model, jax_params, dummy_input_np, time_np, device
            )
            result["sample_idx"] = sample_idx
            result["seed"] = seed
            result["input"] = dummy_input_np
            results.append(result)

            if sample_idx in plot_indices:
                plot_filename = (
                    f"comparison_sample_{sample_idx:03d}_t{time_val:.2f}.png"
                )
                _plot_comparison(
                    result["pt_output"],
                    result["jax_output"],
                    dummy_input_np,
                    plot_filename,
                    time_val=time_val,
                    sample_idx=sample_idx,
                )

            print(f"{sample_idx:4d} | {time_val:5.2f} -> {result['rel_l2_diff']:12.6e}")

        except Exception as e:
            print(f"{sample_idx:4d} | ERROR: {e}")
            continue

    # Aggregate statistics
    print("\n" + "=" * 20)
    print("AGGREGATED RESULTS")
    print("=" * 20)

    l2_diffs = [r["l2_diff"] for r in results]
    rel_l2_diffs = [r["rel_l2_diff"] for r in results]
    max_abs_diffs = [r["max_abs_diff"] for r in results]
    mean_abs_diffs = [r["mean_abs_diff"] for r in results]

    pt_mins = [r["pt_min"] for r in results]
    pt_maxs = [r["pt_max"] for r in results]
    pt_means = [r["pt_mean"] for r in results]
    jax_mins = [r["jax_min"] for r in results]
    jax_maxs = [r["jax_max"] for r in results]
    jax_means = [r["jax_mean"] for r in results]

    print(f"\n  Total samples tested: {len(results)}")

    print(f"\n  L2 Difference (Primary Metric):")
    print(f"    Mean:   {np.mean(l2_diffs):.6e}")
    print(f"    Std:    {np.std(l2_diffs):.6e}")
    print(f"    Min:    {np.min(l2_diffs):.6e}")
    print(f"    Max:    {np.max(l2_diffs):.6e}")
    print(f"    Median: {np.median(l2_diffs):.6e}")

    print(f"\n  Relative L2 Difference:")
    print(f"    Mean:   {np.mean(rel_l2_diffs):.6e}")
    print(f"    Std:    {np.std(rel_l2_diffs):.6e}")
    print(f"    Min:    {np.min(rel_l2_diffs):.6e}")
    print(f"    Max:    {np.max(rel_l2_diffs):.6e}")
    print(f"    Median: {np.median(rel_l2_diffs):.6e}")

    print(f"\n  PyTorch Output Statistics (across all samples):")
    print(
        f"    Min  - mean: {np.mean(pt_mins):.6f}, min: {np.min(pt_mins):.6f}, max: {np.max(pt_mins):.6f}"
    )
    print(
        f"    Max  - mean: {np.mean(pt_maxs):.6f}, min: {np.min(pt_maxs):.6f}, max: {np.max(pt_maxs):.6f}"
    )
    print(
        f"    Mean - mean: {np.mean(pt_means):.6f}, min: {np.min(pt_means):.6f}, max: {np.max(pt_means):.6f}"
    )

    print(f"\n  JAX Output Statistics (across all samples):")
    print(
        f"    Min  - mean: {np.mean(jax_mins):.6f}, min: {np.min(jax_mins):.6f}, max: {np.max(jax_mins):.6f}"
    )
    print(
        f"    Max  - mean: {np.mean(jax_maxs):.6f}, min: {np.min(jax_maxs):.6f}, max: {np.max(jax_maxs):.6f}"
    )
    print(
        f"    Mean - mean: {np.mean(jax_means):.6f}, min: {np.min(jax_means):.6f}, max: {np.max(jax_means):.6f}"
    )

    print(f"\n  Max Absolute Difference:")
    print(f"    Mean:   {np.mean(max_abs_diffs):.6e}")
    print(f"    Max:    {np.max(max_abs_diffs):.6e}")

    print(f"\n  Mean Absolute Difference:")
    print(f"    Mean:   {np.mean(mean_abs_diffs):.6e}")
    print(f"    Max:    {np.max(mean_abs_diffs):.6e}")

    tolerance = 1e-3
    passed = sum(1 for r in results if r["rel_l2_diff"] < tolerance)
    print(
        f"\n  Pass Rate (rel_L2 < {tolerance}): {passed}/{len(results)} ({100*passed/len(results):.1f}%)"
    )

    print(f"\n  Worst 5 samples by L2 difference:")
    sorted_results = sorted(results, key=lambda x: x["l2_diff"], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        print(
            f"    {i+1}. Sample {r['sample_idx']}: L2={r['l2_diff']:.2e}, rel_L2={r['rel_l2_diff']:.2e}, t={r['time']:.2f}"
        )

    print(f"\n  L2 Difference by Time Value:")
    time_groups = {}
    for r in results:
        t_rounded = round(r["time"], 1)
        if t_rounded not in time_groups:
            time_groups[t_rounded] = []
        time_groups[t_rounded].append(r["l2_diff"])

    for t in sorted(time_groups.keys()):
        vals = time_groups[t]
        print(
            f"    t={t:.1f}: mean={np.mean(vals):.2e}, min={np.min(vals):.2e}, max={np.max(vals):.2e}, n={len(vals)}"
        )

    _plot_summary(results, "comparison_summary.png")

    return {
        "results": results,
        "l2_mean": np.mean(l2_diffs),
        "l2_std": np.std(l2_diffs),
        "l2_min": np.min(l2_diffs),
        "l2_max": np.max(l2_diffs),
        "rel_l2_mean": np.mean(rel_l2_diffs),
        "rel_l2_max": np.max(rel_l2_diffs),
        "pass_rate": passed / len(results),
    }


def verify_multiple_timesteps(
    pt_model,
    jax_model,
    jax_params: Dict[str, Any],
    num_tests: int = 5,
    device: str = "cpu",
):
    """Run verification with multiple random inputs and timesteps (legacy function)."""
    print("\n" + "=" * 80)
    print(f"RUNNING {num_tests} VERIFICATION TESTS")
    print("=" * 80)

    H, W = 128, 128
    MODEL_C = 4
    results = []

    for test_idx in range(num_tests):
        seed = 42 + test_idx

        dummy_input_np = generate_smooth_input(H, W, MODEL_C, seed=seed)
        time_np = np.array([test_idx / max(num_tests - 1, 1)], dtype=np.float32)

        result = run_single_comparison(
            pt_model, jax_model, jax_params, dummy_input_np, time_np, device
        )
        result["test_idx"] = test_idx
        results.append(result)

        status = "[PASS]" if result["rel_l2_diff"] < 1e-3 else "[FAIL]"
        print(
            f"  Test {test_idx+1}: t={time_np[0]:.2f}, L2={result['l2_diff']:.2e}, rel_L2={result['rel_l2_diff']:.2e} {status}"
        )

    print("\n" + "-" * 60)
    passed = sum(1 for r in results if r["rel_l2_diff"] < 1e-3)
    print(f"  Passed: {passed}/{num_tests}")
    print(f"  Max L2 diff: {max(r['l2_diff'] for r in results):.2e}")
    print(f"  Mean L2 diff: {np.mean([r['l2_diff'] for r in results]):.2e}")

    return all(r["rel_l2_diff"] < 1e-3 for r in results)


if __name__ == "__main__":
    import argparse
    import torch
    from convert import convert_model

    parser = argparse.ArgumentParser(
        description="Compare PyTorch and JAX model outputs"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the PyTorch model checkpoint (e.g., /path/to/poseidonT)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples for comprehensive comparison (default: 20)",
    )
    parser.add_argument(
        "--num-plots",
        type=int,
        default=5,
        help="Number of comparison plots to generate (default: 5)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run single example verification instead of comprehensive comparison",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the converted JAX model to a .msgpack file",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert model
    jax_model, jax_params, pt_model, config = convert_model(
        args.model_path,
        save=args.save,
        verbose=False,
    )

    # Run comparison
    if args.single:
        verify_model_outputs(pt_model, jax_model, jax_params, device=device)
    else:
        run_comprehensive_comparison(
            pt_model,
            jax_model,
            jax_params,
            num_samples=args.num_samples,
            num_plots=args.num_plots,
            device=device,
        )
