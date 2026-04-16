#!/usr/bin/env python
"""
Script to convert MindSpore PDEformer checkpoint to JAX-compatible format.

This script should be run in an environment with MindSpore installed.
You can create a separate environment for the conversion:

    # Create a new environment for conversion
    uv venv mindspore_convert --python 3.9
    mindspore_convert/Scripts/activate  # Windows
    # or: source mindspore_convert/bin/activate  # Linux/Mac
    
    pip install mindspore numpy
    
    # Run conversion
    python convert_checkpoint.py pdeformer2-small.ckpt pdeformer2-small_jax.npz

After conversion, you can load the weights in your JAX environment.
"""
import argparse
import json
import sys
from pathlib import Path


def convert_checkpoint(input_ckpt: str, output_npz: str, output_log: str = None):
    """Convert MindSpore checkpoint to numpy format."""
    try:
        import numpy as np
        from mindspore import load_checkpoint, Tensor
    except ImportError as e:
        print("Error: MindSpore is required for conversion.")
        print("Please install MindSpore in this environment:")
        print("  pip install mindspore")
        print()
        print("Alternatively, you can:")
        print("  1. Create a separate virtual environment")
        print("  2. Install MindSpore there")
        print("  3. Run this conversion script")
        print("  4. Load the converted .npz file in your JAX environment")
        sys.exit(1)
    
    print(f"Loading checkpoint from: {input_ckpt}")
    param_dict = load_checkpoint(input_ckpt)
    
    # Extract weights
    weights = {}
    param_info = []
    
    for name, param in param_dict.items():
        if isinstance(param, Tensor):
            arr = param.asnumpy()
        else:
            arr = np.array(param)
        
        weights[name] = arr
        param_info.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        })
    
    # Save as .npz
    print(f"Saving {len(weights)} parameters to: {output_npz}")
    np.savez_compressed(output_npz, **weights)
    
    # Optionally save parameter info
    if output_log:
        with open(output_log, 'w') as f:
            json.dump({
                "source": input_ckpt,
                "num_params": len(weights),
                "total_elements": sum(np.prod(p.shape) for p in weights.values()),
                "parameters": param_info,
            }, f, indent=2)
        print(f"Saved parameter log to: {output_log}")
    
    # Print summary
    total_params = sum(np.prod(p.shape) for p in weights.values())
    print(f"\nConversion complete!")
    print(f"  Total parameters: {len(weights)}")
    print(f"  Total elements: {total_params:,}")
    print(f"  Output file size: {Path(output_npz).stat().st_size / 1e6:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MindSpore PDEformer checkpoint to numpy format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "input",
        help="Path to input MindSpore .ckpt file"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path to output .npz file (default: input name with .npz extension)"
    )
    parser.add_argument(
        "--log", "-l",
        default=None,
        help="Path to save parameter info JSON log"
    )
    
    args = parser.parse_args()
    
    # Default output name
    output = args.output or str(Path(args.input).with_suffix('.npz'))
    
    convert_checkpoint(args.input, output, args.log)


if __name__ == "__main__":
    main()
