#!/usr/bin/env python3
"""
Run every per-architecture parity compare script, parse the [PASS]/[FAIL]
lines, and emit a Markdown table summarising:

    | Architecture | Test | Input → Output | Max abs diff | Rel L2 |

Used to keep ``docs/architectures.md`` honest about what's tested and at
what tolerance. Run after any change to a ``scripts/compare_<name>.py``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Architecture → (compare script, fixed input-shape per test name).
# Input shapes are looked up by test-name prefix below.
ARCHITECTURES = [
    ("Transolver", "compare_transolver.py", "thuml/Transolver"),
    ("SFNO", "compare_sfno.py", "NVIDIA/torch-harmonics"),
    ("FFNO", "compare_ffno.py", "alasdairtran/fourierflow"),
    ("FNO", "compare_fno.py", "neuraloperator/neuraloperator"),
    ("WNO", "compare_wno.py", "TapasTripura/WNO (structural only)"),
    ("DiT", "compare_dit.py", "facebookresearch/DiT"),
    ("GNOT", "compare_gnot.py", "HaoZhongkai/GNOT"),
]

# Map test-name (or substring) to (input shape string, kind-of-input).
# "Input" here is the shape the user-facing model accepts, NOT internal
# PT layout. For multi-input models we show both.
INPUT_SHAPES = {
    "Transolver Irregular": "(64, 2) coords + (64, 1) func",
    "Transolver Structured2D": "(16, 16, 2) coords + (16, 16, 1) func",
    "SHT forward": "(32, 64) lat-lon",
    "SHT inverse": "(8, 8) spectral",
    "SphericalConv2d": "(32, 64, 2)",
    "SFNO2d full": "(16, 32, 3)",
    "FactorizedSpectralConv2d": "(16, 16, 8)",
    "FactorizedSpectralConv3d": "(8, 10, 12, 6)",
    "SpectralConv1d": "(32, 4)",
    "SpectralConv2d": "(16, 20, 3)",
    "SpectralConv3d": "(12, 14, 16, 3)",
    "WNO1d structural": "(32, 2)",
    "WNO2d structural": "(32, 32, 2)",
    "WNO3d gradient-flow": "(16, 16, 16, 1)",
    "DiTBlock (upstream conv.)": "(16, 32) tokens + (32,) cond",
    "LinearAttention": "(1, 12, 32)",
    "LinearCrossAttention": "(1, 10, 32) query + 2×(1, 16, 32) branches",
    "CrossAttentionBlock": "(1, 10, 32) query + 2×(1, 16, 32) branches",
}

LINE_RE = re.compile(
    r"\[(?P<status>PASS|FAIL)\]\s+(?P<name>.+?)\s+shape=\((?P<shape>[^)]+)\)"
    r"(?:\s+max_abs=(?P<maxabs>[\d.e+-]+))?"
    r"(?:\s+mean_abs=(?P<meanabs>[\d.e+-]+))?"
    r"(?:\s+rel_l2=(?P<rel>[\d.e+-]+))?"
)

# WNO emits a different format because it's structural-only.
LINE_RE_WNO = re.compile(
    r"\[(?P<status>PASS|FAIL)\]\s+(?P<name>WNO[\dD]+ \w+)\s+(?:shape=\((?P<shape>[^)]+)\)\s+)?"
    r"finite="
)


def run_compare(script: Path) -> str:
    cmd = [
        "pixi",
        "run",
        "--environment",
        "dev",
        "python",
        str(script),
    ]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return proc.stdout + proc.stderr


def parse_lines(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        m = LINE_RE.search(line)
        if m:
            d = m.groupdict()
            d["shape"] = "(" + d["shape"] + ")"
            out.append(d)
            continue
        m = LINE_RE_WNO.search(line)
        if m:
            d = m.groupdict()
            d["shape"] = "(" + d["shape"] + ")" if d["shape"] else "—"
            d.setdefault("maxabs", None)
            d.setdefault("rel", None)
            out.append(d)
    return out


def main():
    print()
    print(
        "| Architecture | Test | Input → Output | Max abs diff | Rel L2 | PyTorch reference |"
    )
    print("|---|---|---|---|---|---|")
    all_pass = True
    for arch, script_name, ref in ARCHITECTURES:
        script = REPO_ROOT / "scripts" / script_name
        out = run_compare(script)
        rows = parse_lines(out)
        if not rows:
            print(f"| {arch} | (no test output parsed) | — | — | — | {ref} |")
            all_pass = False
            continue
        for i, r in enumerate(rows):
            status = r["status"]
            if status != "PASS":
                all_pass = False
            inp = INPUT_SHAPES.get(r["name"].strip(), "?")
            io = f"{inp} → {r['shape']}"
            maxabs = r["maxabs"] or "—"
            rel = r["rel"] or "—"
            arch_cell = arch if i == 0 else ""
            ref_cell = ref if i == 0 else ""
            print(
                f"| {arch_cell} | {r['name'].strip()} | {io} | {maxabs} | {rel} | {ref_cell} |"
            )
    print()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
