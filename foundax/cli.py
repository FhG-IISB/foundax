import runpy
from pathlib import Path


def _run_script(path: Path) -> None:
    runpy.run_path(str(path), run_name="__main__")


def convert_entry() -> None:
    root = Path(__file__).resolve().parents[1]
    _run_script(root / "scripts" / "convert.py")


def compare_entry() -> None:
    root = Path(__file__).resolve().parents[1]
    _run_script(root / "scripts" / "compare.py")
