"""Portable locations for figure inputs and generated outputs.

Set SAGE_FIGURE_INPUT_ROOT and SAGE_FIGURE_OUTPUT_ROOT to directories outside
the repository if the datasets or results are stored elsewhere.
"""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path(
    os.environ.get("SAGE_FIGURE_INPUT_ROOT", REPO_ROOT / "figure_inputs")
).expanduser().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("SAGE_FIGURE_OUTPUT_ROOT", REPO_ROOT / "figure_outputs")
).expanduser().resolve()


def _within(root: Path, relative: str) -> str:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Figure path must stay within {root}: {relative}")
    return str(path)


def input_path(relative: str) -> str:
    """Return the location of a required external dataset or result."""
    return _within(INPUT_ROOT, relative)


def output_path(relative: str) -> str:
    """Return the location for a generated figure or intermediate result."""
    path = Path(_within(OUTPUT_ROOT, relative))
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def font_path() -> str:
    """Return an optional override or a bundled Matplotlib TrueType font."""
    override = os.environ.get("SAGE_FIGURE_FONT")
    if override:
        return str(Path(override).expanduser().resolve())
    from matplotlib import font_manager

    return font_manager.findfont("DejaVu Sans")
