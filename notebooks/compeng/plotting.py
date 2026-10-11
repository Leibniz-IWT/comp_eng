"""Shared matplotlib styling for all course notebooks.

Call :func:`setup` once in the (hidden) setup cell of a notebook. Figures then
share one look in the book, in the slides and in the live notebook.
"""

from __future__ import annotations

import matplotlib as mpl

STYLE: dict[str, object] = {
    "figure.figsize": (9.0, 4.5),
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "figure.constrained_layout.use": True,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
}


def setup() -> None:
    """Apply the course plotting style to matplotlib's global rcParams."""
    mpl.rcParams.update(STYLE)
