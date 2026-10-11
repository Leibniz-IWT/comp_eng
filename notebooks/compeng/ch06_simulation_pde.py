"""Helpers for chapter 6 (Simulation III: PDEs): plots and the animation of the 1-D heat and wave equations.

The notebook keeps the discretised right-hand sides and time loops visible; this module only
turns the stored solutions into figures.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes


def plot_contours(
    u_store: Sequence[np.ndarray], t_store: Sequence[float], x: np.ndarray, n_lines: int = 20
) -> Axes:
    """Plot ``n_lines`` temperature profiles T(x) evenly spaced in time, coloured by time in minutes."""
    fig, ax = plt.subplots()
    t_end = t_store[-1] / 60.0
    norm = plt.Normalize(0.0, t_end)
    cmap = plt.get_cmap("viridis")
    for n in range(n_lines):
        t = t_end / n_lines * n
        ax.plot(x, u_store[int(len(u_store) / n_lines * n)], color=cmap(norm(t)))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="time $t$ / min")
    ax.set_xlabel("$x$ / m")
    ax.set_ylabel("$T$ / K")
    return ax


def save_heat_animation(
    u_store: Sequence[np.ndarray],
    t_store: Sequence[float],
    x: np.ndarray,
    path: str | Path,
    interval: int = 200,
    step: int = 10,
    ylim: tuple[float, float] = (200.0, 1400.0),
) -> Path:
    """Animate the temperature profile (every ``step``-th stored state, ``interval`` ms per frame) and save it as a GIF."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(*ylim)
    ax.set_xlabel("$x$ / m")
    ax.set_ylabel("$T$ / K")
    (line,) = ax.plot(x, u_store[0], color="tab:orange")
    title = ax.set_title(f"Time: {t_store[0]:.0f} s")

    def update(frame: int) -> tuple:
        line.set_ydata(u_store[frame])
        title.set_text(f"Time: {t_store[frame]:.0f} s")
        return line, title

    ani = FuncAnimation(fig, update, frames=range(0, len(u_store), step), interval=interval, blit=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(path, writer=PillowWriter(fps=1000.0 / interval))
    plt.close(fig)  # the GIF is shown from markdown; do not also display the last frame
    return path


def plot_wave(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> Axes:
    """Filled contour plot of the wave displacement ``y[i, n] = y(x_i, t_n)`` in the x-t plane."""
    X, T = np.meshgrid(x, t)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, T, y.T, levels=50, cmap="viridis")
    ax.grid(False)  # grid lines would cut through the filled contours
    fig.colorbar(cs, ax=ax, label="y (displacement)")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Time t")
    ax.set_title("Contour Plot of Wave Equation Solution")
    return ax
