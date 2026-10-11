"""Plotting helpers for the simulation tutorial: orbits and energy of the integrators, LBM flow fields.

The integrators and the LBM solver stay visible in the notebook (they are the exercise);
only the figure assembly lives here.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# Same colour for the same integrator in every figure of the tutorial
INTEGRATOR_COLORS: dict[str, str] = {
    "Forward Euler": "C0",
    "Backward Euler": "C1",
    "Symplectic Euler": "C2",
}


def plot_orbits(orbits: dict[str, np.ndarray], earth_radius: float) -> Axes:
    """Plot the x-y orbits (state arrays with columns x, y, vx, vy) of several integrators around Earth (to scale)."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    # Earth to scale: an orbit that ends at its surface was stopped there (r < R_earth)
    ax.add_patch(plt.Circle((0.0, 0.0), earth_radius, color="0.85", zorder=0, label="Earth"))
    for label, y in orbits.items():
        ax.plot(y[:, 0], y[:, 1], label=label, color=INTEGRATOR_COLORS.get(label))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Orbits (up to 100 periods)")
    ax.axis("equal")
    ax.legend(frameon=True)  # the spirals fill the corners: keep the legend readable
    return ax


def plot_energy(energies: dict[str, tuple[np.ndarray, np.ndarray]], period: float) -> Axes:
    """Plot the normalised energy E(t)/|E(0)| of several integrators against the number of orbital periods."""
    fig, ax = plt.subplots()
    for label, (t, E) in energies.items():
        ax.plot(t / period, E / abs(E[0]), label=label, color=INTEGRATOR_COLORS.get(label))
    ax.set_xlabel("Number of Periods")
    ax.set_ylabel("Normalized Energy E(t)/|E(0)|")
    ax.set_title("Energy Drift Comparison")
    if len(energies) > 1:
        ax.legend()
    return ax


def plot_flow(
    snapshots: Sequence[tuple[int, np.ndarray, np.ndarray]],
    u_in_max: float,
    cylinder: tuple[float, float, float],
    fin: tuple[float, float] | None = None,
    n_show: int = 4,
) -> np.ndarray:
    """Plot velocity magnitude and vorticity of ``n_show`` evenly spaced (t, |u|, vorticity) snapshots, one row each."""
    rows = np.unique(np.linspace(0, len(snapshots) - 1, n_show).round().astype(int))
    fig, axs = plt.subplots(len(rows), 2, figsize=(12.0, 1.6 * len(rows) + 0.6), sharex=True, sharey=True, squeeze=False)
    cx, cy, r = cylinder
    for ax_row, k in zip(axs, rows, strict=True):
        t, velocity_mag, vorticity = snapshots[k]
        im0 = ax_row[0].imshow(velocity_mag.T, origin="lower", cmap="RdBu", vmin=0, vmax=u_in_max * 1.5)
        im1 = ax_row[1].imshow(vorticity.T, origin="lower", cmap="RdBu", vmin=-0.1, vmax=0.1)
        ax_row[0].set_title(f"Velocity Magnitude, t = {t}")
        ax_row[1].set_title(f"Vorticity, t = {t}")
        for ax in ax_row:
            ax.grid(False)
            ax.add_patch(plt.Circle((cx, cy), r, color="black", fill=False, linewidth=2))
            if fin is not None:
                ax.hlines(cy, fin[0], fin[1], colors="black", linewidth=2)
        ax_row[0].set_ylabel("y")
    for ax in axs[-1]:
        ax.set_xlabel("x")
    fig.colorbar(im0, ax=axs[:, 0], shrink=0.9)
    fig.colorbar(im1, ax=axs[:, 1], shrink=0.9)
    return axs
