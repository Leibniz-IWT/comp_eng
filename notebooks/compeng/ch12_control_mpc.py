"""Helpers for chapter 12 (model-predictive control): plots of the closed-loop MPC simulations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_mpc(
    y: np.ndarray,
    u: np.ndarray,
    dt: float,
    y_ref: float,
    u_min: float,
    u_max: float,
    state_label: str = "Position $y_1$",
    input_label: str = "Input $u$",
    titles: tuple[str, str] = ("State Trajectory", "Control Input"),
) -> tuple[Axes, Axes]:
    """Plot the controlled state against its reference (top) and the applied input with its bounds (bottom)."""
    y, u = np.ravel(y), np.ravel(u)
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9.0, 5.5))
    ax1.plot(np.arange(len(y)) * dt, y, color="C0", label=state_label)
    ax1.axhline(y_ref, color="C3", linestyle="--", label="Reference")
    ax2.step(np.arange(len(u)) * dt, u, where="post", color="C0", label=input_label)  # zero-order hold
    ax2.axhline(u_max, color="C3", linestyle="--", label="Input bounds")
    ax2.axhline(u_min, color="C3", linestyle="--")
    ax1.set_ylabel(state_label)
    ax2.set_ylabel(input_label)
    ax2.set_xlabel("Time $t = k\\,\\Delta t$ [s]")
    ax1.set_title(titles[0])
    ax2.set_title(titles[1])
    ax1.legend()
    ax2.legend()
    return ax1, ax2
