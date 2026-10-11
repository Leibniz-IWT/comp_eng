"""Helper code for the tutorial notebook ``03_tutorials-model_class.ipynb``.

Only plotting helpers, widget callbacks and the data for Exercise 3 live here.
The models, solvers and simulations that the exercises are about stay in the
notebook, where students read, run and modify them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# --- Exercise 2: benchmarking ---------------------------------------------------


def plot_results(results: dict[str, tuple[float, float]]) -> Axes:
    """Bar chart of the percentage error (log scale, left) and run time (right) per method."""
    labels = list(results)
    errors = np.array([results[name][0] for name in labels], dtype=float)
    times_us = 1e6 * np.array([results[name][1] for name in labels], dtype=float)
    x = np.arange(len(labels))
    width = 0.35

    _, ax1 = plt.subplots()
    ax1.bar(x - width / 2, errors, width, color="tab:blue")
    ax1.set_ylabel("Error (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_yscale("log")  # log scale for error due to small values
    for xi in x[errors == 0]:  # an exact result has no bar on a log axis, so label it
        ax1.text(xi - width / 2, 0.01, "0", transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, times_us, width, color="tab:red")
    ax2.set_ylabel("Time (µs)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.grid(False)

    ax1.set_xticks(x, labels, rotation=45, ha="right")
    ax1.set_title("Root-Finding Method Comparison for $f(x) = x^3 - 2x - 5$")
    return ax1


def plot_profile(x: np.ndarray, c: np.ndarray, t: float) -> Axes:
    """Concentration profile c(x) at time t."""
    _, ax = plt.subplots()
    ax.plot(x, c)
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("Concentration $c$ (mol/m³)")
    ax.set_title(f"Concentration Profile at t = {t:g} s")
    return ax


# --- Exercise 3: model property tests -------------------------------------------

A_LINEAR = np.array([[2.0, 1.0], [1.0, 3.0]])


def property_test_data(t: np.ndarray) -> tuple[np.ndarray, ...]:
    """Inputs X and noise-free outputs (Y_linear, ..., Y_dynamic) for Exercise 3; row i is sampled at t[i]."""
    # Input data (100 points, x_1, x_2 in [-5, 5]): the diagonal and the anti-diagonal
    X = np.array([
        [-5.0, -5.0], [-4.8, -4.8], [-4.6, -4.6], [-4.4, -4.4], [-4.2, -4.2],
        [-4.0, -4.0], [-3.8, -3.8], [-3.6, -3.6], [-3.4, -3.4], [-3.2, -3.2],
        [-3.0, -3.0], [-2.8, -2.8], [-2.6, -2.6], [-2.4, -2.4], [-2.2, -2.2],
        [-2.0, -2.0], [-1.8, -1.8], [-1.6, -1.6], [-1.4, -1.4], [-1.2, -1.2],
        [-1.0, -1.0], [-0.8, -0.8], [-0.6, -0.6], [-0.4, -0.4], [-0.2, -0.2],
        [0.0, 0.0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8],
        [1.0, 1.0], [1.2, 1.2], [1.4, 1.4], [1.6, 1.6], [1.8, 1.8],
        [2.0, 2.0], [2.2, 2.2], [2.4, 2.4], [2.6, 2.6], [2.8, 2.8],
        [3.0, 3.0], [3.2, 3.2], [3.4, 3.4], [3.6, 3.6], [3.8, 3.8],
        [4.0, 4.0], [4.2, 4.2], [4.4, 4.4], [4.6, 4.6], [4.8, 4.8],
        [5.0, 5.0], [4.8, -4.8], [4.6, -4.6], [4.4, -4.4], [4.2, -4.2],
        [4.0, -4.0], [3.8, -3.8], [3.6, -3.6], [3.4, -3.4], [3.2, -3.2],
        [3.0, -3.0], [2.8, -2.8], [2.6, -2.6], [2.4, -2.4], [2.2, -2.2],
        [2.0, -2.0], [1.8, -1.8], [1.6, -1.6], [1.4, -1.4], [1.2, -1.2],
        [1.0, -1.0], [0.8, -0.8], [0.6, -0.6], [0.4, -0.4], [0.2, -0.2],
        [0.0, 0.0], [-0.2, 0.2], [-0.4, 0.4], [-0.6, 0.6], [-0.8, 0.8],
        [-1.0, 1.0], [-1.2, 1.2], [-1.4, 1.4], [-1.6, 1.6], [-1.8, 1.8],
        [-2.0, 2.0], [-2.2, 2.2], [-2.4, 2.4], [-2.6, 2.6], [-2.8, 2.8],
        [-3.0, 3.0], [-3.2, 3.2], [-3.4, 3.4], [-3.6, 3.6], [-3.8, 3.8],
        [-4.0, 4.0], [-4.2, 4.2], [-4.4, 4.4], [-4.6, 4.6], [-4.8, 4.8]
    ])
    x1, x2 = X[:, 0], X[:, 1]

    # Linear model outputs: f(x) = A x, A = [[2, 1], [1, 3]]
    Y_linear = X @ A_LINEAR.T
    # Non-linear model outputs: f(x) = [x_1^2, sin(x_2)]
    Y_nonlinear = np.column_stack([x1**2, np.sin(x2)])
    # Uncoupled model outputs: f_1 = 2x_1 - 4, f_2 = 3x_2 - 6
    Y_uncoupled = np.column_stack([2 * x1 - 4, 3 * x2 - 6])
    # Coupled model outputs: f_1 = 2x_1 + x_2 - 5, f_2 = x_1 - 3x_2 + 1
    Y_coupled = np.column_stack([2 * x1 + x2 - 5, x1 - 3 * x2 + 1])
    # Symmetric model outputs: f = x_1 + x_2
    Y_symmetric = (x1 + x2)[:, None]
    # Asymmetric model outputs: f = 2x_1 + 3x_2
    Y_asymmetric = (2 * x1 + 3 * x2)[:, None]

    # Static model outputs: f(x) = A x + b, A = [[2, 1], [1, 3]], b = [5, 7]
    Y_static = X @ A_LINEAR.T + np.array([5.0, 7.0])
    # Dynamic model outputs: y(t_i) of the ODE dy/dt = -A y, y(0) = [1, 0], A = [[0, 1], [-2, -3]],
    # sampled at the times t (row i at t[i]), solved analytically for simplicity:
    # y_1(t) = 2 e^t - e^(2t), y_2(t) = 2 e^(2t) - 2 e^t
    Y_dynamic = np.column_stack([2 * np.exp(t) - np.exp(2 * t), 2 * np.exp(2 * t) - 2 * np.exp(t)])

    return (X, Y_linear, Y_nonlinear, Y_uncoupled, Y_coupled,
            Y_symmetric, Y_asymmetric, Y_static, Y_dynamic)


def plot_property_data(X: np.ndarray, Y_linear: np.ndarray, Y_nonlinear: np.ndarray,
                       Y_uncoupled: np.ndarray, Y_coupled: np.ndarray,
                       Y_symmetric: np.ndarray, Y_asymmetric: np.ndarray) -> np.ndarray:
    """Scatter the outputs of the six black-box data sets against x_1 (2 x 3 panels)."""
    panels = [
        ("Linear Model", Y_linear, ["$f_1$", "$f_2$"]),
        ("Non-Linear Model", Y_nonlinear, ["$f_1$ ($x_1^2$)", r"$f_2$ ($\sin(x_2)$)"]),
        ("Uncoupled Model", Y_uncoupled, ["$f_1$", "$f_2$"]),
        ("Coupled Model", Y_coupled, ["$f_1$", "$f_2$"]),
        ("Symmetric Model", Y_symmetric, ["$f$"]),
        ("Asymmetric Model", Y_asymmetric, ["$f$"]),
    ]
    _, axs = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (title, Y, labels) in zip(axs.flat, panels, strict=True):
        for j, label in enumerate(labels):
            ax.scatter(X[:, 0], Y[:, j], s=15, label=label)
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel(r"$f(\mathbf{x})$")
        if len(labels) > 1:
            ax.legend()
    return axs


# --- Exercise 5: widget callbacks (they return None: interact displays return values) ---


def plot_transfer(burn1_duration: float, burn2_duration: float,
                  simulate: Callable[[float, float], Any],
                  r_geo: float, ab: float, tof_approx: float) -> None:
    """Widget callback: simulate the LEO-to-GEO transfer with ``simulate`` and plot the trajectory."""
    sol = simulate(burn1_duration, burn2_duration)
    if not sol.success:
        print("Integration failed!")
        return

    _, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(0, 0, "o", color="tab:blue", label="Earth")
    ax.plot(sol.y[0], sol.y[1], "-", color="tab:orange",
            label=f"Trajectory (Burn1: {burn1_duration:.0f}s, Burn2: {burn2_duration:.0f}s)")

    # GEO circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r_geo * np.cos(theta), r_geo * np.sin(theta), color="gray", linestyle=":", label="GEO Orbit")

    # S/C at approx. arrival
    idx_tof = np.argmin(np.abs(sol.t - tof_approx))
    ax.plot(sol.y[0][idx_tof], sol.y[1][idx_tof], "o", color="tab:orange", label="S/C at Approx. Arrival")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("LEO-to-GEO Transfer with Event-Triggered Burns\n"
                 f"(Approx. Δv1: {ab * burn1_duration:.0f} m/s, Δv2: {ab * burn2_duration:.0f} m/s)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))


def plot_three_body(perturbation: float, simulate: Callable[[float], Any], d_binary: float,
                    pos_planet2: Callable[[float], np.ndarray]) -> None:
    """Widget callback: simulate the three-body problem with ``simulate``, plot trajectory and Poincaré map."""
    sol = simulate(perturbation)
    if not sol.success:
        print("Integration failed!")
        return

    _, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Trajectory plot
    axs[0].plot(0, 0, "o", color="tab:blue", label="Planet 1")
    theta = np.linspace(0, 2 * np.pi, 200)
    axs[0].plot(d_binary * np.cos(theta), d_binary * np.sin(theta), color="gray", linestyle=":",
                label="Planet 2 Orbit")
    axs[0].plot(sol.y[0], sol.y[1], "-", color="tab:orange", linewidth=0.5,  # thin: ~100 revolutions
                label="Third Body Trajectory")
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("y (m)")
    axs[0].set_title(f"Trajectory with Velocity Perturbation: {perturbation}%")
    axs[0].legend(loc="upper right", frameon=True)  # frame keeps the text readable over the orbit circle
    axs[0].set_aspect("equal", adjustable="datalim")

    # Poincaré map (x vs vx at y=0 crossings in the frame rotating with Planet 2)
    if len(sol.t_events[0]) > 2:  # need multiple crossings for a map
        x, y, vx, vy = sol.y_events[0].T
        e = np.array([pos_planet2(t) for t in sol.t_events[0]])
        ex, ey = (e / np.linalg.norm(e, axis=1, keepdims=True)).T  # rotating x axis: direction to Planet 2
        # at y = 0 the frame rotation only adds a y velocity, so v_x is the projection of v on that axis
        axs[1].scatter(ex * x + ey * y, ex * vx + ey * vy, s=15, color="tab:orange")
        axs[1].set_title("Poincaré Map (y=0 crossings)\nframe rotating with Planet 2")
    else:
        axs[1].set_title("Poincaré Map (insufficient crossings)")
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("$v_x$ (m/s)")
