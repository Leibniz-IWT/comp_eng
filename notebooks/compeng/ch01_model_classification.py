"""Helpers for chapter 1, model classification I: building blocks.

Plotting functions and ``ipywidgets.interact`` callbacks used by
``notebooks/01_model-classification.ipynb``. The models themselves (right-hand
sides, residuals and solver calls) stay visible in the notebook; the widget
callbacks receive them as arguments, so a model edited in the notebook is the
one the sliders use.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Arc, Circle, Polygon, Rectangle
from scipy.integrate import solve_ivp
from scipy.optimize import root

# --- 1.4 Continuous vs. discrete P-controller ----------------------------------


def simulate_euler(
    step: Callable[..., float], y0: float, dt: float, t_end: float, *args: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``y_next = step(y, dt, *args)`` on a uniform grid from 0 to ``t_end``; return ``(t, y)``."""
    n = int(t_end / dt) + 1
    t = np.linspace(0, t_end, n)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        y[i] = step(y[i - 1], dt, *args)
    return t, y


def plot_p_controller(
    t_cont: np.ndarray,
    y_cont: np.ndarray,
    t_disc: np.ndarray,
    y_disc: np.ndarray,
    r: float,
    dt_cont: float,
    h: float,
) -> Axes:
    """Overlay the quasi-continuous and the sampled step response of the P-controller."""
    _, ax = plt.subplots()
    ax.plot(t_cont, y_cont, color="tab:blue", label=f"Continuous (small Δt = {dt_cont:g})")
    ax.step(t_disc, y_disc, where="post", color="tab:green", linestyle="--", label=f"Discrete (h = {h:g})")
    ax.axhline(r, color="black", alpha=0.5, label=f"Setpoint r={r:g}")
    ax.set(xlabel="Time", ylabel="State y", title="Continuous vs. Discrete P-Controller Response")
    ax.legend()
    return ax


# --- 2.1 Static example: steady-state heat conduction on a square plate --------


def plot_plate(T_interior: np.ndarray, T_left: float, T_right: float, T_bot: float, T_top: float) -> Axes:
    """Add the fixed edge temperatures to the interior solution and show the plate temperature."""
    T = np.zeros((T_interior.shape[0] + 2, T_interior.shape[1] + 2))
    T[0, :] = T_bot
    T[:, 0] = T_left
    T[:, -1] = T_right
    T[-1, :] = T_top
    T[1:-1, 1:-1] = T_interior
    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(T, origin="lower", cmap="inferno", extent=[0, 1, 0, 1])
    fig.colorbar(im, ax=ax, label="Temperature [°C]")
    ax.set(title=f"Steady-state plate temperature (T_top = {T_top:.0f} °C)", xlabel="x", ylabel="y")
    ax.grid(False)
    return ax


def plate_callback(
    F: Callable[[np.ndarray, float], np.ndarray],
    nx: int,
    ny: int,
    T_left: float,
    T_right: float,
    T_bot: float,
) -> Callable[[float], None]:
    """Return an ``interact`` callback that solves ``F(x, T_top) = 0`` and plots the plate."""

    def update(T_top: float = 100.0) -> None:
        sol = root(F, np.zeros((nx - 2) * (ny - 2)), args=(T_top,), method="krylov", tol=1e-10)
        plot_plate(sol.x.reshape(ny - 2, nx - 2), T_left, T_right, T_bot, T_top)

    return update


# --- 2.1 Dynamic examples ---------------------------------------------------------


def plot_decay(t: np.ndarray, y: np.ndarray, t_exact: np.ndarray, y_exact: np.ndarray) -> Axes:
    """Compare the numerical solution of dy/dt = -k y with the exact solution."""
    _, ax = plt.subplots()
    ax.plot(t, y, lw=2, label="solve_ivp")
    ax.plot(t_exact, y_exact, "--", label="exact $e^{-kt}$")
    ax.set(xlabel="t", ylabel="y(t)", title="Exponential decay: numerical vs. exact")
    ax.legend()
    return ax


def plot_sho(t: np.ndarray, y: np.ndarray, m: float, k: float, omega: float) -> Axes:
    """Plot the displacement of the simple harmonic oscillator."""
    _, ax = plt.subplots()
    ax.plot(t, y)
    ax.set(xlabel="Time [s]", ylabel="Displacement [m]", title=f"SHO: m={m}, k={k}, ω={omega:.2f} rad/s")
    return ax


def sho_callback(rhs: Callable[[float, np.ndarray, float], list[float]]) -> Callable[..., None]:
    """Return an ``interact`` callback that integrates ``rhs(t, y, omega)`` and plots the displacement."""

    def update(m: float = 1.0, k: float = 4.0, x0: float = 1.0, v0: float = 0.0, t_end: float = 10.0) -> None:
        omega = np.sqrt(k / m)
        sol = solve_ivp(rhs, [0, t_end], [x0, v0], args=(omega,), max_step=0.05)
        plot_sho(sol.t, sol.y[0], m, k, omega)

    return update


# --- 2.2 Linear vs. non-linear ------------------------------------------------------


def draw_truss() -> Axes:
    """Support frame to scale: node 1 pinned, node 2 on a roller, payload m at node 3, unknowns u1, u2, u3."""
    _, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set(xlim=(-0.4, 1.75), ylim=(-0.32, 1.2), aspect="equal", title="Satellite-bus support frame (schematic)")
    ax.axis("off")
    nodes = [(0.0, 0.0), (1.2, 0.0), (0.6, 0.8)]  # m; bars 1-3 and 2-3 are 1.0 m long, cos(alpha) = 0.6
    (x1, y1), (x2, y2), (x3, y3) = nodes
    ax.add_patch(Rectangle((-0.3, -0.16), 1.9, 0.06, facecolor="none", edgecolor="0.4", hatch="///", lw=0))
    ax.plot([-0.3, 1.6], [-0.1, -0.1], color="black", lw=2)  # bus deck
    ax.text(0.6, -0.19, "bus deck", ha="center", va="top")
    ax.add_patch(Polygon([(x1, y1), (x1 - 0.07, -0.1), (x1 + 0.07, -0.1)], facecolor="0.85", edgecolor="black"))
    ax.add_patch(Polygon([(x2, y2), (x2 - 0.07, -0.07), (x2 + 0.07, -0.07)], facecolor="0.85", edgecolor="black"))
    for dx in (-0.04, 0.04):  # roller wheels
        ax.add_patch(Circle((x2 + dx, -0.085), 0.015, facecolor="white", edgecolor="black"))
    for (xa, ya), (xb, yb), label, rot, (ox, oy) in (
        (nodes[0], nodes[1], "1.2 m", 0, (0, 0.05)),
        (nodes[0], nodes[2], "1.0 m", 53.13, (-0.06, 0.045)),
        (nodes[1], nodes[2], "1.0 m", -53.13, (0.06, 0.045)),
    ):
        ax.plot([xa, xb], [ya, yb], color="tab:blue", lw=4, solid_capstyle="round", zorder=1)
        ax.text((xa + xb) / 2 + ox, (ya + yb) / 2 + oy, label, rotation=rot, ha="center", va="center", fontsize=11)
    ax.add_patch(Arc((x1, y1), 0.36, 0.36, theta1=0, theta2=53.13))
    ax.text(0.22, 0.1, r"$\alpha$", ha="center", va="center")
    ax.plot([x1, x2], [y1, y2], "o", ms=11, mfc="white", mec="black", mew=1.5)
    ax.plot(x3, y3, "o", ms=24, mfc="0.85", mec="black", mew=1.5)  # node 3 carries the payload mass
    ax.text(x3, y3, "$m$", ha="center", va="center", fontsize=13)
    for (x, y), label, (dx, dy) in zip(nodes, "123", ((-0.1, 0.1), (0.1, 0.1), (-0.14, 0.04)), strict=True):
        ax.text(x + dx, y + dy, label, ha="center", va="center", fontsize=14)
    dof = {"arrowstyle": "-|>", "color": "black", "lw": 1.5}  # unknown displacements, positive directions
    ax.annotate("", xy=(x2 + 0.28, y2), xytext=(x2 + 0.04, y2), arrowprops=dof)
    ax.annotate("", xy=(x3 + 0.3, y3), xytext=(x3 + 0.07, y3), arrowprops=dof)
    ax.annotate("", xy=(x3, y3 + 0.3), xytext=(x3, y3 + 0.07), arrowprops=dof)
    for x, y, label in ((x2 + 0.31, y2, "$u_1$"), (x3 + 0.33, y3, "$u_2$"), (x3 + 0.03, y3 + 0.3, "$u_3$")):
        ax.text(x, y, label, va="center", fontsize=14)
    ax.annotate("", xy=(x3, y3 - 0.3), xytext=(x3, y3 - 0.07), arrowprops={**dof, "color": "tab:red", "lw": 2})
    ax.text(x3, y3 - 0.33, "$mg$", color="tab:red", ha="center", va="top", fontsize=14)  # load -mg along u3
    ax.text(-0.38, 1.15, r"all bars: $EA = 6\times10^{6}$ N", va="top", fontsize=11)
    return ax


def plot_spring_responses(t_lin: np.ndarray, x_lin: np.ndarray, t_non: np.ndarray, x_non: np.ndarray) -> Axes:
    """Overlay the displacement of the linear spring and of the Duffing oscillator."""
    _, ax = plt.subplots()
    ax.plot(t_lin, x_lin, label="Linear spring")
    ax.plot(t_non, x_non, label="Duffing")
    ax.set(xlabel="t", ylabel="x(t)", title="Linear vs non-linear response")
    ax.legend()
    return ax


def plot_hcw(radial: np.ndarray, along_track: np.ndarray) -> Axes:
    """Plot the HCW relative trajectory with along-track on the horizontal and radial on the vertical axis."""
    _, ax = plt.subplots()
    ax.plot(along_track, radial)
    ax.set(xlabel="Along-track (km)", ylabel="Radial (km)", title="Linearized Satellite Relative Motion")
    return ax


# --- 2.4 Symmetry -----------------------------------------------------------------------


def plot_reflection_symmetry(X1: np.ndarray, X2: np.ndarray, psi: np.ndarray) -> Axes:
    """Contour plot of psi(x1, x2) on the first-quadrant square with the reflection line x2 = x1."""
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    filled = ax.contourf(X1, X2, psi, levels=20, cmap="viridis")
    fig.colorbar(filled, ax=ax, label="ψ(x₁, x₂)")
    ax.contour(X1, X2, psi, levels=20, colors="black", linewidths=0.5)
    hi = X1.max()
    ax.plot([0, hi], [0, hi], "k--", label="Reflection line: x₂ = x₁")
    ax.set(xlabel="x₁", ylabel="x₂", xlim=(0, hi), ylim=(0, hi), aspect="equal")
    ax.set_title("Contour Plot of Multimodal Function\n(with Reflection Symmetry over x₂ = x₁)")
    ax.legend(frameon=True)  # framed: the legend sits on top of the filled contours
    return ax
