"""Plotting helpers and widget callbacks for chapter 7: convex and linear optimization.

Used by ``notebooks/07_optimisation-linear.ipynb``. The objective functions, the
``linprog``/``minimize`` calls and the gradient-descent loops stay visible in the
notebook; this module only draws the figures. The functions passed to
``ipywidgets.interact`` return ``None`` because ``interact`` displays any other
return value below the figure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Shared look of the LP/QP figures (feasible region, simplex path, IPM path)
REGION_STYLE: dict[str, Any] = {"facecolor": "lightblue", "edgecolor": "tab:blue", "alpha": 0.5}
LP_XLIM = (-0.2, 2.0)
LP_YLIM = (-0.2, 1.6)
MULTIMODAL_LIM = (0.0, 1.5)  # plotting domain of the multimodal example in section 7.7


def _lp_axes(title: str) -> Axes:
    """Return equal-aspect axes framed around the 2D LP example."""
    _, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.set(xlim=LP_XLIM, ylim=LP_YLIM, aspect="equal", xlabel="$x_1$", ylabel="$x_2$", title=title)
    return ax


def _draw_region(ax: Axes, vertices: np.ndarray, label: str | None = None) -> None:
    """Shade the convex polygon spanned by ``vertices``."""
    ax.add_patch(Polygon(vertices, closed=True, label=label, **REGION_STYLE))


def _barrier(X1: np.ndarray, X2: np.ndarray, mu: float) -> np.ndarray:
    """Log-barrier objective of the 2D LP on a grid (NaN outside the feasible region)."""
    valid = (X1 > 0) & (X2 > 0) & (X1 + X2 < 2) & (X1 < 1.5) & (X2 < 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_terms = np.log(X1) + np.log(X2) + np.log(2 - X1 - X2) + np.log(1.5 - X1) + np.log(1 - X2)
    z = -X1 - X2 - mu * log_terms
    z[~valid] = np.nan
    return z


def plot_brute_force_1d(x: np.ndarray, y: np.ndarray, x_min: float, y_min: float) -> Axes:
    """Plot the sampled 1D objective and the best grid point."""
    _, ax = plt.subplots()
    ax.plot(x, y, alpha=0.5, label=r"$f(x) = x^2 + \sin(2\pi x)$")
    ax.scatter(x, y, color="gray", alpha=0.7, label="Points sampled with brute force search")
    ax.scatter(
        x_min, y_min, marker="X", s=120, color="red", zorder=3,
        label=f"Minimum found at x={x_min:.2f}, f(x)={y_min:.2f}",
    )
    ax.set(xlabel="$x$", ylabel="$f(x)$")
    ax.legend()
    return ax


def plot_brute_force_2d(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X1: np.ndarray, X2: np.ndarray,
    x1_min: float, x2_min: float, z_min: float,
) -> Axes:
    """Show the grid samples and the best one on a contour plot of f."""
    X1d, X2d = np.meshgrid(np.linspace(X1.min(), X1.max(), 100), np.linspace(X2.min(), X2.max(), 100))
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    cs = ax.contourf(X1d, X2d, f(X1d, X2d), levels=20, cmap="viridis", alpha=0.7)
    fig.colorbar(cs, ax=ax, label="$f(x_1, x_2)$")
    ax.scatter(X1, X2, color="gray", s=12, label="Sampled Points")
    ax.scatter(
        x1_min, x2_min, color="red", s=80, zorder=3,
        label=f"Minimum at ({x1_min:.2f}, {x2_min:.2f}), f={z_min:.2f}",
    )
    ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal", title="Brute Force Search on 2D Quadratic Function")
    fig.legend(loc="outside lower center", ncols=2)
    return ax


def plot_convexity(a: float = -1.5, b: float = 1.5) -> None:
    """Widget callback: chord between a and b on a convex and on a nonconvex function."""
    if a >= b:
        print("Please set a < b.")
        return
    x = np.linspace(-2, 2, 1000)
    x_segment = x[(x >= a) & (x <= b)]
    cases = (
        (lambda t: t**2, "$f(x) = x^2$ (Convex)", "Convex Function"),
        (lambda t: t**4 - 3 * t**2, "$f(x) = x^4 - 3x^2$ (Nonconvex)", "Nonconvex Function"),
    )
    _, axs = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for ax, (func, label, title) in zip(axs, cases, strict=True):
        chord = func(a) + (x_segment - a) * (func(b) - func(a)) / (b - a)
        ax.plot(x, func(x), color="tab:blue", label=label)
        ax.plot(x_segment, chord, color="red", linestyle="--", label="Line segment")
        ax.scatter([a, b], [func(a), func(b)], color="green", s=60, zorder=5)
        ax.set(title=title, xlabel="$x$", ylabel="$f(x)$")
        ax.legend()


def plot_feasible_region_2d(vertices: np.ndarray) -> Axes:
    """Shade the 2D LP feasible region with dashed contours of z = x1 + x2."""
    ax = _lp_axes("2D LP Feasible Region (Convex Polytope)")
    _draw_region(ax, vertices)
    ax.plot(vertices[:, 0], vertices[:, 1], "o", color="tab:blue", ms=8)
    X1, X2 = np.meshgrid(np.linspace(0, 2, 100), np.linspace(0, 2, 100))
    ax.contour(X1, X2, X1 + X2, levels=10, colors="gray", linestyles="dashed")
    return ax


def plot_feasible_region_3d() -> Axes:
    """Draw the tetrahedron x1, x2, x3 >= 0, x1 + x2 + x3 <= 1 and its vertices."""
    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = [corners[[0, 1, 2]], corners[[0, 1, 3]], corners[[0, 2, 3]], corners[[1, 2, 3]]]
    fig = plt.figure(figsize=(6.5, 6.0))
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Poly3DCollection(faces, linewidths=1, **REGION_STYLE))
    ax.scatter(*corners.T, color="red", s=50)
    ax.set(
        xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", xlim=(0, 1), ylim=(0, 1), zlim=(0, 1),
        title="3D LP Feasible Region (Convex Polytope)",
    )
    ax.set_box_aspect(None, zoom=0.85)
    return ax


def plot_simplex_path(vertices: np.ndarray, bfs_points: np.ndarray) -> Axes:
    """Draw the basic feasible solutions visited by the simplex method."""
    ax = _lp_axes("Simplex Method BFS Path")
    _draw_region(ax, vertices)
    ax.plot(bfs_points[:, 0], bfs_points[:, 1], "o-", color="red", ms=8)
    return ax


def plot_simplex_step(steps: Sequence[dict[str, Any]], vertices: np.ndarray, step: int = 0) -> None:
    """Widget callback: simplex path up to ``step`` over contours of z = x1 + x2."""
    path = np.array([s["bfs"] for s in steps[: step + 1]])
    ax = _lp_axes(steps[step]["desc"])
    X1, X2 = np.meshgrid(np.linspace(-0.5, 2.5, 100), np.linspace(-0.5, 2.5, 100))
    ax.contour(X1, X2, X1 + X2, levels=np.arange(0, 3.1, 0.25), colors="gray", linestyles="dashed", alpha=0.7)
    _draw_region(ax, vertices)
    ax.plot(path[:, 0], path[:, 1], "o-", color="red", ms=8, label="Path")
    ax.scatter(
        *path[-1], color="green", s=120, zorder=3,
        label=f"Current BFS: {steps[step]['bfs']}, z={steps[step]['z']}",
    )
    ax.legend(loc="upper right", frameon=True)


def plot_ipm_step(path: np.ndarray, mu_values: Sequence[float], vertices: np.ndarray, step: int = 0) -> None:
    """Widget callback: IPM path up to ``step`` over objective and barrier contours."""
    X1, X2 = np.meshgrid(np.linspace(0.01, 1.99, 100), np.linspace(0.01, 1.99, 100))
    ax = _lp_axes(f"IPM Step {step}: μ = {mu_values[step]} decreasing")
    _draw_region(ax, vertices)
    ax.contour(X1, X2, X1 + X2, levels=8, colors="gray", linestyles="dashed")
    ax.contour(X1, X2, _barrier(X1, X2, mu_values[step]), levels=8, colors="tab:blue", linestyles="solid")
    ax.plot(path[: step + 1, 0], path[: step + 1, 1], "o-", color="green", label="IPM Path")
    ax.scatter(*path[step], color="red", s=120, zorder=3, label=f"Step {step}")
    contour_keys = [
        Line2D([], [], color="gray", linestyle="--", label="Original Objective Contours"),
        Line2D([], [], color="tab:blue", label="Barrier Function Contours"),
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=contour_keys + handles, loc="upper right", frameon=True)


def plot_qp(x_opt: np.ndarray) -> Axes:
    """Contours of the QP objective, its feasible triangle and the computed optimum."""
    X1, X2 = np.meshgrid(np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cs = ax.contour(X1, X2, X1**2 + X2**2 + X1 + X2, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="$f(x_1, x_2)$")
    _draw_region(ax, np.array([[0, 0], [1, 0], [0, 1]]), label="Feasible Region")
    ax.scatter(*x_opt, color="red", s=100, zorder=3, clip_on=False, label="Optimum")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal", title="QP Objective Contours and Feasible Region")
    ax.legend(loc="upper right", frameon=True)
    return ax


def plot_least_squares(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> Axes:
    """Scatter the noisy data together with the fitted straight line."""
    _, ax = plt.subplots()
    ax.scatter(x, y, label="Data")
    ax.plot(x, params[0] * x + params[1], color="red", label="Fit")
    ax.set(xlabel="$x$", ylabel="$y$", title="Linear Least-Squares Fit")
    ax.legend()
    return ax


def plot_gd_1d(
    f: Callable[[np.ndarray], np.ndarray],
    descent: Callable[[float, float], np.ndarray],
    alpha: float = 0.01,
    x_start: float = 1.5,
) -> None:
    """Widget callback: run ``descent(x_start, alpha)`` and draw its path on f(x)."""
    history = descent(x_start, alpha)
    t = np.linspace(-2, 2, 1000)
    _, ax = plt.subplots()
    ax.plot(t, f(t), label="$f(x)$")
    ax.plot(history, f(history), "o-", color="red", label="GD Path")
    ax.set(xlabel="$x$", ylabel="$f(x)$", title=f"GD with α={alpha:g}, start={x_start:g}")
    ax.legend()


def plot_gd_2d(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    run_gd: Callable[[float, float, float], np.ndarray],
    start_x1: float = 0.7,
    start_x2: float = 0.6,
    alpha: float = 0.01,
) -> None:
    """Widget callback: run ``run_gd`` from (start_x1, start_x2) and draw the path on contours of f."""
    path = run_gd(start_x1, start_x2, alpha)
    X1, X2 = np.meshgrid(np.linspace(*MULTIMODAL_LIM, 200), np.linspace(*MULTIMODAL_LIM, 200))
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    # Filled contours: line contours of this function give a ~0.5 MB PNG, and widget outputs
    # above ~0.2 MB stall execution through nbclient (tools/check_notebooks.py --execute).
    cs = ax.contourf(X1, X2, f(X1, X2), levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="$f(x_1, x_2)$")
    ax.plot(path[:, 0], path[:, 1], "r-", marker="o", markersize=3, label="GD Path")
    ax.scatter(*path[0], color="green", edgecolors="white", s=100, zorder=3, label="Start")
    ax.scatter(*path[-1], color="black", edgecolors="white", s=100, zorder=3, label="End")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title="Gradient Descent on Multimodal Function")
    ax.legend(loc="upper right", frameon=True)


def plot_multimodal_surface(f: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Axes:
    """Draw the 3D surface of the multimodal function on its plotting domain."""
    X1, X2 = np.meshgrid(np.linspace(*MULTIMODAL_LIM, 100), np.linspace(*MULTIMODAL_LIM, 100))
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X1, X2, f(X1, X2), cmap="viridis", edgecolor="none")
    ax.set(
        xlabel="$x_1$", ylabel="$x_2$", zlabel="$f(x_1, x_2)$",
        title="3D Surface Plot of the Multimodal Function",
    )
    ax.set_box_aspect(None, zoom=0.85)
    return ax
