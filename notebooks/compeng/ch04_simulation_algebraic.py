"""Helpers for Chapter 4: Simulation I, algebraic systems.

Used by ``notebooks/04_simulation-algebraic.ipynb``. The solvers themselves
(Gauss-Newton and its damped / trust-region variants, Gauss-Seidel, the sympy
and finite-difference derivatives) stay in the notebook; this module holds the
plotting code and the widget callback.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

Residual = Callable[[np.ndarray], np.ndarray]
Point = Sequence[float] | np.ndarray

# One colour per entity in every figure of the chapter (colour-vision-deficiency checked).
C_START = "tab:pink"  # starting point x_k / x_0 (square marker)
C_NEXT = "tab:blue"  # next iterate x_{k+1} / converged point (circle marker)
C_STEP = "tab:red"  # Gauss-Newton step and iteration path
C_TANGENT = "tab:red"  # first-order (tangent) model
C_QUAD = "tab:orange"  # second-order (quadratic) model
PLANE_COLORS = ("tab:blue", "tab:orange", "tab:pink")
MS = 9  # marker size of highlighted points


def plot_taylor_models(x0: float = 1.5, a: float = 1.0, n: float = 2.0) -> None:
    """Widget callback: f(x) = a x^n - 2 with its tangent and quadratic model at x0, plus f' and f''."""
    x_min, x_max = -2.0, 8.0
    x = np.linspace(x_min, x_max, 400)

    def f(x_val: np.ndarray | float) -> np.ndarray | float:
        return a * x_val**n - 2

    def df(x_val: np.ndarray | float) -> np.ndarray | float:
        return a * n * x_val ** (n - 1)

    def ddf(x_val: np.ndarray | float) -> np.ndarray | float:
        return a * n * (n - 1) * x_val ** (n - 2)

    fig = plt.figure(figsize=(9.5, 5.0))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.6, 1.0))
    ax = fig.add_subplot(grid[:, 0])
    ax_d1 = fig.add_subplot(grid[0, 1], sharex=ax)
    ax_d2 = fig.add_subplot(grid[1, 1], sharex=ax)

    # x**n is undefined for x < 0 and non-integer n; those points are simply not drawn.
    with np.errstate(invalid="ignore", divide="ignore"):
        ax.plot(x, f(x), "k-", label=r"$f(x) = a x^n - 2$")
        ax.plot(x0, f(x0), "s", color=C_START, ms=MS, label="Guess $x_0$")
        ax.plot(x, f(x0) + df(x0) * (x - x0), "--", color=C_TANGENT, label="Tangent at $x_0$")
        if abs(df(x0)) > 1e-10:
            x1 = x0 - f(x0) / df(x0)
            if x_min < x1 < x_max:
                ax.plot([x1, x1], [0, f(x1)], "k--", lw=1)
                ax.plot(x1, 0, "kx", ms=10, mew=2, label="Tangent intersection $x_1$")
                ax.plot(x1, f(x1), "o", color=C_NEXT, ms=MS, label="$f(x_1)$")
        y_quad = f(x0) + df(x0) * (x - x0) + 0.5 * ddf(x0) * (x - x0) ** 2
        ax.plot(x, y_quad, "--", color=C_QUAD, label="Quadratic approx")
        ax_d1.plot(x, df(x), "-", color=C_TANGENT)
        ax_d1.plot(x0, df(x0), "s", color=C_START, ms=MS)
        ax_d2.plot(x, ddf(x), "-", color=C_QUAD)
        ax_d2.plot(x0, ddf(x0), "s", color=C_START, ms=MS)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend(loc="upper left")
    ax_d1.set_ylabel("$df/dx$")
    ax_d1.tick_params(labelbottom=False)
    ax_d2.set_ylabel("$d^2f/dx^2$")
    ax_d2.set_xlabel("$x$")
    # returns None: interact would otherwise print the return value below the figure


def _objective_grid(
    F: Residual, lim: tuple[float, float], n: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate S(x) = 0.5 ||F(x)||^2 on an n x n grid over lim x lim (F must accept stacked arrays)."""
    g = np.linspace(lim[0], lim[1], n)
    x1, x2 = np.meshgrid(g, g)
    s = 0.5 * np.sum(np.asarray(F(np.array([x1, x2]))) ** 2, axis=0)
    return x1, x2, s


def _contour_objective(ax: Axes, F: Residual, lim: tuple[float, float]) -> None:
    """Draw contours of S(x) with logarithmically spaced levels (detail near the roots)."""
    x1, x2, s = _objective_grid(F, lim)
    levels = np.logspace(-4, np.log10(s.max() + 1e-10), 40)
    ax.contour(x1, x2, s, levels=levels, cmap="viridis", linewidths=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def _arrow(ax: Axes, start: Point, end: Point) -> None:
    """Draw a step arrow from start to end."""
    props = {"arrowstyle": "-|>", "color": C_STEP, "lw": 2, "mutation_scale": 18, "shrinkA": 0, "shrinkB": 0}
    ax.annotate("", xy=(end[0], end[1]), xytext=(start[0], start[1]), arrowprops=props)


def plot_gn_step(
    F: Residual, x_k: np.ndarray, x_next: np.ndarray, lim: tuple[float, float] = (-3.0, 3.0)
) -> tuple[Axes, Axes]:
    """3D surface of S(x) = 0.5 ||F(x)||^2 next to its contour plot with one Gauss-Newton step."""
    fig = plt.figure(figsize=(10.0, 5.2))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    x1, x2, s = _objective_grid(F, lim, n=100)
    ax3d.plot_surface(x1, x2, s, cmap="viridis", alpha=0.8)
    ax3d.set_box_aspect(None, zoom=0.85)
    ax3d.zaxis.set_major_locator(MaxNLocator(4))
    ax3d.set_xlabel("$x_1$")
    ax3d.set_ylabel("$x_2$")
    ax3d.set_zlabel(r"$S(\mathbf{x})$", labelpad=8)
    ax3d.set_title(r"Surface plot of $S(\mathbf{x})$")

    ax = fig.add_subplot(1, 2, 2)
    _contour_objective(ax, F, lim)
    _arrow(ax, x_k, x_next)
    step = Line2D([], [], color=C_STEP, lw=2, label=r"Search direction $\alpha_k \mathbf{p}_k$")
    start = ax.plot(*x_k, "s", color=C_START, ms=MS, label=r"Start $\mathbf{x}_k$")[0]
    nxt = ax.plot(*x_next, "o", color=C_NEXT, ms=MS, label=r"Next $\mathbf{x}_{k+1}$")[0]
    fig.legend(handles=[step, start, nxt], loc="outside lower center", ncols=3)
    ax.set_title("Contour plot with Gauss-Newton step")
    return ax3d, ax


def plot_scalar_step(x_k: float, x_next: float) -> Axes:
    """s(x) = 0.5 (x^2 - 2)^2 with the 1D Newton (= Gauss-Newton) step from x_k to x_next."""

    def s(x: np.ndarray | float) -> np.ndarray | float:
        return 0.5 * (x**2 - 2) ** 2

    x = np.linspace(-3, 3, 400)
    fig, ax = plt.subplots()
    ax.plot(x, s(x), "k-", label="$s(x)$")
    _arrow(ax, (x_k, s(x_k)), (x_next, s(x_k)))
    ax.plot(x_k, s(x_k), "s", color=C_START, ms=MS, label="Start $x_k$")
    ax.plot(x_next, s(x_next), "o", color=C_NEXT, ms=MS, label="Next $x_{k+1}$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$s(x)$")
    ax.set_title("Scalar parabolic example with step trace")
    ax.legend(loc="upper center")
    return ax


def plot_local_models(
    S: Callable[[np.ndarray], float],
    tangent_plane: Callable[[np.ndarray, np.ndarray], float],
    quadratic_approx: Callable[[np.ndarray, np.ndarray], float],
    x_k: np.ndarray,
    p_k: np.ndarray,
    half_width: float = 1.0,
) -> tuple[Axes, Axes]:
    """True S, tangent plane and Gauss-Newton quadratic model near x_k: 3D view and slice along p_k."""
    d = np.linspace(-half_width, half_width, 25)
    x1, x2 = np.meshgrid(x_k[0] + d, x_k[1] + d)
    pts = np.stack([x1, x2], axis=-1)

    def on_grid(fun: Callable[[np.ndarray], float]) -> np.ndarray:
        return np.array([[fun(p) for p in row] for row in pts])

    x_next = x_k + p_k
    fig = plt.figure(figsize=(10.0, 5.2))
    # Drawn in call order: both models lie mostly below S, the search trace stays visible on top.
    ax3d = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    wire = {"rstride": 3, "cstride": 3, "lw": 1.0}
    ax3d.plot_wireframe(x1, x2, on_grid(lambda p: tangent_plane(p, x_k)), color=C_TANGENT, **wire)
    ax3d.plot_wireframe(x1, x2, on_grid(lambda p: quadratic_approx(p, x_k)), color=C_QUAD, **wire)
    ax3d.plot_surface(x1, x2, on_grid(S), cmap="viridis", alpha=0.6)
    seg = x_k + np.linspace(0, 1, 30)[:, None] * p_k
    ax3d.plot(seg[:, 0], seg[:, 1], [S(p) for p in seg], "k-", lw=3)
    ax3d.plot([x_k[0]], [x_k[1]], [S(x_k)], "s", color=C_START, ms=MS)
    ax3d.plot([x_next[0]], [x_next[1]], [S(x_next)], "o", color=C_NEXT, ms=MS)
    ax3d.view_init(elev=25, azim=-55)  # z-axis on the right: its label is not clipped by tight bbox
    ax3d.set_box_aspect(None, zoom=0.8)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax3d.set_xlabel("$x_1$")
    ax3d.set_ylabel("$x_2$")
    ax3d.set_zlabel(r"$S(\mathbf{x})$", labelpad=6)
    ax3d.set_title(r"Local models around $\mathbf{x}_k$")

    alphas = np.linspace(-0.25, 1.5, 200)
    line = x_k + alphas[:, None] * p_k
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(alphas, [tangent_plane(p, x_k) for p in line], "--", color=C_TANGENT)
    ax.plot(alphas, [quadratic_approx(p, x_k) for p in line], "--", color=C_QUAD)
    ax.plot(alphas, [S(p) for p in line], "k-", lw=3)
    ax.plot(0, S(x_k), "s", color=C_START, ms=MS)
    ax.plot(1, S(x_next), "o", color=C_NEXT, ms=MS)
    ax.set_xlabel(r"$\alpha$ in $\mathbf{x}_k + \alpha\,\mathbf{p}_k$")
    ax.set_ylabel(r"$S$")
    ax.set_title(r"Slice along the search direction $\mathbf{p}_k$")
    handles = [
        Patch(color=plt.get_cmap("viridis")(0.5), alpha=0.6, label=r"$S(\mathbf{x})$"),
        Line2D([], [], color=C_TANGENT, ls="--", label="Tangent plane"),
        Line2D([], [], color=C_QUAD, ls="--", label="Quadratic approx (Hessian curvature)"),
        Line2D([], [], color="k", lw=3, label="Search direction trace"),
        Line2D([], [], ls="", marker="s", ms=MS, color=C_START, label=r"$\mathbf{x}_k$"),
        Line2D([], [], ls="", marker="o", ms=MS, color=C_NEXT, label=r"$\mathbf{x}_{k+1}$ ($\alpha_k = 1$)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    return ax3d, ax


def plot_gn_path(
    F: Residual,
    path: np.ndarray,
    lim: tuple[float, float] = (-3.0, 3.0),
    title: str = "Contour plot with Gauss-Newton iterations",
) -> Axes:
    """Contour plot of S(x) = 0.5 ||F(x)||^2 with the iterates of a solver."""
    path = np.asarray(path)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _contour_objective(ax, F, lim)
    ax.plot(path[:, 0], path[:, 1], "-o", color=C_STEP, ms=4, lw=1.5, label="Iteration path")
    ax.plot(*path[0], "s", color=C_START, ms=MS, label="Start")
    ax.plot(*path[-1], "o", color=C_NEXT, ms=MS, label="Converged")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return ax


def _plane_label(a: np.ndarray, b: float) -> str:
    """LaTeX label 'a_1 x_1 + a_2 x_2 + a_3 x_3 = b' with unit coefficients and zeros dropped."""
    out = ""
    for j, coef in enumerate(a, start=1):
        if coef == 0:
            continue
        mag = "" if abs(coef) == 1 else f"{abs(coef):g}"
        sign = ("-" if coef < 0 else "") if not out else (" - " if coef < 0 else " + ")
        out += f"{sign}{mag}x_{j}"
    return f"${out} = {b:g}$"


def plot_planes(
    A: np.ndarray, b: np.ndarray, sol: np.ndarray, lim: tuple[float, float] = (-2.0, 4.0)
) -> Axes:
    """The planes a_i^T x = b_i of a 3x3 linear system and their intersection point sol."""
    g = np.linspace(lim[0], lim[1], 10)
    x1, x2 = np.meshgrid(g, g)
    fig = plt.figure(figsize=(8.0, 5.0))
    # computed_zorder=False draws in call order, so the solution point stays visible on top
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    handles: list[Patch | Line2D] = []
    for a_i, b_i, color in zip(A, b, PLANE_COLORS):
        ax.plot_surface(x1, x2, (b_i - a_i[0] * x1 - a_i[1] * x2) / a_i[2], color=color, alpha=0.45)
        handles.append(Patch(color=color, alpha=0.45, label=_plane_label(a_i, b_i)))
    ax.scatter(*sol, s=100, color="k", edgecolor="w", linewidth=1.5, depthshade=False)
    label = "Solution $({:g}, {:g}, {:g})$".format(*np.round(sol, 10))
    handles.append(Line2D([], [], ls="", marker="o", ms=MS, color="k", mec="w", label=label))
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title("Three Planes Intersecting at Solution Point")
    fig.legend(handles=handles, loc="outside right center")
    return ax
