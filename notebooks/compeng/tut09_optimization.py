"""Plotting helpers and widget callbacks for the optimization tutorial.

Used by ``notebooks/09_tutorials-optimization.ipynb`` and its separate solutions
notebook. Only figures, slider wiring and an on-disk cache for slow SHGO runs live
here: the objective functions, solver calls and algorithm loops that students read,
run, modify or complete stay visible in the notebooks and are passed to these
helpers as arguments.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from scipy.optimize import OptimizeResult, shgo

Func1D = Callable[[Any], Any]
CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "cache"

# ---------------------------------------------------------------------------
# Part 1: convexity, penalty functions, gradient descent
# ---------------------------------------------------------------------------


def plot_convexity(a: float, b: float, convex_func: Func1D, nonconvex_func: Func1D) -> None:
    """Widget callback: line segment between a and b on a convex and a non-convex function."""
    if a >= b:
        print("Please set a < b.")
        return
    x = np.linspace(-2, 2, 1000)
    x_segment = x[(x >= a) & (x <= b)]
    _, axs = plt.subplots(1, 2)
    panels = (
        (axs[0], convex_func, "Convex Function"),
        (axs[1], nonconvex_func, "Nonconvex Function"),
    )
    for ax, func, title in panels:
        y_line = func(a) + (x_segment - a) * (func(b) - func(a)) / (b - a)
        ax.plot(x, func(x), color="tab:blue", label="$f(x)$")
        ax.plot(x_segment, y_line, color="tab:red", linestyle="--", label="Line segment")
        ax.scatter([a, b], [func(a), func(b)], color="tab:green", s=60, zorder=5)
        ax.set(title=title, xlabel="$x$", ylabel="$f(x)$")
        ax.legend()


def gd_demo(alpha: float, x_start: float, f: Func1D, grad_f: Func1D) -> None:
    """Widget callback: 100 fixed-step gradient-descent iterations, path drawn on f."""
    x = x_start
    history = [x]
    for _ in range(100):
        x -= alpha * grad_f(x)
        history.append(x)
    xs = np.linspace(-2, 2, 1000)
    _, ax = plt.subplots()
    ax.plot(xs, f(xs), label="$f(x)$")
    ax.plot(history, f(np.array(history)), "ro-", label="GD Path")
    ax.set(xlabel="$x$", ylabel="$f(x)$", title=f"GD with α={alpha}, start={x_start}")
    ax.legend()


def plot_penalties(xspan: np.ndarray, g_1: Func1D, g_2: Func1D, x_l: float, x_u: float) -> Axes:
    """Upper and lower quadratic penalty (barrier) functions over xspan."""
    _, ax = plt.subplots()
    ax.plot(xspan, g_1(xspan), label=rf"Upper bound $x_u ={x_u}$, $g_i = \max(0, x - x_u)^2$")
    ax.plot(xspan, g_2(xspan), label=rf"Lower bound $x_l ={x_l}$, $g_i = \max(0, -(x - x_l))^2$")
    ax.set(xlabel="$x$", ylabel="$g_i(x)$", title="Penalty functions")
    ax.legend()
    return ax


def plot_gd(
    start_x1: float,
    start_x2: float,
    alpha: float,
    f: Callable[[Any, Any], Any],
    optimizer: Callable[[float, float, float], np.ndarray],
    lower: tuple[float, float],
) -> None:
    """Widget callback: optimizer path from (start_x1, start_x2) on contours of f."""
    path = optimizer(start_x1, start_x2, alpha)
    X1, X2 = np.meshgrid(np.linspace(lower[0], 1.5, 200), np.linspace(lower[1], 1.5, 200))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # thin, non-antialiased contour lines keep the stored PNG small (70 kB instead of 380 kB)
    cs = ax.contour(X1, X2, f(X1, X2), levels=20, cmap="viridis", alpha=0.7,
                    linewidths=1.0, antialiased=False)
    fig.colorbar(cs, ax=ax, label="$f(x_1, x_2)$")
    ax.plot(path[:, 0], path[:, 1], "r-", marker="o", markersize=3, label="GD Path")
    ax.scatter(path[0, 0], path[0, 1], color="green", s=100, label="Start", zorder=3)
    ax.scatter(path[-1, 0], path[-1, 1], color="black", s=100, label="End", zorder=3)
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title="Gradient Descent on Multimodal Function")
    ax.legend()


# ---------------------------------------------------------------------------
# Part 2: non-linear least squares, SGD vs. BFGS
# ---------------------------------------------------------------------------


def plot_decay_fit(
    t: np.ndarray, r_noisy: np.ndarray, r0_true: float, beta_true: float, params: Sequence[float]
) -> Axes:
    """Noisy orbital-decay data, true model and fitted model r0 * exp(-beta * t)."""
    r_fit = params[0] * np.exp(-params[1] * t)
    _, ax = plt.subplots()
    ax.scatter(t, r_noisy, label="Noisy data")
    ax.plot(t, r0_true * np.exp(-beta_true * t), "k--", label="True model")
    ax.plot(t, r_fit, "r-", label="Fitted model")
    ax.set(xlabel="Time t (days)", ylabel="Altitude r (km)", title="NLLS Fit: Orbital Decay")
    ax.legend()
    return ax


def interactive_sgd(
    x0: float,
    lr: float,
    beta: float,
    max_epochs: int,
    sigma: float,
    sgd: Callable[..., tuple[np.ndarray, int]],
    bfgs: Callable[[float], tuple[float, np.ndarray, int]],
    f: Func1D,
) -> None:
    """Widget callback: SGD path (left) and BFGS path (right) on the noisy 1D function f."""
    path_sgd, evals_sgd = sgd(x0, lr, beta, max_epochs, sigma)
    x_opt_det, path_det, evals_det = bfgs(x0)

    x_vals = np.linspace(-1, 4, 200)
    y_vals = f(x_vals)  # noise-free for plot
    y_noisy_overlay = y_vals + sigma * np.random.randn(len(x_vals))  # noisy overlay, no fixed seed

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax in axs:
        ax.plot(x_vals, y_noisy_overlay, color="gray", alpha=0.7, label="Noisy overlay")
        ax.plot(x_vals, y_vals, "k-", alpha=0.3, label="Noise-free f(x)")
    x_sgd = path_sgd[-1]
    axs[0].plot(path_sgd, f(path_sgd), "tab:red", marker="o", label="SGD Path", markersize=4)
    axs[0].plot(x_sgd, f(x_sgd), "x", color="tab:red", markersize=10,
                label=f"Final SGD: $f({x_sgd:.2f}) = {f(x_sgd):.2f}$")
    axs[0].set(title=f"SGD ({evals_sgd} evals)", xlabel="x", ylabel="f(x)")
    axs[1].plot(path_det, f(path_det), "tab:blue", marker="o", label="BFGS Path", markersize=4)
    axs[1].plot(x_opt_det, f(x_opt_det), "x", color="tab:blue", markersize=10,
                label=f"Final BFGS: $f({x_opt_det:.2f}) = {f(x_opt_det):.2f}$")
    axs[1].set(title=f"BFGS ({evals_det} evals)", xlabel="x")
    for ax in axs:
        ax.legend()
    fig.suptitle("SGD vs. BFGS on Noisy/Non-Linear Function")


# ---------------------------------------------------------------------------
# Part 3: orbit intersections, DIRECT, MINLP, Lagrange points, LBM fin
# ---------------------------------------------------------------------------


def plot_orbits(a1: float, b1: float, a2: float, b2: float, roots: Sequence | np.ndarray) -> Axes:
    """Two centred ellipses (orbits) and, if any were found, their intersection points."""
    theta = np.linspace(0, 2 * np.pi, 100)
    _, ax = plt.subplots()
    ax.plot(a1 * np.cos(theta), b1 * np.sin(theta), "-", label="Orbit 1")
    ax.plot(a2 * np.cos(theta), b2 * np.sin(theta), "-", label="Orbit 2")
    if len(roots) > 0:
        roots = np.asarray(roots)
        ax.plot(roots[:, 0], roots[:, 1], "ko", label="Roots")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title="Orbital Intersections")
    ax.set_aspect("equal")
    ax.legend()
    return ax


def plot_direct_iteration(
    it: int,
    intervals: Sequence[Any],
    all_points: set[tuple[float, float]],
    po: Sequence[Any],
    x_plot: np.ndarray | None = None,
    f_plot: np.ndarray | None = None,
    ylim: tuple[float, float] | None = None,
) -> Axes:
    """One 1D DIRECT state: intervals, evaluated points, Lipschitz bounds on the PO intervals."""
    _, ax = plt.subplots()
    if x_plot is not None and f_plot is not None:
        ax.plot(x_plot, f_plot, "tab:blue", linewidth=1.2, label="f(x)")
    for i, inter in enumerate(intervals):
        ax.axvline(inter.a, color="tab:gray", linestyle="--", alpha=0.3)
        ax.axvline(inter.b, color="tab:gray", linestyle="--", alpha=0.3)
        if inter in po:
            ax.text(inter.c, inter.f_c + 0.2, f"PO Int {i + 1}", fontsize=10, ha="center")
    ax.scatter([p[0] for p in all_points], [p[1] for p in all_points], color="tab:orange",
               label="Evaluated points", zorder=3)
    k_global, k_local = 1, 10  # example slopes K for the bound cones
    for inter in po:
        x_left, x_right = inter.c - inter.d / 2, inter.c + inter.d / 2
        for k, color in ((k_global, "tab:green"), (k_local, "tab:purple")):
            low = inter.f_c - k * (inter.d / 2)
            ax.plot([x_left, inter.c, x_right], [low, inter.f_c, low],
                    color=color, linestyle="--", alpha=0.5)
    ax.plot([], [], "tab:gray", linestyle="--", label="Interval boundaries")
    ax.plot([], [], "tab:green", linestyle="--", label="Global Lipschitz bound (low K)")
    ax.plot([], [], "tab:purple", linestyle="--", label="Local Lipschitz bound (high K)")
    ax.set(title=f"Iteration {it}: Lipschitz Bounds on Potentially Optimal (PO) Intervals",
           xlabel="$x$", ylabel="$f(x)$")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return ax


def plot_direct_state(it: int, states: Sequence[tuple]) -> None:
    """Widget callback: show stored DIRECT state number it (objective curve hidden)."""
    intervals, all_points, po, _f_best = states[it]
    plot_direct_iteration(it, intervals, all_points, po, ylim=(-10, 2))


def _landscape(
    eff_obj: Callable[[Sequence[float]], float],
    n_bounds: tuple[float, float],
    a_bounds: tuple[float, float],
    feasible_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate eff_obj([n, a]) on a 50 x 50 grid; optionally mask n * a > 4 with NaN."""
    a_vals = np.linspace(*a_bounds, 50)
    N, A = np.meshgrid(np.linspace(*n_bounds, 50), a_vals)
    F = np.vectorize(lambda n, a: eff_obj([n, a]))(N, A)
    if feasible_only:
        F = np.where(N * A <= 4, F, np.nan)
    return N, A, F, a_vals


def plot_minlp_landscape(
    eff_obj: Callable[[Sequence[float]], float],
    x_relaxed: Sequence[float],
    n_int: int,
    a_opt: float,
    n_bounds: tuple[float, float] = (0, 6),
    a_bounds: tuple[float, float] = (0.5, 4.0),
) -> Axes:
    """Feasible MINLP landscape (n vs. a), best a per integer n, relaxed and rounded optimum."""
    N, A, F, a_vals = _landscape(eff_obj, n_bounds, a_bounds, feasible_only=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    cf = ax.contourf(N, A, F, levels=50, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="Objective (-efficiency)")
    for ni in range(int(n_bounds[0]), int(n_bounds[1]) + 1):  # integer n
        a_max = 4 / ni if ni > 0 else 4
        a_valid = a_vals[a_vals <= a_max]
        f_ni = [eff_obj([ni, ai]) for ai in a_valid]
        ax.plot([ni] * len(a_valid), a_valid, "k--", alpha=0.3)
        if f_ni:
            label = "Integer points" if ni == n_bounds[0] else None
            ax.scatter(ni, a_valid[np.argmin(f_ni)], c="k", label=label, clip_on=False, zorder=3)
    n_line = np.linspace(4 / a_bounds[1], n_bounds[1], 200)
    ax.plot(n_line, 4 / n_line, color="tab:red", linestyle="--", label="Cons: n*a=4")
    ax.plot(x_relaxed[0], x_relaxed[1], "x", color="tab:orange", label="Relaxed global minimum",
            markersize=12, mew=2.5, clip_on=False, zorder=4)
    ax.plot(n_int, a_opt, "+", color="tab:green", label="Rounded Int Opt", markersize=14, mew=2.5,
            clip_on=False, zorder=4)
    ax.set(xlabel="n (thrusters)", ylabel="a (throttle area m²)",
           title="MINLP Landscape with Constraint")
    ax.legend(loc="upper right")
    return ax


def plot_landscape_3d(
    eff_obj: Callable[[Sequence[float]], float],
    n_bounds: tuple[float, float],
    a_bounds: tuple[float, float],
) -> Axes:
    """3D surface of eff_obj([n, a]) over the full (n, a) box."""
    N, A, F, _ = _landscape(eff_obj, n_bounds, a_bounds)
    fig = plt.figure(figsize=(7, 5.2))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(N, A, F, cmap="viridis", alpha=0.8, antialiased=False)  # smaller PNG
    ax.set_xlabel("n (thrusters)", labelpad=8)
    ax.set_ylabel("a (throttle area m²)", labelpad=12)
    ax.set_zlabel("Objective (-efficiency)", labelpad=14)
    ax.set_title("3D Surface Plot of MINLP Landscape")
    ax.set_box_aspect(None, zoom=0.82)  # keep all axis labels inside the figure
    return ax


def cached_shgo(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    n: float,
    iters: int,
    name: str = "tut09_lagrange_shgo",
    recompute: bool = False,
    cache_dir: str | pathlib.Path = CACHE_DIR,
) -> OptimizeResult:
    """shgo(func, bounds, n=n, iters=iters), loaded from <cache_dir>/<name>.npz when made with the same inputs."""
    # The cache is used when bounds, n, iters and func's values at three probe points match
    # (so an edited objective or mu is recomputed); it is written when missing or with recompute=True.
    lo, hi = np.asarray(bounds, dtype=float).T
    probe = np.array([func(lo + s * (hi - lo)) for s in (0.1234, 0.5678, 0.9012)])
    key = {"bounds": np.asarray(bounds, dtype=float), "n": np.asarray(float(n)),
           "iters": np.asarray(iters), "probe": probe}
    path = pathlib.Path(cache_dir) / f"{name}.npz"
    if path.exists() and not recompute:
        with np.load(path) as data:
            if all(data[k].shape == v.shape and np.allclose(data[k], v, rtol=1e-10, atol=0)
                   for k, v in key.items()):
                print(f"Loaded cache/{path.name}; pass recompute=True to rerun SHGO")
                return OptimizeResult({k: data[k].item() if data[k].ndim == 0 else data[k]
                                       for k in data.files if k not in key})
    res = shgo(func, bounds, n=n, iters=iters)
    if recompute or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **key, **res)
    return res


def plot_lagrange_points(equilibria: np.ndarray, mu: float) -> Axes:
    """Lagrange points found by SHGO together with the two primaries (rotating frame)."""
    _, ax = plt.subplots()
    ax.scatter(equilibria[:, 0], equilibria[:, 1], color="r", s=50, label="Lagrange Points")
    ax.scatter([-mu, 1 - mu], [0, 0], color="k", s=25, label="Sun & Earth", zorder=3)
    ax.set(title="Lagrange Points in Earth-Sun System", xlabel="$x$", ylabel="$y$")
    ax.set_aspect("equal")
    ax.legend()
    return ax


def plot_orbit_near_equilibrium(sol: Any, eq: np.ndarray, mu: float) -> Axes:
    """Trajectory (sol.y[0], sol.y[1]) of a perturbed state near equilibrium eq."""
    _, ax = plt.subplots()
    ax.plot(sol.y[0], sol.y[1], label="Perturbed orbit")
    ax.scatter([-mu, 1 - mu], [0, 0], color="k", label="Primaries")
    ax.scatter(eq[0], eq[1], color="r", label="Equilibrium", zorder=3)
    ax.set(title="Orbit near Equilibrium (Unstable Example)", xlabel="$x$", ylabel="$y$")
    ax.legend()
    return ax


def plot_flow_field(
    velocity_mag: np.ndarray,
    vorticity: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    fin_start_x: int,
    fin_end_x: int,
    u_in_max: float,
    t: int,
) -> tuple[Axes, Axes]:
    """LBM snapshot: velocity magnitude (top) and vorticity (bottom) with cylinder and fin."""
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 3.9), sharex=True)
    fields = (
        (velocity_mag, "Velocity Magnitude", "Blues", 0, u_in_max * 1.5),
        (vorticity, "Vorticity", "RdBu", -0.1, 0.1),
    )
    for ax, (field, title, cmap, vmin, vmax) in zip(axs, fields):
        im = ax.imshow(field.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        fig.colorbar(im, ax=ax)
        ax.add_patch(Circle((cx, cy), r, facecolor="0.75", edgecolor="black", linewidth=1.5))
        ax.hlines(cy, fin_start_x, fin_end_x, colors="black", linewidth=2)
        ax.set_title(title)
        ax.grid(False)
    axs[1].set_xlabel("x (lattice units)")
    fig.suptitle(f"t = {t}")
    return axs[0], axs[1]
