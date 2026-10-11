"""Helper code for chapter 9: global optimisation and mixed-variable problems.

Used by ``notebooks/09_optimisation-global.ipynb``. The notebook keeps the test
functions, the SciPy solver calls and the short hand-written algorithms visible;
this module holds the figures, the step-by-step DIRECT and branch-and-bound
illustrations and the simulated-annealing widget callback.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from heapq import heappop, heappush

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy.interpolate import UnivariateSpline
from scipy.optimize import OptimizeResult

Func = Callable[[np.ndarray], np.ndarray]

CURVE = "k"  # objective curve in the 1D algorithm illustrations (DIRECT, BnB, SA)
SAMPLES = "tab:orange"  # evaluated points
LOWER, UPPER, BEST = "tab:red", "tab:blue", "tab:green"  # BnB colours named in the text


# --- Introduction ---------------------------------------------------------------------


def plot_vdw_cubic(cubic: Func, roots: np.ndarray, V_range: tuple[float, float] = (0.03, 0.5)) -> Axes:
    """Plot the van der Waals cubic residual near zero and mark its real roots."""
    V = np.linspace(*V_range, 1000)
    real = np.sort(roots[np.isclose(np.imag(roots), 0)].real)
    _, ax = plt.subplots()
    ax.plot(V, cubic(V), label="Cubic residual")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.plot(real, np.zeros_like(real), "o", color=SAMPLES, markersize=9, label="Roots (np.roots)")
    ax.set_ylim(-0.06, 0.03)  # at full scale the bump between the first two roots is invisible
    ax.set(xlabel="Molar volume $V$ (L/mol)", ylabel="Cubic residual", title="VdW Cubic Equation with Three Real Roots")
    ax.legend(loc="upper left")
    return ax


def plot_ackley_landscape(func: Callable[[np.ndarray, np.ndarray], np.ndarray], lim: float = 5.0) -> tuple[Axes, Axes]:
    """3D surface (left) and filled contour map with the global minimum (right) of ``func(x, y)``."""
    x = np.linspace(-lim, lim, 201)
    X, Y = np.meshgrid(x, x)
    Z = func(X, Y)
    fig = plt.figure(figsize=(11, 4.8))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
    ax3d.set(title="3D Surface Plot of the Ackley Function", xlabel="$x$", ylabel="$y$", zlabel="$f(x, y)$")
    ax2d = fig.add_subplot(1, 2, 2)
    cs = ax2d.contourf(X, Y, Z, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax2d, label="$f(x, y)$")
    ax2d.plot(0, 0, "*", color="w", markeredgecolor="k", markersize=16)
    ax2d.set(title="Contour map (star: global minimum)", xlabel="$x$", ylabel="$y$", aspect="equal")
    ax2d.grid(False)
    return ax3d, ax2d


def plot_function_1d(func: Func, bounds: Sequence[float], title: str = "") -> Axes:
    """Plot a 1D objective over ``bounds``."""
    x = np.linspace(*bounds, 1000)
    _, ax = plt.subplots()
    ax.plot(x, func(x))
    ax.set(xlabel="$x$", ylabel="$f(x)$", title=title)
    return ax


# --- 9.2.1 DIRECT: simplified 1D illustration -------------------------------------------


class _Interval:
    """Subinterval [a, b] of the DIRECT illustration with the objective value at its centre."""

    __slots__ = ("a", "b", "f_c")

    def __init__(self, a: float, b: float, f_c: float) -> None:
        self.a, self.b, self.f_c = a, b, f_c

    @property
    def c(self) -> float:
        return (self.a + self.b) / 2

    @property
    def d(self) -> float:
        return self.b - self.a


def _potentially_optimal(intervals: list[_Interval], f_best: float, epsilon: float, K: float = 1.0) -> list[_Interval]:
    """Simplified selection: lower bound with slope K beats f_best by epsilon, or lowest centre value."""
    min_f = min(i.f_c for i in intervals)
    ordered = sorted(intervals, key=lambda i: (i.d, i.f_c), reverse=True)  # large intervals first
    return [i for i in ordered if i.f_c - K * i.d / 2 < f_best - epsilon * abs(f_best) or i.f_c <= min_f]


def _trisect(inter: _Interval, func: Func) -> tuple[_Interval, _Interval, _Interval]:
    """Divide an interval into thirds; the centre third keeps the old function value."""
    third = inter.d / 3
    a_l, b_l = inter.a, inter.a + third
    a_r, b_r = inter.a + 2 * third, inter.b
    left = _Interval(a_l, b_l, float(func((a_l + b_l) / 2)))
    centre = _Interval(inter.a + third, inter.a + 2 * third, inter.f_c)
    right = _Interval(a_r, b_r, float(func((a_r + b_r) / 2)))
    return left, centre, right


def _direct_snapshot(intervals: list[_Interval], po: list[_Interval], points: list[tuple[float, float]]) -> dict:
    """Copy the state of one DIRECT iteration for plotting."""
    return {
        "edges": sorted({e for i in intervals for e in (i.a, i.b)}),
        "po": [(i.c, i.d, i.f_c) for i in po],
        "points": np.array(points),
    }


def direct_1d_demo(
    func: Func, bounds: Sequence[float], max_iters: int = 3, epsilon: float = 1e-4
) -> tuple[list[dict], float, float]:
    """Run a simplified 1D DIRECT (trisect potentially optimal intervals); return snapshots, x_best, f_best."""
    a, b = bounds
    intervals = [_Interval(a, b, float(func((a + b) / 2)))]
    f_best, x_best = intervals[0].f_c, intervals[0].c
    points = [(x_best, f_best)]
    po = _potentially_optimal(intervals, f_best, epsilon)
    history = [_direct_snapshot(intervals, po, points)]
    for _ in range(max_iters):
        new: list[_Interval] = []
        for inter in po:
            left, centre, right = _trisect(inter, func)
            new += [left, centre, right]
            points += [(left.c, left.f_c), (right.c, right.f_c)]
            f_min, x_min = min([(left.f_c, left.c), (right.f_c, right.c), (inter.f_c, inter.c)], key=lambda t: t[0])
            if f_min < f_best:
                f_best, x_best = f_min, x_min
        intervals = [i for i in intervals if i not in po] + new
        po = _potentially_optimal(intervals, f_best, epsilon)
        history.append(_direct_snapshot(intervals, po, points))
    return history, x_best, f_best


def plot_direct_iterations(
    func: Func, bounds: Sequence[float], history: list[dict], k_global: float = 1.0, k_local: float = 10.0
) -> np.ndarray:
    """One panel per DIRECT iteration: intervals, samples and slope ±K bound lines on potentially optimal intervals."""
    ncols = 2
    nrows = -(-len(history) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows + 1), sharex=True, sharey=True, squeeze=False)
    x = np.linspace(*bounds, 1000)
    fx = func(x)
    for it, (ax, snap) in enumerate(zip(axes.flat, history, strict=False)):  # spare panels stay empty
        ax.plot(x, fx, color=CURVE, alpha=0.6)
        for e in snap["edges"]:
            ax.axvline(e, color="tab:gray", linestyle="--", linewidth=1, alpha=0.5)
        for c, d, f_c in snap["po"]:
            for K, color in ((k_global, "tab:green"), (k_local, "m")):
                ax.plot([c - d / 2, c, c + d / 2], [f_c - K * d / 2, f_c, f_c - K * d / 2], color=color, linestyle="--")
        ax.plot(*snap["points"].T, "o", color=SAMPLES, markersize=7, zorder=3)
        ax.set(title=f"Iteration {it}", xlabel="$x$", ylabel="$f(x)$", ylim=(fx.min() - 2.5, fx.max() + 1.0))
        ax.label_outer()
    _hide_unused(axes, len(history))
    handles = [
        Line2D([], [], color=CURVE, alpha=0.6, label="$f(x)$"),
        Line2D([], [], color=SAMPLES, marker="o", linestyle="", label="Evaluated points"),
        Line2D([], [], color="tab:gray", linestyle="--", label="Interval boundaries"),
        Line2D([], [], color="tab:green", linestyle="--",
               label=f"Global Lipschitz bound (low K = {k_global:g}: broad exploration)"),
        Line2D([], [], color="m", linestyle="--",
               label=f"Local Lipschitz bound (high K = {k_local:g}: focused refinement)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    fig.suptitle("DIRECT: global (green) vs local (magenta) Lipschitz bounds on potentially optimal intervals")
    return axes


def _hide_unused(axes: np.ndarray, n_used: int) -> None:
    """Switch off spare panels and restore x tick labels on the panels above them."""
    nrows, ncols = axes.shape
    for k in range(n_used, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].set_axis_off()
        if r > 0 and (r - 1) * ncols + c < n_used:
            axes[r - 1, c].tick_params(labelbottom=True)
            axes[r - 1, c].set_xlabel("$x$")


# --- 9.2.2 SHGO -----------------------------------------------------------------------------


def plot_shgo_minima(func: Func, bounds: Sequence[tuple[float, float]], res: OptimizeResult) -> Axes:
    """Filled contours of ``func([x, y])`` with SHGO's unique local minima (res.xl) and global minimum (res.x)."""
    unique_idx = np.unique(np.round(res.xl, decimals=4), axis=0, return_index=True)[1]  # drop numerical duplicates
    unique_xl = res.xl[unique_idx]
    (x0, x1), (y0, y1) = bounds
    X, Y = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cs = ax.contourf(X, Y, func(np.array([X, Y])), levels=50, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="$f(x, y)$")
    ax.scatter(*unique_xl.T, color="red", marker="x", s=130, linewidths=2.5, label="Local Minima (from res.xl)",
               clip_on=False, zorder=3)  # boundary minima sit on the axes frame
    ax.scatter(*res.x, color="white", edgecolors="k", marker="*", s=350, label="Global Minimum (res.x)", zorder=4)
    ax.set(xlabel="$x$", ylabel="$y$", title="Peaks Function with SHGO Local Minima", aspect="equal")
    ax.grid(False)
    ax.legend(loc="upper right")
    return ax


# --- 9.2.3 Branch and bound: 1D illustration -----------------------------------------------


class _Node:
    """BnB node on [a, b]: lower bound by dense sampling, upper bound from a, b and the midpoint."""

    __slots__ = ("a", "b", "mid", "lower", "upper", "x_upper")

    def __init__(self, func: Func, a: float, b: float, n_dense: int) -> None:
        self.a, self.b, self.mid = a, b, (a + b) / 2
        self.lower = float(np.min(func(np.linspace(a, b, n_dense))))  # proxy for an interval-arithmetic bound
        samples = [a, b, self.mid]
        values = [float(func(s)) for s in samples]
        k = int(np.argmin(values))
        self.upper, self.x_upper = values[k], samples[k]


def _bnb_snapshot(active: list, pruned: list[tuple[float, float]], points: dict[float, float], f_best: float) -> dict:
    """Copy the state of one BnB iteration for plotting."""
    return {
        "active": [(n.a, n.b, n.lower, n.upper) for _, _, n in active],
        "pruned": list(pruned),
        "points": np.array(sorted(points.items())).reshape(-1, 2),
        "f_best": f_best,
    }


def bnb_1d_demo(
    func: Func, bounds: Sequence[float], max_iters: int = 4, n_dense: int = 200
) -> tuple[list[dict], float, float]:
    """Best-first BnB with bisection (priority queue on the lower bound); return snapshots, x_best, f_best."""
    f_best, x_best = np.inf, np.nan
    root = _Node(func, bounds[0], bounds[1], n_dense)
    active = [(root.lower, 0, root)]  # (lower bound, tie-breaking counter, node)
    counter = 1
    points: dict[float, float] = {}
    pruned: list[tuple[float, float]] = []
    history = [_bnb_snapshot(active, pruned, points, f_best)]
    for _ in range(max_iters):
        if not active:
            break
        lower, _, node = heappop(active)
        if lower >= f_best:
            pruned.append((node.a, node.b))
        else:
            if node.upper < f_best:
                f_best, x_best = node.upper, node.x_upper
            points.update({s: float(func(s)) for s in (node.a, node.b, node.mid)})
            for child in (_Node(func, node.a, node.mid, n_dense), _Node(func, node.mid, node.b, n_dense)):
                if child.lower < f_best:
                    heappush(active, (child.lower, counter, child))
                    counter += 1
                    if child.upper < f_best:
                        f_best, x_best = child.upper, child.x_upper
                else:
                    pruned.append((child.a, child.b))
        history.append(_bnb_snapshot(active, pruned, points, f_best))
    return history, x_best, f_best


def plot_bnb_iterations(
    func: Func, bounds: Sequence[float], history: list[dict], ylim: tuple[float, float] = (-2.0, 6.0)
) -> np.ndarray:
    """One panel per BnB iteration: active intervals with lower/upper bounds, pruned intervals and f_best."""
    ncols = 3
    nrows = -(-(len(history) + 1) // ncols)  # one spare panel holds the legend
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows + 0.5), sharex=True, sharey=True, squeeze=False)
    x = np.linspace(*bounds, 1000)
    fx = func(x)
    for it, (ax, snap) in enumerate(zip(axes.flat, history, strict=False)):  # spare panels stay empty
        ax.plot(x, fx, color=CURVE, alpha=0.8)
        for a, b in snap["pruned"]:
            ax.axvspan(a, b, color="tab:gray", alpha=0.25, linewidth=0)
        for a, b, lower, upper in snap["active"]:
            for edge in (a, b):
                ax.axvline(edge, color="tab:gray", linestyle="--", linewidth=1, alpha=0.6)
            ax.hlines(lower, a, b, color=LOWER, linewidth=2.2)
            ax.hlines(upper, a, b, color=UPPER, linestyle="--", linewidth=2.2)
        if np.isfinite(snap["f_best"]):
            ax.axhline(snap["f_best"], color=BEST, linewidth=1.8)
        ax.plot(*snap["points"].T, "o", color=SAMPLES, markersize=7, zorder=3)
        ax.set(title=f"Iteration {it}", xlabel="$x$", ylabel="$f(x)$", ylim=ylim)
        ax.label_outer()
    _hide_unused(axes, len(history))
    handles = [
        Line2D([], [], color=CURVE, alpha=0.8, label="$f(x)$"),
        Line2D([], [], color=SAMPLES, marker="o", linestyle="", label="Evaluated points"),
        Line2D([], [], color="tab:gray", linestyle="--", label="Active interval boundaries"),
        Line2D([], [], color=LOWER, linewidth=2.2, label="Lower Bound"),
        Line2D([], [], color=UPPER, linestyle="--", linewidth=2.2, label="Upper Bound"),
        Line2D([], [], color=BEST, label="Current f_best"),
        Patch(color="tab:gray", alpha=0.25, label="Pruned"),
    ]
    axes.flat[len(history)].legend(handles=handles, loc="center")
    fig.suptitle("BnB Active Intervals with Bounds")
    return axes


# --- 9.2.4 Response surfaces --------------------------------------------------------------


def plot_kriging(
    X_sample: np.ndarray, y_sample: np.ndarray, X_pred: np.ndarray, mu_pred: np.ndarray, ci: np.ndarray
) -> Axes:
    """Kriging (GP) mean with its 95% credible band, the data and an interpolating spline for comparison."""
    spline = UnivariateSpline(X_sample, y_sample, s=0)
    _, ax = plt.subplots()
    ax.plot(X_sample, y_sample, "rs", markersize=8, label="Data (squares)")
    ax.plot(X_pred, mu_pred, "r-", label="Kriging Mean")
    ax.fill_between(X_pred, mu_pred - ci, mu_pred + ci, color="gray", alpha=0.5, label="95% Credible Interval")
    ax.plot(X_pred, spline(X_pred), "b--", label="Spline Interpolation")
    ax.set(xlabel="$x$", ylabel="$f(x)$", title="Kriging Interpolation with Credible Intervals")
    ax.legend()
    return ax


# --- 9.3.4 Simulated annealing widget ---------------------------------------------------------


def plot_sa_trajectories(
    func: Func,
    bounds: Sequence[float],
    traj_slow: list[tuple[float, float]],
    traj_fast: list[tuple[float, float]],
    step: int,
) -> None:
    """Widget callback: SA iterates up to ``step`` on the landscape (left) and their f(x_k) history (right)."""
    fig, (ax, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.5), width_ratios=[1.25, 1])
    x = np.linspace(*bounds, 1000)
    ax.plot(x, func(x), color=CURVE, alpha=0.6, label="$f(x)$")
    for traj, color, name in ((traj_slow, "tab:green", "Slow"), (traj_fast, "tab:red", "Fast")):
        xs, fs = np.array(traj[: min(step + 1, len(traj))]).T
        ax.plot(xs, fs, "o", color=color, markersize=4, alpha=0.3)
        ax.plot(xs[-1], fs[-1], "o", color=color, markersize=12, markeredgecolor="k", label=f"{name} Current")
        ax_hist.plot(np.arange(len(fs)), fs, color=color, label=f"{name} Cooling Trajectory")
    ax.set(xlabel="$x$", ylabel="$f(x)$", title=f"SA Trajectories at Step {step}")
    ax.legend(loc="upper left")
    ax_hist.set(xlabel="Step $k$", ylabel="$f(x_k)$", title="Objective value of the current point", xlim=(0, None))
    ax_hist.legend(loc="upper right")


# --- 9.4 Mixed-variable optimisation ---------------------------------------------------------


def plot_combinatorial_explosion(max_n: int = 6) -> Axes:
    """Number of combinations 10^n of an n-digit lock on a log scale."""
    n = np.arange(1, max_n + 1)
    _, ax = plt.subplots()
    ax.semilogy(n, 10.0**n, "bo-", markersize=8)
    ax.set(
        xlabel="Number of Digits $n$",
        ylabel="Combinations $\\propto$ time $O(10^n)$",
        title="Combinatorial Explosion for n-Digit Lock ($10^n$ Combinations)",
    )
    return ax


def plot_branch_and_cut() -> Axes:
    """LP relaxation of max x + y s.t. x + 2y <= 4, 2x + y <= 4 (x integer) before and after the cut x + y <= 2.5."""
    before = np.array([[0, 0], [0, 2], [4 / 3, 4 / 3], [2, 0]])
    after = np.array([[0, 0], [0, 2], [1, 1.5], [1.5, 1], [2, 0]])
    _, ax = plt.subplots(figsize=(7, 5.5))
    ax.add_patch(Polygon(before, closed=True, alpha=0.3, label="LP Relaxation"))
    ax.add_patch(Polygon(after, closed=True, alpha=0.5, color="orange", label="After Cut"))
    ax.plot([0, 2.5], [2.5, 0], "r--", label="Cut: $x + y \\leq 2.5$")
    ax.plot([0, 0], [0, 2], "k-", linewidth=5, solid_capstyle="butt", label="Feasible points with integer $x$")
    ax.plot([1, 1], [0, 1.5], "k-", linewidth=5, solid_capstyle="butt")
    ax.plot(2, 0, "ko", markersize=8, clip_on=False)
    ax.plot(4 / 3, 4 / 3, "X", color="tab:red", markersize=12, label="LP optimum (fractional), removed by the cut")
    ax.set(xlabel="$x$ (integer)", ylabel="$y$", title="Branch and Cut: Tightening Feasible Region")
    ax.set(xlim=(0, 3), ylim=(0, 3), aspect="equal")
    ax.legend(loc="upper right")
    return ax


def plot_alpha_bb(x: np.ndarray, f: np.ndarray, relax_lower: np.ndarray) -> Axes:
    """Non-convex function and its alphaBB convex underestimator."""
    _, ax = plt.subplots()
    ax.plot(x, f, "b-", label="Non-convex f(x)")
    ax.plot(x, relax_lower, "r--", label="Convex Lower Relaxation (αBB)")
    ax.set(xlabel="$x$", ylabel="$f(x)$", title="Convex Relaxation of Non-convex Function")
    ax.legend()
    return ax


def plot_penalty(penalty: Callable[[np.ndarray], np.ndarray]) -> Axes:
    """Plot a penalty function of the relaxed binary variable z on [0, 1]."""
    z = np.linspace(0, 1, 1000)
    _, ax = plt.subplots()
    ax.plot(z, penalty(z), color="tab:purple")
    ax.set(xlabel="$z$ (continuous relaxation of integer)", ylabel="Penalty $p(z)$",
           title="Cubic Penalty Function for Binary Relaxation")
    return ax
