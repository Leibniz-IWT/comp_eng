"""Helper code for chapter 3, *Model classification III: non-linearity & chaos*.

Used by ``notebooks/03_chaos.ipynb``. The notebook keeps the model equations
(right-hand sides such as ``ddp``, ``pendulum`` and ``cr3bp``) and the solver
calls visible and passes them to the functions below, which do the plotting,
the widget wiring, the vectorised bifurcation sweep and the cached Poincaré
section.
"""

from __future__ import annotations

import io
import pathlib
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import FloatSlider, IntSlider, interact
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "cache"

TIME_LABEL = r"Time $t/T$ (drive periods, $T = 2\pi/\omega$)"
Y_LABEL = r"$y$ (radians)"

# Colours shared by all phase-space figures
FLOW = "tab:blue"  # vector field and flow arrows
TRAJECTORY = "k"  # simulated trajectories
STABLE = "tab:green"
UNSTABLE = "tab:red"
MARGINAL = "tab:orange"
DRIVE_CYCLE_COLORS = ("tab:red", "tab:purple", "tab:orange")

# Logistic map: first two period doublings and the accumulation point r_inf
R_DOUBLING = (3.0, 1 + np.sqrt(6))
R_INF = 3.5699456

RHS = Callable[..., ArrayLike]


def use_compact_png() -> None:
    """Store inline figures as 256-colour palette PNGs (about 3x smaller notebook outputs)."""
    from IPython import get_ipython
    from IPython.core.pylabtools import print_figure
    from matplotlib.figure import Figure
    from PIL import Image

    def png(fig: Figure) -> bytes | None:
        data = print_figure(fig, "png")  # same rendering as the inline backend
        if data is None:  # figure without axes
            return None
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image = image.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        out = io.BytesIO()
        image.save(out, format="png", optimize=True)
        return out.getvalue()

    shell = get_ipython()
    if shell is not None:
        plt.close(plt.figure())  # load the inline backend first, it registers its own formatter
        shell.display_formatter.formatters["image/png"].for_type(Figure, png)


# --------------------------------------------------------------------------
# Driven damped pendulum (DDP)
# --------------------------------------------------------------------------
def plot_time_series(
    sols: OptimizeResult | Sequence[OptimizeResult],
    T: float,
    title: str,
    labels: Sequence[str] | None = None,
    t_start: float = 0.0,
) -> Axes:
    """Plot y(t) of one or more solve_ivp solutions against time in drive periods."""
    sols = [sols] if isinstance(sols, OptimizeResult) else list(sols)
    _, ax = plt.subplots()
    for i, sol in enumerate(sols):
        mask = sol.t / T >= t_start
        label = None if labels is None else labels[i]
        ax.plot(sol.t[mask] / T, sol.y[0, mask], "-" if i == 0 else "--", label=label)
    ax.set(xlabel=TIME_LABEL, ylabel=Y_LABEL, title=title)
    if labels is not None and len(sols) > 1:
        ax.legend()
    return ax


def plot_drive_cycles(sol: OptimizeResult, T: float, t_start: int, title: str) -> Axes:
    """Time series with the values y(kT) at integer drive periods marked by dashed lines."""
    _, ax = plt.subplots()
    mask = sol.t / T >= t_start
    ax.plot(sol.t[mask] / T, sol.y[0, mask], alpha=0.7)
    for i, k in enumerate(range(t_start, round(sol.t[-1] / T) + 1)):
        y = sol.sol(k * T)[0]  # needs dense_output=True
        color = DRIVE_CYCLE_COLORS[i % len(DRIVE_CYCLE_COLORS)]
        ax.vlines(k, min(0, y), max(0, y), linestyles="dashed", colors=color)
        ax.hlines(y, t_start, k, linestyles="dashed", colors=color)
    ax.set(xlabel=TIME_LABEL, ylabel=Y_LABEL, title=title)
    return ax


def plot_period_doubling(
    sols: Sequence[OptimizeResult],
    gammas: Sequence[float],
    periods: Sequence[str],
    T: float,
    n_show: int = 16,
) -> np.ndarray:
    """Stacked steady-state time series (last `n_show` drive periods) with y(kT) marked."""
    _, axs = plt.subplots(len(sols), 1, figsize=(10, 2.1 * len(sols)),
                          sharex=True, sharey=True, squeeze=False)
    axs = axs[:, 0]
    for ax, sol, gamma, period in zip(axs, sols, gammas, periods, strict=True):
        k_end = round(sol.t[-1] / T)
        mask = sol.t / T >= k_end - n_show
        ax.plot(sol.t[mask] / T, sol.y[0, mask], alpha=0.8)
        k = np.arange(k_end - n_show, k_end + 1)
        ax.plot(k, sol.sol(k * T)[0], "o", color="tab:red", ms=6, label=r"$y(kT)$")
        ax.set(ylabel=Y_LABEL, title=f"γ = {gamma}: {period}")
    axs[0].legend(loc="upper right")
    axs[-1].set_xlabel(TIME_LABEL)
    return axs


def ddp_explorer(simulate: Callable[..., OptimizeResult], T: float) -> None:
    """Interactive DDP time series: sliders for γ, periods, plot start and initial state."""

    def update(gamma: float, periods: int, t_start: int, y_0: float, dy_0: float) -> None:
        sol = simulate(gamma, periods=periods, y_0=y_0, dy_0=dy_0)
        plot_time_series(sol, T, f"DDP Oscillation for γ = {gamma:g}", t_start=t_start)

    interact(
        update,
        gamma=FloatSlider(min=0.9, max=1.105, step=0.001, value=0.9, readout_format=".3f"),
        periods=IntSlider(min=1, max=50, step=1, value=6),
        t_start=IntSlider(min=0, max=40, step=1, value=0),
        y_0=FloatSlider(min=0, max=np.pi * 4, step=0.001, value=0),
        dy_0=FloatSlider(min=-np.pi * 4, max=np.pi * 4, step=0.001, value=0),
    )


def bifurcation_sweep(
    ddp: RHS,
    gammas: ArrayLike,
    beta: float,
    omega0: float,
    omega: float,
    y_0: float = 0.0,
    n_transient: int = 500,
    n_samples: int = 100,
) -> np.ndarray:
    """Stroboscopic samples y(kT) after `n_transient` drive periods, one row per γ."""
    # All γ values are integrated together as one vector ODE (same RK45 and tolerances
    # as `simulate`): a few seconds instead of ~10 minutes for a Python loop over γ.
    gammas = np.asarray(gammas, dtype=float)
    n = gammas.size
    T = 2 * np.pi / omega

    def rhs(t: float, s: np.ndarray) -> np.ndarray:
        return np.ravel(ddp(t, s.reshape(2, n), beta, omega0, gammas, omega))

    t_samples = np.arange(n_transient + 1, n_transient + n_samples + 1) * T
    z0 = np.concatenate([np.full(n, y_0), np.zeros(n)])
    sol = solve_ivp(rhs, [0, t_samples[-1]], z0, t_eval=t_samples, rtol=1e-8, atol=1e-8)
    return sol.y[:n]


def plot_bifurcation(
    gammas: np.ndarray, y_samples: np.ndarray, doubling: Sequence[float] = ()
) -> Axes:
    """Bifurcation diagram of the DDP; optionally mark the period-doubling points γ_1, γ_2, ..."""
    _, ax = plt.subplots()
    ax.scatter(np.repeat(gammas, y_samples.shape[1]), y_samples.ravel(), s=0.5, color="k")
    ax.set(
        xlabel=r"$\gamma$",
        ylabel=r"$y$ (radians) at stroboscopic times",
        title=f"Bifurcation Diagram for DDP (γ from {gammas[0]:g} to {gammas[-1]:g})",
    )
    y_low = y_samples.min()
    close = 0.03 * (gammas[-1] - gammas[0])  # labels closer than this are pushed apart
    for i, gamma_i in enumerate(doubling):
        j = np.argmin(np.abs(gammas - gamma_i))  # index of gamma_i in the sweep
        ax.plot([gamma_i, gamma_i], [y_low, y_samples[j].max()], "--", lw=1.5)
        ha = "center"
        if i + 1 < len(doubling) and doubling[i + 1] - gamma_i < close:
            ha = "right"
        elif i > 0 and gamma_i - doubling[i - 1] < close:
            ha = "left"
        ax.text(gamma_i, y_low - 0.01, rf"$\gamma_{i + 1}$", ha=ha, va="top")
    if doubling:
        ax.set_ylim(bottom=y_low - 0.07)
    return ax


# --------------------------------------------------------------------------
# Phase space
# --------------------------------------------------------------------------
def plot_phase_line(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    stable: Sequence[float] = (),
    unstable: Sequence[float] = (),
) -> tuple[Axes, Axes]:
    """Flow function f(x) (left) and the 1D phase line with flow arrows and fixed points (right)."""
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    ax0.plot(x, f(x))
    ax0.axhline(0, color="k", linestyle="--")
    ax0.set(xlabel="$x$", ylabel=r"$\dot{x} = f(x)$", title="Flow Function")

    ax1.axhline(0, color="k", linewidth=2)
    arrow = {"arrowstyle": "-|>", "color": FLOW, "lw": 2, "mutation_scale": 18}
    for xi in np.linspace(x[0], x[-1], 20):  # points right where f > 0, left where f < 0
        ax1.annotate("", xy=(xi + 0.05 * np.sign(f(xi)), 0), xytext=(xi, 0), arrowprops=arrow)
    ax1.plot(unstable, np.zeros(len(unstable)), "o", color=UNSTABLE, ms=10, label="Unstable",
             zorder=3)
    ax1.plot(stable, np.zeros(len(stable)), "o", color=STABLE, ms=10, label="Stable", zorder=3)
    ax1.set(xlabel="$x$ (Phase Line)", yticks=[], title="1D Phase Line",
            xlim=(x[0] - 0.1, x[-1] + 0.1), ylim=(-0.1, 0.3))
    ax1.legend(loc="upper right")
    return ax0, ax1


def _phase_portrait(
    ax: Axes,
    rhs: RHS,
    ics: Sequence[Sequence[float]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    t_end: float,
    traj_label: str | None = None,
) -> None:
    """Draw streamlines of a planar vector field and the trajectories starting at `ics`."""
    X, Y = np.meshgrid(np.linspace(*xlim, 20), np.linspace(*ylim, 20))
    U, V = rhs(0.0, np.array([X, Y]))
    ax.streamplot(X, Y, U, V, color=FLOW, density=1, linewidth=1)
    t = np.linspace(0, t_end, 1000)  # negative t_end integrates backward in time
    for ic in ics:  # tight tolerances keep closed orbits (center) closed
        z = solve_ivp(rhs, [0, t_end], ic, dense_output=True, rtol=1e-8, atol=1e-8).sol(t)
        ax.plot(z[0], z[1], color=TRAJECTORY, lw=2, label=traj_label)
    ax.set(xlim=xlim, ylim=ylim)


def plot_pendulum(
    pendulum: RHS,
    ics: Sequence[Sequence[float]],
    theta_lim: tuple[float, float],
    title: str = "2D Phase Plane: Damped Pendulum",
    traj_label: str | None = None,
) -> Axes:
    """Phase plane (θ, v) of the damped pendulum with its fixed points (0, 0) and (π, 0)."""
    _, ax = plt.subplots(figsize=(9, 5))
    _phase_portrait(ax, pendulum, ics, theta_lim, (-3, 3), 20, traj_label)
    ax.plot(0, 0, "o", color=STABLE, ms=10, mec="k", label="Stable", zorder=3)
    ax.plot(np.pi, 0, "o", color=UNSTABLE, ms=10, mec="k", label="Unstable (Saddle)", zorder=3)
    ax.set(xlabel=r"$\theta$", ylabel="$v$", title=title)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    return ax


def pendulum_explorer(pendulum: RHS) -> None:
    """Interactive damped-pendulum phase plane with sliders for the initial angle and velocity."""

    def update(theta0: float, v0: float) -> None:
        plot_pendulum(pendulum, [[theta0, v0]], (-np.pi - 1, np.pi + 1),
                      title="Interactive Phase Plane: Damped Pendulum",
                      traj_label=f"Trajectory from ({theta0:.2f}, {v0:.2f})")

    interact(
        update,
        theta0=FloatSlider(min=-np.pi, max=np.pi, step=0.1, value=0.5, description="Initial θ:"),
        v0=FloatSlider(min=-3, max=3, step=0.1, value=0.0, description="Initial v:"),
    )


def plot_critical_point(
    A: ArrayLike,
    name: str,
    t_end: float = 5.0,
    ics: Sequence[Sequence[float]] = ((1.5, 1.5), (-1.5, 1.5), (1.5, -1.5)),
) -> Axes:
    """Phase portrait of the linear system dy/dt = A y around the critical point y* = 0."""
    A = np.asarray(A, dtype=float)
    re = np.linalg.eigvals(A).real
    color = STABLE if np.all(re < 0) else UNSTABLE if np.any(re > 0) else MARGINAL
    _, ax = plt.subplots(figsize=(5, 5))
    _phase_portrait(ax, lambda _t, y: np.tensordot(A, y, axes=1), ics, (-2, 2), (-2, 2), t_end)
    ax.plot(0, 0, "o", color=color, ms=10, mec="k", zorder=3)
    ax.set(xlabel="$y_1$", ylabel="$y_2$", title=f"Phase Portrait: {name}", aspect="equal")
    return ax


# --------------------------------------------------------------------------
# Logistic map
# --------------------------------------------------------------------------
def plot_cascade(r_vals: np.ndarray, X: np.ndarray) -> Axes:
    """Heat map of the logistic-map iterates x_n over the parameter r and the iteration n."""
    fig, ax = plt.subplots(figsize=(9, 5))
    dr = r_vals[1] - r_vals[0]
    extent = (r_vals[0] - dr / 2, r_vals[-1] + dr / 2, 0.5, X.shape[0] + 0.5)
    im = ax.imshow(X, origin="lower", aspect="auto", interpolation="nearest",
                   cmap="viridis", vmin=0, vmax=1, extent=extent)
    fig.colorbar(im, ax=ax, label="$x_n$ (state)", pad=0.02)
    ax.grid(False)
    for r in (*R_DOUBLING, R_INF):
        ax.axvline(r, color="w", ls="--", lw=1.2)
    top = ax.secondary_xaxis("top")  # name the regions between the dashed lines
    centres = [(r_vals[0] + 3) / 2, sum(R_DOUBLING) / 2, (R_DOUBLING[1] + R_INF) / 2,
               (R_INF + r_vals[-1]) / 2]
    top.set_xticks(centres, ["period 1", "period 2", "period 4", r"chaos ($r > r_\infty$)"])
    top.tick_params(length=0)
    ax.set(xlabel="$r$ (parameter)", ylabel="$n$ (iteration)",
           title="Period-Doubling Cascade in the Logistic Map")
    return ax


def plot_logistic_iterates(
    r_values: Sequence[float], iterates: Sequence[Sequence[float]]
) -> np.ndarray:
    """One panel of iterates x_n against n per value of r (2 x 2 grid)."""
    _, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, r, xs in zip(axs.flat, r_values, iterates, strict=False):  # up to four values of r
        ax.plot(np.arange(len(xs)), xs, ".", color="tab:blue", ms=8)
        ax.set(title=f"r = {r}", ylim=(0, 1))
    for ax in axs[1]:
        ax.set_xlabel("Iteration $n$")
    for ax in axs[:, 0]:
        ax.set_ylabel("$x_n$")
    return axs


def plot_logistic_bifurcation(r_vals: np.ndarray, x_vals: np.ndarray) -> Axes:
    """Bifurcation diagram of the logistic map: attractor values x (one row of x_vals per r)."""
    _, ax = plt.subplots()
    ax.plot(np.repeat(r_vals, x_vals.shape[1]), x_vals.ravel(), "k,", alpha=0.1)
    ax.set(xlabel="$r$", ylabel="$x$ (attractor values)",
           title="Bifurcation Diagram of the Logistic Map")
    return ax


# --------------------------------------------------------------------------
# Circular restricted three-body problem (CR3BP)
# --------------------------------------------------------------------------
def poincare_section(
    rhs: RHS,
    ics: ArrayLike,
    t_span: tuple[float, float],
    args: tuple[float, ...] = (),
    cache: str | None = None,
    recompute: bool = False,
) -> list[np.ndarray]:
    """Crossings of the plane y = 0 with ẏ > 0 as (x, ẋ) arrays, one per initial state."""
    # The cache file in notebooks/cache/ is used when it was made with the same ics,
    # t_span and args (it does not track edits of `rhs`: pass recompute=True then).
    # It is written when missing or with recompute=True.
    ics = np.asarray(ics, dtype=float)
    key = {"ics": ics, "t_span": np.asarray(t_span, dtype=float),
           "args": np.asarray(args, dtype=float)}
    path = CACHE_DIR / cache if cache else None
    if path is not None and path.exists() and not recompute:
        data = np.load(path)
        if all(data[k].shape == v.shape and np.allclose(data[k], v) for k, v in key.items()):
            return np.split(data["points"].astype(float), data["splits"])

    def crossing(_t: float, state: np.ndarray, *_args: float) -> float:
        return state[1]

    crossing.direction = 1  # only count upward crossings (ẏ > 0)
    print(f"Integrating {len(ics)} orbits (several minutes) ...")
    sections = []
    for ic in ics:
        sol = solve_ivp(rhs, t_span, ic, args=args, method="RK45", rtol=1e-8, atol=1e-8,
                        events=crossing)
        sections.append(sol.y_events[0][:, [0, 2]])
    if path is not None and (recompute or not path.exists()):
        path.parent.mkdir(parents=True, exist_ok=True)
        splits = np.cumsum([len(s) for s in sections])[:-1]
        np.savez_compressed(path, points=np.concatenate(sections).astype(np.float32),
                            splits=splits, **key)
    return sections


def plot_poincare(sections: Sequence[np.ndarray], values: Sequence[float], mu: float) -> Axes:
    """Poincaré section points (x, ẋ), coloured per orbit by its initial ẏ."""
    fig, ax = plt.subplots()
    norm = plt.Normalize(min(values), max(values))
    cmap = plt.get_cmap("viridis")
    for points, value in zip(sections, values, strict=True):
        ax.scatter(points[:, 0], points[:, 1], s=1, color=cmap(norm(value)), linewidths=0)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label=r"initial $\dot{y}$ of the orbit")
    ax.set(xlabel="$x$", ylabel=r"$\dot{x}$",
           title=rf"Poincaré Section ($y=0$, $\dot{{y}}>0$) for CR3BP, $\mu={mu}$")
    return ax
