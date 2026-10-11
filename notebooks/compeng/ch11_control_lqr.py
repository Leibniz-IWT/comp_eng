"""Helpers for chapter 11 (state-space and optimal linear control): simulations and plots."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def _lti_rhs(_t: float, y: np.ndarray, A: np.ndarray, B: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Right-hand side dy/dt = A y + B u for a constant input u."""
    return A @ y + B @ u


def plot_states(
    t: np.ndarray, y: np.ndarray, labels: Sequence[str], title: str = "", ylabel: str = "States"
) -> Axes:
    """Plot each row of y against time t; a single series is labelled on the y-axis instead of a legend."""
    _, ax = plt.subplots()
    for yi, label in zip(np.atleast_2d(y), labels, strict=True):
        ax.plot(t, yi, label=label)
    ax.set_xlabel("Time [s]")
    if len(labels) > 1:
        ax.set_ylabel(ylabel)
        ax.legend()
    else:
        ax.set_ylabel(labels[0])
    ax.set_title(title)
    return ax


def _output_axes(t: np.ndarray, z: np.ndarray, title: str, label: str | None, ax: Axes | None) -> Axes:
    """Plot the output z(t) on a new figure, or add it to ax (keeping its title); legend if labelled."""
    if ax is None:
        _, ax = plt.subplots()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Output $z$")
        ax.set_title(title)
    ax.plot(t, z, label=label)
    if label is not None:
        ax.legend()
    return ax


def plot_step_response(
    A: np.ndarray,
    C: np.ndarray,
    B: np.ndarray | None = None,
    D: np.ndarray | None = None,
    title: str = "Step Response",
    t_end: float = 10,
    label: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Simulate the unit-step response of an LTI system from y(0) = 0 and plot z(t); pass ax to overlay."""
    B = np.array([[0], [1]]) if B is None else B
    D = np.array([[0]]) if D is None else D
    t_eval = np.linspace(0, t_end, 100)
    u_values = (t_eval > 0).astype(float).reshape(1, -1)  # u(t) = 1 for t > 0

    def state_eq(t: float, y: np.ndarray) -> np.ndarray:
        return A @ y + B[:, 0] * (1.0 if t > 0 else 0.0)

    sol = solve_ivp(state_eq, [0, t_end], np.zeros(A.shape[0]), t_eval=t_eval)
    z = (C @ sol.y + D @ u_values)[0]
    return _output_axes(sol.t, z, title, label, ax)


def plot_free_response(
    A: np.ndarray,
    C: np.ndarray,
    y0: Sequence[float],
    title: str = "Free Response",
    t_end: float = 10,
    label: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot the unforced output z(t) = C e^{At} y(0) (u = 0) from the initial state y0; pass ax to overlay."""
    t = np.linspace(0, t_end, 200)
    z = np.array([C @ expm(A * ti) @ np.asarray(y0, float) for ti in t])[:, 0]
    return _output_axes(t, z, title, label, ax)


def plot_decoupler_comparison(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, u_no_dec: np.ndarray, u_dec: np.ndarray, t_end: float = 10
) -> tuple[Axes, Axes]:
    """Simulate constant inputs without and with a static decoupler and plot the outputs side by side."""
    t_eval = np.linspace(0, t_end, 100)
    _, axs = plt.subplots(1, 2, sharey=True)
    for ax, u, title in zip(axs, (u_no_dec, u_dec), ("Without Decoupler", "With Decoupler"), strict=True):
        sol = solve_ivp(_lti_rhs, [0, t_end], np.zeros(A.shape[0]), args=(A, B, np.asarray(u, float)), t_eval=t_eval)
        for i, z_i in enumerate(C @ sol.y, start=1):
            ax.plot(sol.t, z_i, label=f"$z_{i}$")
        ax.set_xlabel("Time [s]")
        ax.set_title(title)
        ax.legend()
    axs[0].set_ylabel("Outputs")
    return axs[0], axs[1]


def simulate_noisy_system(
    Ad: np.ndarray,
    Bd: np.ndarray,
    C: np.ndarray,
    Qn_d: np.ndarray,
    Rn: np.ndarray,
    y0: Sequence[float],
    N: int,
    dt: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate y_{k+1} = Ad y_k + Bd u_k + w_k, z_k = C y_k + v_k with u = 0; return t, y, z, u."""
    rng = np.random.default_rng(seed)
    n, p = Ad.shape[0], C.shape[0]
    u = np.zeros((Bd.shape[1], N))  # No input for simplicity
    y = np.zeros((n, N))
    z = np.zeros((p, N))
    y[:, 0] = y0
    z[:, 0] = C @ y[:, 0] + rng.multivariate_normal(np.zeros(p), Rn)
    for k in range(1, N):
        w = rng.multivariate_normal(np.zeros(n), Qn_d)  # process noise
        y[:, k] = Ad @ y[:, k - 1] + Bd @ u[:, k - 1] + w
        z[:, k] = C @ y[:, k] + rng.multivariate_normal(np.zeros(p), Rn)  # measurement noise
    return np.arange(N) * dt, y, z, u


def plot_estimates(
    t: np.ndarray, y_true: np.ndarray, y_hat: np.ndarray, z: np.ndarray | None = None, title: str = ""
) -> tuple[Axes, Axes]:
    """Plot true states (solid) and estimates (dashed): angle y_1 on top with optional measurements, rate y_2 below."""
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9.0, 5.5))
    if z is not None:
        ax1.plot(t, np.ravel(z), ".", color="0.45", markersize=7, label="Noisy measurement $z$")
    ax1.plot(t, y_true[0], color="C0", label="True $y_1$")
    ax1.plot(t, y_hat[0], "--", color="C1", label=r"Estimate $\hat{y}_1$")
    ax2.plot(t, y_true[1], color="C0", label="True $y_2$")
    ax2.plot(t, y_hat[1], "--", color="C1", label=r"Estimate $\hat{y}_2$")
    ax1.set_ylabel("Angle $y_1$")
    ax2.set_ylabel("Rate $y_2$")
    ax2.set_xlabel("Time [s]")
    ax1.set_title(title)
    ax1.legend()
    ax2.legend()
    return ax1, ax2
