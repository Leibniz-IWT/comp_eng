"""Helpers for ``05_simulation-ode-dae.ipynb`` (simulation of ODE and DAE systems).

The integrators themselves (explicit and implicit Euler, RK4, adaptive Euler,
Verlet, the implicit DAE solver) and the ``solve_ivp`` calls stay visible in the
notebook. This module holds the figure code and the forward Euler stiffness
widget.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

LINESTYLES = ("-", "--", ":")  # keeps coinciding state variables distinguishable


def plot_dydt_sol(
    t_values: ArrayLike,
    y_values: ArrayLike,
    label: str = "Forward Euler",
    title: str = "Forward Euler Method for ODE",
    xlabel: str = "Time $t$",
    ylabel: str = "$y(t)$",
    marker: str | None = None,
) -> Axes:
    """Plot one numerical solution; the line keeps its label so ``plt.legend()`` works after extra curves."""
    _, ax = plt.subplots()
    ax.plot(t_values, y_values, marker=marker, markersize=4, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return ax


def plot_with_analytical(
    t: ArrayLike,
    y: ArrayLike,
    t_exact: ArrayLike,
    y_exact: ArrayLike,
    label: str,
    title: str,
) -> Axes:
    """Plot a numerical solution (markers at the computed steps) against the analytical one."""
    _, ax = plt.subplots()
    ax.plot(t, y, marker="o", markersize=4, label=label)
    ax.plot(t_exact, y_exact, "k--", label="Analytical")
    ax.set(xlabel="Time $t$", ylabel="$y(t)$", title=title)
    ax.legend()
    return ax


def euler_stiff(dt: float, lam: float = -50.0, y0: float = 1.0, t_f: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Forward Euler for the test equation dy/dt = lam * y on [0, t_f] with step size dt."""
    t, y = 0.0, y0
    t_values, y_values = [t], [y]
    while t < t_f:
        y = y + lam * y * dt
        t += dt
        t_values.append(t)
        y_values.append(y)
    return np.array(t_values), np.array(y_values)


def plot_euler_stiff(dt: float, lam: float = -50.0, y0: float = 1.0, t_f: float = 0.5) -> Axes:
    """Forward Euler against the analytical solution y0 * exp(lam * t) of the stiff test equation."""
    t, y = euler_stiff(dt, lam, y0, t_f)
    t_exact = np.linspace(0.0, t_f, 100)
    ax = plot_with_analytical(
        t,
        y,
        t_exact,
        y0 * np.exp(lam * t_exact),
        label=f"Forward Euler ($\\Delta t$ = {dt:.3f})",
        title="Forward Euler for Stiff ODE",
    )
    ax.set_ylim(-2, 2)  # clip the divergent iterates so the oscillations stay visible
    ax.legend(loc="upper right", frameon=True)  # framed: the divergent lines cross the legend area
    return ax


def euler_stiff_widget(dt: float) -> None:
    """``interact`` callback: forward Euler on dy/dt = -50 y, y(0) = 1, for the slider step size dt."""
    plot_euler_stiff(dt)


def plot_amplification_factors(
    dt_values: ArrayLike,
    r_forward: ArrayLike,
    r_implicit: ArrayLike,
    lambda_val: float,
) -> Axes:
    """Plot |R| of forward and implicit Euler against the step size for the test equation."""
    _, ax = plt.subplots()
    ax.plot(dt_values, r_forward, label="Forward Euler $|R|$")
    ax.plot(dt_values, r_implicit, label="Implicit Euler $|R|$")
    ax.axhline(1, color="k", linestyle="--", label="Stability boundary ($|R| \\leq 1$)")
    ax.set(
        xlabel="Step size $\\Delta t$",
        ylabel="$|R(\\Delta t\\, \\lambda)|$",
        title=f"Stability Analysis for $\\lambda$ = {lambda_val:g}",
    )
    ax.legend()
    return ax


def plot_orbit(q_values: np.ndarray) -> Axes:
    """Plot a planar orbit (positions in m, one row per step) around the central body at the origin."""
    _, ax = plt.subplots()
    ax.plot(q_values[:, 0], q_values[:, 1], label="Verlet Orbit")
    ax.plot(0, 0, "ro", markersize=8, label="Earth")
    ax.set(xlabel="$x$ (m)", ylabel="$y$ (m)", title="Verlet Integrator for Circular Orbit (5 Periods)")
    ax.set_aspect("equal")  # square axes box, no stretched x-range
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return ax


def plot_solution(t_values: ArrayLike, y_values: np.ndarray, labels: list[str], title: str) -> Axes:
    """Plot every state variable (column of ``y_values``) of a DAE solution against time."""
    _, ax = plt.subplots()
    for i, label in enumerate(labels):
        ax.plot(t_values, y_values[:, i], linestyle=LINESTYLES[i % len(LINESTYLES)], label=label)
    ax.set(xlabel="Time $t$", ylabel=labels[0] if len(labels) == 1 else "State variables", title=title)
    if len(labels) > 1:
        ax.legend()
    return ax


def plot_coupled_displacements(t: ArrayLike, y: np.ndarray) -> Axes:
    """Plot the displacements x1, x2 (first two states) of the coupled mass-spring system."""
    _, ax = plt.subplots()
    ax.plot(t, y[0], label="$x_1$ (mass 1)")
    ax.plot(t, y[1], label="$x_2$ (mass 2)")
    ax.set(xlabel="Time $t$ (s)", ylabel="Displacement (m)", title="Coupled Mass-Spring System")
    ax.legend()
    return ax
