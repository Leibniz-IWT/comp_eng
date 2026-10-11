"""Helper code for chapter 2, Model classification II: Degrees of Freedom (DOF) analysis.

The notebook ``02_dof-analysis.ipynb`` keeps the models (residuals, right-hand
sides, objective functions) and the solver calls visible. This module holds the
figure code and the compressible flow function used by the cold gas thruster
examples.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_tank_height(
    t: np.ndarray,
    h: np.ndarray,
    title: str = "Water Height in Draining Tank Over Time",
    setpoint: float | None = None,
) -> Axes:
    """Plot the tank water height h(t), optionally with the controller setpoint."""
    _, ax = plt.subplots()
    ax.plot(t, h)
    if setpoint is not None:
        ax.axhline(setpoint, color="tab:red", linestyle="--", label="Setpoint")
        ax.legend()
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Water Height $h$ (m)")
    ax.set_title(title)
    return ax


def mass_flow(
    P_up: float | np.ndarray, P_down: float, A: float, gamma: float, R: float, T: float
) -> float | np.ndarray:
    """Isentropic mass flow rate [kg/s] through an opening of area A, choked or subsonic."""
    if np.ndim(P_up) > 0:  # array of upstream pressures: evaluate element by element
        return np.array([mass_flow(p, P_down, A, gamma, R, T) for p in P_up])
    if P_up <= 0:
        return 0.0
    pr = P_down / P_up
    if pr >= 1:  # no pressure drop, no flow (the subsonic formula is only valid for pr < 1)
        return 0.0
    crit_pr = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    if pr <= crit_pr:  # choked
        return A * P_up * np.sqrt(gamma / (R * T)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    # subsonic
    return A * P_up * np.sqrt(2 * gamma / (R * T * (gamma - 1))) * (pr ** (2 / gamma) - pr ** ((gamma + 1) / gamma)) ** 0.5


def plot_thruster_masses(t: np.ndarray, m_t: np.ndarray, m_ch: np.ndarray, title: str) -> tuple[Axes, Axes]:
    """Plot tank mass m_t(t) and chamber mass m_ch(t) on twin y-axes coloured like their curves."""
    _, ax1 = plt.subplots()
    ax1.plot(t, m_t, color="tab:blue")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Tank Mass $m_t$ [kg]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(t, m_ch, color="tab:red")
    ax2.set_ylabel("Chamber Mass $m_{ch}$ [kg]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.grid(False)  # keep only the grid of the tank-mass axis
    ax1.set_title(title)
    return ax1, ax2


def plot_objective(A_t_vals: np.ndarray, obj_vals: list[float] | np.ndarray) -> Axes:
    """Plot the objective function against the nozzle throat area A_t on log-log axes."""
    _, ax = plt.subplots()
    ax.loglog(A_t_vals, obj_vals)
    ax.set_xlabel("$A_t$ (m$^2$)")
    ax.set_ylabel("Objective Value")
    ax.set_title("Objective Function vs. Nozzle Throat Area")
    return ax


def plot_flow_fit(m_t_data: np.ndarray, m_dot_in_data: np.ndarray, m_dot_in_fit: np.ndarray) -> Axes:
    """Scatter the noisy inflow data against tank mass and overlay the least-squares fit."""
    _, ax = plt.subplots()
    ax.scatter(m_t_data, m_dot_in_data, label="Noisy Data")
    ax.plot(m_t_data, m_dot_in_fit, color="tab:red", label="Least-Squares Fit")
    ax.set_xlabel("Tank Mass $m_t$ [kg]")
    ax.set_ylabel(r"Inflow Rate $\dot{m}_{in}$ [kg/s]")
    ax.set_title("Fitting Thruster Valve Friction (Overdetermined → Least-Squares)")
    ax.legend()
    return ax
