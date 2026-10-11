"""Helpers for chapter 10 (classical feedback control): figures and widget callbacks.

The notebook keeps the control calls visible (``ct.tf``, ``ct.feedback``,
``ct.step_response``, ``ct.frequency_response``, ``ct.margin``,
``ct.root_locus_map``). The functions here only turn the python-control data
objects those calls return into figures.
"""

from __future__ import annotations

import control as ct
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.ticker import MultipleLocator

SETPOINT_STYLE: dict[str, object] = {"color": "tab:red", "ls": "--", "label": "Setpoint"}
RE_LABEL = r"Re$(s) = \sigma$"
IM_LABEL = r"Im$(s) = \omega$"


def plot_laplace_integrands(t: np.ndarray, f_t: np.ndarray, sigmas: list[float]) -> np.ndarray:
    """Plot the integrand f(t) e^(-sigma t) with the area under it shaded, one panel per sigma."""
    fig, axs = plt.subplots(len(sigmas), 1, figsize=(9, 7), sharex=True)
    for ax, sigma in zip(axs, sigmas, strict=True):
        integrand = f_t * np.exp(-sigma * t)  # s = sigma (omega = 0): decay only, no rotation
        ax.plot(t, integrand, color="tab:blue")
        ax.fill_between(t, integrand, color="tab:blue", alpha=0.3)
        ax.set_title(rf"Integrand for $\sigma = {sigma}$ (so $s = {sigma}$), shaded area approximates integral")
        ax.set_ylabel(r"$f(t)\,e^{-\sigma t}$")
    axs[-1].set_xlabel("Time $t$ (s)")
    return axs


def plot_block_diagram() -> Axes:
    """Draw the unity-feedback loop R(s) -> C(s) -> G(s) -> Y(s) with its feedback path."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set(xlim=(0, 10), ylim=(0.3, 2.8), aspect="equal")  # equal aspect keeps the junction circular
    ax.axis("off")
    arrow = {"arrowstyle": "->", "mutation_scale": 20, "lw": 1.5, "color": "k"}
    line = {"color": "k", "lw": 1.5}
    ax.add_patch(Circle((1.2, 2), 0.3, fill=False, lw=1.5))  # summing junction
    for x0, label in ((2.6, "$C(s)$"), (5.2, "$G(s)$")):
        ax.add_patch(Rectangle((x0, 1.5), 1.5, 1, fill=False, lw=1.5))
        ax.text(x0 + 0.75, 2, label, ha="center", va="center", fontsize=14)
    for start, end in (((0.1, 2), (0.9, 2)), ((1.5, 2), (2.6, 2)), ((4.1, 2), (5.2, 2)),
                       ((6.7, 2), (9.0, 2)), ((1.2, 0.6), (1.2, 1.7))):
        ax.add_patch(FancyArrowPatch(start, end, **arrow))
    ax.plot([8, 8, 1.2], [2, 0.6, 0.6], **line)  # feedback path from the branch point
    ax.plot(8, 2, "ko", ms=5)
    ax.text(0.78, 2.12, "+", ha="center", va="bottom", fontsize=16)
    ax.text(0.98, 1.45, "\N{MINUS SIGN}", ha="right", va="center", fontsize=16)
    for x, label in ((0.45, "$R(s)$"), (2.05, "$E(s)$"), (4.65, "$U(s)$"), (7.4, "$Y(s)$")):
        ax.text(x, 2.12, label, ha="center", va="bottom", fontsize=13)
    ax.text(4.6, 0.72, "Feedback ($H(s)=1$)", ha="center", va="bottom", fontsize=12)
    return ax


def step_metrics(t: np.ndarray, y: np.ndarray, y_final: float, setpoint: float = 1.0,
                 band: float = 0.02) -> dict[str, float]:
    """Rise time, overshoot, settling time and steady-state error as defined in the notebook text."""
    t, y = np.asarray(t), np.asarray(y)  # plain arrays (python-control returns NamedSignal objects)
    t_rise =t[np.argmax(y >= 0.9 * y_final)] - t[np.argmax(y >= 0.1 * y_final)]  # 10 % to 90 % of final value
    overshoot = max(0.0, (y.max() - setpoint) / setpoint * 100)  # percentage above the setpoint
    outside = np.flatnonzero(np.abs(y - y_final) > band * abs(y_final))
    t_settle = t[min(outside[-1] + 1, len(t) - 1)] if outside.size else t[0]  # stays inside the band after this
    return {"rise_time": t_rise, "overshoot": overshoot, "settling_time": t_settle, "ss_error": setpoint - y_final}


def plot_step_metrics(resp: ct.TimeResponseData, y_final: float, setpoint: float = 1.0) -> Axes:
    """Plot a step response with the setpoint, the 2 % settling band and the four metrics annotated."""
    m = step_metrics(resp.time, resp.outputs, y_final, setpoint)
    fig, ax = plt.subplots()
    ax.plot(resp.time, resp.outputs, label="Output $y(t)$")
    ax.axhline(setpoint, **{**SETPOINT_STYLE, "label": "Setpoint $r(t)=1$"})
    ax.axhspan(0.98 * y_final, 1.02 * y_final, color="tab:gray", alpha=0.25, label=r"$\pm 2\,\%$ of final value")
    ax.axvline(m["settling_time"], color="tab:gray", ls=":")
    summary = (f"Rise time: {m['rise_time']:.2f} s\nOvershoot: {m['overshoot']:.1f} %\n"
               f"Settling time: {m['settling_time']:.2f} s\nSteady-state error: {m['ss_error']:.3f}")
    ax.text(0.97, 0.05, summary, transform=ax.transAxes, ha="right", va="bottom",
            bbox={"boxstyle": "round", "fc": "white", "ec": "0.7"})
    ax.set(xlabel="Time (s)", ylabel="Response", title="Step Response with Metrics", ylim=(-0.05, 1.5))
    ax.legend(loc="upper right")
    return ax


def plot_step_responses(responses: dict[str, ct.TimeResponseData], title: str, setpoint: float = 1.0,
                        ylim: tuple[float, float] | None = None) -> Axes:
    """Overlay labelled step responses together with the setpoint line."""
    fig, ax = plt.subplots()
    for label, resp in responses.items():
        ax.plot(resp.time, resp.outputs, label=label)
    ax.axhline(setpoint, **SETPOINT_STYLE)
    ax.set(xlabel="Time (s)", ylabel="Output", title=title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="upper right")
    return ax


def plot_pole_zero_maps(maps: list[ct.PoleZeroData], titles: list[str], lim: float = 1.5) -> np.ndarray:
    """Side-by-side pole-zero maps (x poles, o zeros) on equal axes, imaginary axis dashed red."""
    fig, axs = plt.subplots(1, len(maps), figsize=(10, 3.8), sharex=True, sharey=True)
    for ax, pz, title in zip(axs, maps, titles, strict=True):
        ax.axvline(0, color="tab:red", ls="--")
        ax.plot(pz.poles.real, pz.poles.imag, "x", ms=11, mew=2.5, color="tab:blue")
        ax.plot(pz.zeros.real, pz.zeros.imag, "o", ms=10, mew=2, mfc="none", color="tab:blue")
        ax.set(title=title, xlabel=RE_LABEL, xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
    axs[0].set_ylabel(IM_LABEL)
    return axs


def plot_bode_margins(resp: ct.FrequencyResponseData, gm: float, pm: float, w_pc: float, w_gc: float,
                      title: str) -> tuple[Axes, Axes]:
    """Bode magnitude/phase from a frequency response with the gain margin and phase margin marked."""
    w, mag_db = resp.omega, 20 * np.log10(resp.magnitude)
    phase_deg = np.degrees(np.unwrap(resp.phase))  # python-control returns the phase wrapped to (-180, 180]
    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax_mag.semilogx(w, mag_db)
    ax_ph.semilogx(w, phase_deg)
    ax_mag.axhline(0, color="tab:red", ls="--")  # 0 dB line
    ax_ph.axhline(-180, color="tab:red", ls="--")  # -180 deg line
    arrow = {"arrowstyle": "<->", "color": "k", "lw": 1.5}
    box = {"fc": "white", "ec": "none", "alpha": 0.85}
    if np.isfinite(w_pc):  # GM: distance of |G| below 0 dB at the phase crossover frequency
        m_pc = np.interp(np.log(w_pc), np.log(w), mag_db)
        ax_mag.annotate("", xy=(w_pc, 0), xytext=(w_pc, m_pc), arrowprops=arrow)
        if m_pc > 0:  # negative GM: label below the 0 dB line, left of the arrow (curve is above)
            ax_mag.text(w_pc / 1.15, -6, f"GM: {20 * np.log10(gm):.2f} dB", ha="right", va="top", bbox=box)
        else:  # positive GM: label between curve and 0 dB line, right of the arrow
            ax_mag.text(w_pc * 1.15, m_pc / 2, f"GM: {20 * np.log10(gm):.2f} dB", ha="left", va="center", bbox=box)
    else:
        ax_mag.text(0.97, 0.95, r"GM: $\infty$", transform=ax_mag.transAxes, ha="right", va="top", bbox=box)
    if np.isfinite(w_gc):  # PM: distance of the phase above -180 deg at the gain crossover frequency
        p_gc = np.interp(np.log(w_gc), np.log(w), phase_deg)
        ax_ph.annotate("", xy=(w_gc, -180), xytext=(w_gc, p_gc), arrowprops=arrow)
        y = (p_gc - 180) / 2 if pm > 0 else p_gc - 8  # left of the arrow, below the curve
        ax_ph.text(w_gc / 1.15, y, f"PM: {pm:.1f}°", ha="right", va="center", bbox=box)
    for wc, label in ((w_pc, r"$\omega_{pc}$"), (w_gc, r"$\omega_{gc}$")):
        if np.isfinite(wc):
            for ax in (ax_mag, ax_ph):
                ax.axvline(wc, color="tab:gray", ls=":")
            ax_mag.text(wc, 1.01, label, transform=ax_mag.get_xaxis_transform(), ha="center", va="bottom")
    ax_mag.set_title(title, pad=22)  # room for the crossover labels above the axes
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_ph.set(xlabel="Frequency (rad/s)", ylabel="Phase (deg)")
    ax_ph.yaxis.set_major_locator(MultipleLocator(45))
    return ax_mag, ax_ph


def plot_root_locus(rl: ct.PoleZeroData, title: str) -> Axes:
    """Plot a root locus from ct.root_locus_map data: branches, open-loop poles (x) and zeros (o)."""
    r = max(1.0, np.abs(np.concatenate([rl.poles, rl.zeros])).max())
    fig, ax = plt.subplots(figsize=(7, 5.4))  # matches the 5:4 equal-aspect view
    ax.axhline(0, color="0.5", lw=1)
    ax.axvline(0, color="0.5", lw=1)  # imaginary axis
    ax.plot(rl.loci.real, rl.loci.imag, color="tab:blue")  # one column per branch
    ax.plot(rl.poles.real, rl.poles.imag, "x", ms=11, mew=2.5, color="k")
    ax.plot(rl.zeros.real, rl.zeros.imag, "o", ms=10, mew=2, mfc="none", color="k")
    ax.set(xlim=(-4 * r, r), ylim=(-2 * r, 2 * r), aspect="equal", xlabel=RE_LABEL, ylabel=IM_LABEL, title=title)
    return ax


def pid_tf(Kp: float, Ki: float, Kd: float) -> ct.TransferFunction:
    """PID controller C(s) = (Kd s^2 + Kp s + Ki) / s, or the pure gain Kp when Ki = Kd = 0."""
    return ct.tf([Kd, Kp, Ki], [1, 0]) if Ki > 0 or Kd > 0 else ct.tf([Kp], [1])


def pid_step_callback(G: ct.TransferFunction, t: np.ndarray, Kp: float, Ki: float, Kd: float) -> None:
    """Widget callback: closed-loop step response of plant G under PID control (returns None for interact)."""
    resp = ct.step_response(ct.feedback(pid_tf(Kp, Ki, Kd) * G, 1), t)
    plot_step_responses({"Output": resp}, title=f"PID: Kp={Kp}, Ki={Ki}, Kd={Kd}", ylim=(0, 2))


def zn_step_response(G: ct.TransferFunction, t: np.ndarray, Ku: float, Pu: float, kp_factor: float,
                     ti_factor: float | None = None, td_factor: float | None = None) -> ct.TimeResponseData:
    """Closed-loop step response with Ziegler-Nichols gains Kp = f Ku, Ti = Pu / f_i, Td = Pu / f_d."""
    Kp = kp_factor * Ku
    Ki = Kp / (Pu / ti_factor) if ti_factor is not None else 0.0  # Ki = Kp / Ti
    Kd = Kp * (Pu / td_factor) if td_factor is not None else 0.0  # Kd = Kp * Td
    return ct.step_response(ct.feedback(pid_tf(Kp, Ki, Kd) * G, 1), t)
