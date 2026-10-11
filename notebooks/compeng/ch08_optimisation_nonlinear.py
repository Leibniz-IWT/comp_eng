"""Helper code for chapter 8, non-linear optimization (``08_optimisation-nonlinear.ipynb``).

Plotting helpers, the Rosenbrock test problem, the synthetic data sets and the
small neural network used in the notebook. The notebook itself keeps the code
that carries the concepts: the SymPy KKT derivation, the trust-region loop, the
``minimize`` / ``least_squares`` calls, backpropagation and the SGD update.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, ConnectionPatch, Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

Func2D = Callable[[np.ndarray, np.ndarray], np.ndarray]


# --- 8.3 KKT conditions ------------------------------------------------------------


def plot_kkt_geometry(
    f: Func2D,
    g: Func2D,
    x_kkt: tuple[float, float],
    grad_f: np.ndarray,
    grad_g: np.ndarray,
    lam: float,
    scale: float = 0.5,
) -> Axes:
    """Contours of f, feasible region g <= 0 and the balanced gradients at a KKT point."""
    x = np.linspace(-1, 2, 100)
    X, Y = np.meshgrid(x, x)
    G = g(X, Y)
    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, f(X, Y), levels=35, cmap="magma")
    fig.colorbar(cs, ax=ax, label="Objective value")
    ax.contour(X, Y, G, levels=[0], colors="r", linestyles="--")
    ax.contourf(X, Y, G, levels=[-np.inf, 0], colors="gray", alpha=0.3)
    tangent = np.array([-grad_g[1], grad_g[0]]) / np.linalg.norm(grad_g)  # along the boundary g = 0
    s = np.array([-3.0, 3.0])
    ax.plot(x_kkt[0] + s * tangent[0], x_kkt[1] + s * tangent[1], "--", color="tab:orange",
            label="Tangent line of g at KKT point")
    ax.plot(*x_kkt, "ro", markersize=8, label=f"KKT point ({x_kkt[0]}, {x_kkt[1]})")
    ax.arrow(*x_kkt, *(scale * grad_f), head_width=0.05, color="tab:blue", label="∇f")
    ax.arrow(*x_kkt, *(scale * lam * grad_g), head_width=0.05, color="tab:red", label="λ ∇g")
    handles, _ = ax.get_legend_handles_labels()
    fig.legend(handles=[Patch(color="gray", alpha=0.3, label="Feasible region"), *handles],
               loc="outside right upper")
    ax.set(xlim=(-1, 2), ylim=(-1, 2), aspect="equal", xlabel="$x$", ylabel="$y$",
           title="Graphical Demonstration of KKT Conditions")
    return ax


def plot_constrained_surface(
    f: Func2D,
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    minimum: tuple[float, float, float],
    focused: bool = False,
) -> Axes:
    """3D surface of f with the constrained path; ``focused`` fades the surface and adds a height ribbon."""
    x = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, x)
    curve_z = f(curve_x, curve_y)
    fig = plt.figure(figsize=(9, 6))
    # draw in the order added, so the path is never hidden behind the (transparent) surface
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.plot_surface(X, Y, f(X, Y), cmap="viridis", alpha=0.1 if focused else 0.7)
    if focused:  # ribbon between the path and the z = 0 plane
        p = np.column_stack([curve_x, curve_y, curve_z])
        verts = [[(*a[:2], 0), tuple(a), tuple(b), (*b[:2], 0)] for a, b in zip(p[:-1], p[1:], strict=True)]
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.5, color="gray"))
    ax.plot(curve_x, curve_y, curve_z, color="tab:red", linewidth=3, label="Constrained path (h=0)")
    ax.scatter(*minimum, color="k", s=100, label="Constrained minimum")
    if focused:
        title = "Focused View: Constrained Path with Height Fill"
    else:
        title = "Full View: Objective Surface and Constrained Path"
    ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$f(x_1, x_2)$", title=title,
           xticks=np.arange(-2, 3), yticks=np.arange(-2, 3))
    ax.set_box_aspect(None, zoom=0.9)  # keeps the z label inside the figure
    ax.legend(loc="upper left")
    ax.view_init(elev=20, azim=120)
    return ax


def plot_constrained_contours(
    f: Func2D,
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    minimum: tuple[float, float],
    grad_f: np.ndarray,
    grad_h: np.ndarray,
    half_width: float = 0.3,
) -> tuple[Axes, Axes]:
    """Top view: filled contours of f with the constrained path, and a zoom on the level set of f tangent to it."""
    x0, y0 = minimum
    f_min = f(x0, y0)
    X, Y = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-2, 2, 300))
    XZ, YZ = np.meshgrid(np.linspace(x0 - half_width, x0 + half_width, 300),
                         np.linspace(y0 - half_width, y0 + half_width, 300))
    FZ = f(XZ, YZ)
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(9, 5.2))
    # filled contours (the viridis surface seen from above); they also keep the PNG small
    ax_full.contourf(X, Y, f(X, Y), levels=20, cmap="viridis")
    ax_full.add_patch(Rectangle((x0 - half_width, y0 - half_width), 2 * half_width, 2 * half_width,
                                fill=False, ec="w", lw=1.5))
    ax_zoom.contourf(XZ, YZ, FZ, levels=np.linspace(FZ.min(), f_min + 0.6, 12), cmap="viridis", extend="max")
    outline = [patheffects.withStroke(linewidth=4, foreground="k")]  # white dashes visible on any background
    ax_zoom.contour(XZ, YZ, FZ, levels=[f_min], colors="w", linestyles="--", linewidths=2).set(path_effects=outline)
    for ax in (ax_full, ax_zoom):
        ax.plot(curve_x, curve_y, color="tab:red", linewidth=3, label="Constrained path (h=0)")
        ax.plot(x0, y0, "ko", markersize=8, markeredgecolor="w", zorder=4, label="Constrained minimum")
    ax_zoom.plot([], [], "w--", linewidth=2, path_effects=outline, label=f"Level set $f = {f_min:.2f}$")
    mu = grad_f @ grad_h / (grad_h @ grad_h)  # grad_f = mu grad_h (projection)
    scale = 0.5 * half_width
    for v, color, name in ((grad_h, "tab:red", r"$\nabla h$"), (grad_f, "tab:blue", rf"$\nabla f = {mu:.2f}\,\nabla h$")):
        ax_zoom.arrow(x0, y0, *(scale * v), width=0.02 * half_width, head_width=0.08 * half_width,
                      length_includes_head=True, color=color, ec="k", lw=0.5, zorder=3, label=name)
    ax_full.set(xlim=(-2, 2), ylim=(-2, 2), title="Top View: Contours of $f$")
    ax_zoom.set(xlim=(x0 - half_width, x0 + half_width), ylim=(y0 - half_width, y0 + half_width),
                title="Zoom: Tangency at the Minimum")
    for ax in (ax_full, ax_zoom):
        ax.set(aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
    fig.legend(*ax_zoom.get_legend_handles_labels(), loc="outside lower center", ncols=3)
    return ax_full, ax_zoom


def plot_reduced_objective(t: np.ndarray, f_t: np.ndarray, minimum: tuple[float, float]) -> Axes:
    """Objective along the constraint path, f(r(t)), with the constrained minimum."""
    fig, ax = plt.subplots()
    ax.plot(t, f_t, color="tab:red", linewidth=3, label=r"$f(\mathbf{r}(t))$")
    ax.plot(*minimum, "ko", markersize=10, label="Constrained minimum")
    ax.set(xlabel="$t$", ylabel=r"$f(\mathbf{r}(t))$")
    ax.legend()
    return ax


# --- 8.4 trust-region methods --------------------------------------------------------


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """Rosenbrock function f(x) = 100 (x2 - x1^2)^2 + (1 - x1)^2, global minimum f(1, 1) = 0."""
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def grad_rosenbrock(x: np.ndarray) -> np.ndarray:
    """Analytical gradient of the Rosenbrock function."""
    return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])


def bfgs_update(B: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """BFGS update of the Hessian approximation B, skipped unless y^T s > 0."""
    if np.dot(y, s) > 0:
        B = B - np.outer(B.dot(s), s.dot(B)) / np.dot(s, B.dot(s)) + np.outer(y, y) / np.dot(y, s)
    return B


def plot_rosenbrock_path(path: np.ndarray) -> Axes:
    """Iterates of an optimizer on log-spaced contours of the Rosenbrock function."""
    x = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, x)
    fig, ax = plt.subplots()
    ax.contour(X, Y, rosenbrock(np.array([X, Y])), levels=np.logspace(0, 5, 35), cmap="viridis", norm=LogNorm())
    ax.plot(path[:, 0], path[:, 1], "r-o", label="Iteration path")
    ax.plot(1, 1, "k*", markersize=14, markeredgecolor="w", zorder=3, label="Global min (1,1)")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title="Trust-Region on Rosenbrock Function")
    ax.legend(loc="lower right", frameon=True)
    return ax


# --- 8.6 non-linear least squares ----------------------------------------------------


def decay_model(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential decay y = a exp(-b t) + c."""
    return a * np.exp(-b * t) + c


def make_decay_data(
    true_params: Sequence[float], n: int = 20, noise: float = 0.5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Noisy samples of the decay model on t in [0, 10] (same numbers as ``np.random.seed(seed)``)."""
    t = np.linspace(0, 10, n)
    return t, decay_model(t, *true_params) + noise * np.random.RandomState(seed).randn(n)


def plot_decay_fit(t: np.ndarray, y: np.ndarray, true_params: Sequence[float], fit_params: Sequence[float]) -> Axes:
    """Noisy data with the true and the fitted decay model."""
    t_fine = np.linspace(t.min(), t.max(), 200)
    fig, ax = plt.subplots()
    ax.scatter(t, y, label="Noisy data")
    ax.plot(t_fine, decay_model(t_fine, *true_params), "k--", label="True model")
    ax.plot(t_fine, decay_model(t_fine, *fit_params), "r-", label="Fitted model")
    ax.set(xlabel="Time $t$", ylabel="Measurement $y$", title="NLLS Fit: Exponential Decay")
    ax.legend()
    return ax


# --- 8.7 neural networks and backpropagation -----------------------------------------


def plot_sigmoid(sigmoid: Callable[[np.ndarray], np.ndarray]) -> Axes:
    """Plot an activation function on z in [-10, 10]."""
    z = np.linspace(-10, 10, 100)
    fig, ax = plt.subplots()
    ax.plot(z, sigmoid(z))
    ax.set(xlabel="$z$", ylabel=r"$\sigma(z)$", title="Sigmoid Activation Function")
    return ax


def draw_nn() -> Axes:
    """Diagram of a 2-3-1 feedforward network with nodes a^(l)_i and weighted connections."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set(xlim=(0, 4), ylim=(0, 4), aspect="equal")
    ax.axis("off")
    layers, x_pos, radius = (2, 3, 1), (0.5, 2.0, 3.5), 0.2
    nodes = [[np.array([x, 2 + i - (n - 1) / 2]) for i in range(n)] for x, n in zip(x_pos, layers, strict=True)]
    for k, layer in enumerate(nodes):
        for i, p in enumerate(layer):
            ax.add_patch(Circle(p, radius, color="lightblue", ec="black"))
            ax.text(*p, f"$a^{{({k})}}_{i}$", ha="center", va="center")
    for k in range(len(layers) - 1):
        for i, pa in enumerate(nodes[k]):
            for j, pb in enumerate(nodes[k + 1]):
                u = (pb - pa) / np.linalg.norm(pb - pa)
                ax.add_patch(ConnectionPatch(pa + u * radius, pb - u * radius, "data", "data", arrowstyle="->",
                                             shrinkA=0, shrinkB=0, mutation_scale=20, fc="w"))
                if k == 0 and i == 0 and j == 0:  # label one example weight
                    ax.text(*((pa + pb) / 2 + [0, 0.1]), "w", fontsize=10)
    for x, name in zip(x_pos, ("Input Layer", "Hidden Layer", "Output Layer"), strict=True):
        ax.text(x, 0.5, name, ha="center")
    ax.set_title("Simple Feedforward Neural Network Architecture")
    return ax


def make_sine_data(rng: np.random.RandomState, n: int = 200, noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Noisy samples of y = sin(2 pi x) + 0.1 x^3 on x in [-1, 1], as column vectors."""
    x = np.linspace(-1, 1, n)[:, np.newaxis]
    return x, np.sin(2 * np.pi * x) + 0.1 * x**3 + noise * rng.randn(n, 1)


def init_params(sizes: Sequence[int], rng: np.random.RandomState) -> list[np.ndarray]:
    """He-initialized weights W^(l) (n_l x n_(l-1)) and zero biases b^(l) (1 x n_l), as [W1, b1, W2, b2, ...]."""
    theta: list[np.ndarray] = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True):
        theta += [rng.randn(n_out, n_in) * np.sqrt(2 / n_in), np.zeros((1, n_out))]
    return theta


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation max(0, z)."""
    return np.maximum(0, z)


def relu_deriv(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU (1 for z > 0, else 0)."""
    return (z > 0).astype(float)


def forward(x: np.ndarray, theta: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Forward pass of the 2-hidden-layer ReLU network; returns y_hat and the cache [z1, a1, z2, a2, z3, y_hat]."""
    W1, b1, W2, b2, W3, b3 = theta
    z1 = x @ W1.T + b1
    a1 = relu(z1)
    z2 = a1 @ W2.T + b2
    a2 = relu(z2)
    z3 = a2 @ W3.T + b3
    y_hat = z3  # linear output for regression
    return y_hat, [z1, a1, z2, a2, z3, y_hat]


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean squared error loss with the factor 1/2."""
    return float(np.mean((y - y_hat) ** 2 / 2))


def plot_nn_fit(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Axes:
    """Noisy training data and the network prediction."""
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.5, label="Noisy data")
    ax.plot(x, y_pred, "r-", label="NN fit")
    ax.set(xlabel="$x$", ylabel="$y$", title="NN Approximation of Non-Linear Function")
    ax.legend()
    return ax


def plot_loss(losses: Sequence[float]) -> Axes:
    """Training loss per epoch on a logarithmic axis."""
    fig, ax = plt.subplots()
    ax.semilogy(losses)
    ax.set(xlabel="Epoch", ylabel="MSE Loss", title="Training Loss")
    return ax


# --- 8.8 stochastic gradient descent -------------------------------------------------


def make_quadratic_regression_data(
    A: np.ndarray, b: np.ndarray, sigma: float, n: int, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Noisy samples of f(x) = 0.5 x^T A x + b^T x with features [x1^2, x1 x2, x2^2, x1, x2, 1]."""
    X = rng.randn(n, 2)
    y = np.array([0.5 * xi.T @ A @ xi + b.T @ xi + sigma * rng.randn() for xi in X])
    X_quad = np.c_[X[:, 0] ** 2, X[:, 0] * X[:, 1], X[:, 1] ** 2, X, np.ones(len(X))]
    return X_quad, y


def plot_sgd_path(A: np.ndarray, b: np.ndarray, path: np.ndarray, x_true: np.ndarray) -> Axes:
    """SGD iterates on contours of the noise-free quadratic 0.5 x^T A x + b^T x."""
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-4, 4, 100))
    Z = 0.5 * (A[0, 0] * xx**2 + 2 * A[0, 1] * xx * yy + A[1, 1] * yy**2) + b[0] * xx + b[1] * yy
    fig, ax = plt.subplots()
    ax.contour(xx, yy, Z, levels=20, cmap="viridis")
    ax.plot(path[:, 0], path[:, 1], "r-o", markersize=4, label="SGD path")
    ax.plot(*x_true, "k*", markersize=14, markeredgecolor="w", zorder=3, label="True min")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title="SGD on Noisy Quadratic")
    ax.legend(loc="upper right", frameon=True)
    return ax
