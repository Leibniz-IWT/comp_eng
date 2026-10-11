"""Helpers for chapter 15 (feature selection for the hot strip mill data): loading, VIF filter, figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from threadpoolctl import threadpool_limits

VIF_INF = 1e4  # VIF >= 1e4 (R^2 >= 0.9999) is reported as inf, as in the original per-feature regressions


def load_data(filenames: list[str], target: str) -> pd.DataFrame:
    """Read the CSV parts (paths relative to the notebooks/ directory) and concatenate them."""
    dfs, log = [], []
    for filename in filenames:
        log.append(f"Loading {filename}...")
        dfs.append(pd.read_csv(filename, low_memory=False))  # infer each column type from all rows at once
    df = pd.concat(dfs, ignore_index=True)
    log.append(f"Total rows loaded: {len(df)}")
    print("\n".join(log))  # printed at once: a single output block on the website
    assert target in df.columns, f"Target variable {target} not found in dataset columns."
    return df


def complete_cases(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split df into the features X and the target y, keeping only rows without missing values."""
    X = df.drop(columns=[target])
    y = df[target]
    mask = ~(X.isna().any(axis=1) | y.isna())
    return X[mask], y[mask]


def calculate_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return the VIF of every feature on the rows without missing values, sorted from highest to lowest."""
    X = df[features].dropna().to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    varying = np.ptp(X, axis=0) > 0
    vif = np.full(len(features), np.inf)  # a constant column is explained by the intercept alone: R^2 = 1
    Xs = X[:, varying] / np.linalg.norm(X[:, varying], axis=0)
    # 1/(1 - R_i^2) of the regression of X_i on all other features equals the diagonal element [C^-1]_ii
    # of the inverse correlation matrix C = Xs^T Xs, so one eigendecomposition replaces one regression per
    # feature. The eigenvalue floor turns exact collinearity (eigenvalue 0) into a huge VIF.
    eigval, eigvec = np.linalg.eigh(Xs.T @ Xs)
    vif[varying] = (eigvec**2 / np.maximum(eigval, 1e-12)).sum(axis=1)
    vif[vif >= VIF_INF] = np.inf
    vif_df = pd.DataFrame({"Feature": features, "VIF": vif})
    return vif_df.sort_values("VIF", ascending=False, kind="stable")  # equal VIFs keep the column order


def remove_high_vif_features(
    df: pd.DataFrame, features: list[str], threshold: float = 10, max_iterations: int = 100
) -> list[str]:
    """Iteratively remove the feature with the highest VIF until all VIF are <= threshold."""
    remaining, log = list(features), []
    with threadpool_limits(limits=1, user_api="blas"):  # one BLAS thread is fastest for these small matrices
        for iteration in range(1, max_iterations + 1):
            feature, max_vif = calculate_vif(df, remaining).iloc[0]  # ties (VIF = inf): first in column order
            prefix = f"Iteration {iteration} - {len(remaining)} features :"
            if max_vif <= threshold:
                log.append(f"{prefix} All VIF <= {threshold}")
                break
            log.append(f"{prefix} Max VIF = {max_vif:.2f}, removing '{feature}'")
            remaining.remove(feature)
        else:
            log.append(f"Stopped after {max_iterations} iterations: VIF above {threshold} remain")
    print("\n".join(log))  # printed at once: a single output block on the website
    return remaining


def plot_lasso_coefficients(coef_df: pd.DataFrame, n_total: int, alpha: float, cutoff: float) -> Axes:
    """Bar chart of the Lasso coefficients in coef_df (sorted by magnitude), negative ones in red."""
    _, ax = plt.subplots(figsize=(9.0, max(4.5, 0.22 * len(coef_df) + 1.3)))
    colors = ["red" if c < 0 else "blue" for c in coef_df["Coefficient"]]
    ax.barh(range(len(coef_df)), coef_df["Coefficient"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(coef_df)), coef_df["Feature"], fontsize=10)
    ax.axvline(0, color="black", alpha=0.3)
    ax.grid(visible=False, axis="y")
    ax.set_xlabel("Lasso Regression Coefficient")
    ax.set_ylabel("Features")
    ax.set_title(f"Lasso Regression Coefficients (|coef| >= {cutoff})\nAlpha = {alpha:.6f}")
    ax.text(0.98, 0.02, f"Features shown: {len(coef_df)} / {n_total}", transform=ax.transAxes, ha="right", va="bottom")
    return ax


def plot_rf_importance(plot_df: pd.DataFrame, n_total: int, cutoff: float, train_score: float) -> Axes:
    """Bar chart of the random forest importances in plot_df (sorted descending), most important on top."""
    _, ax = plt.subplots(figsize=(9.0, max(4.5, 0.18 * len(plot_df) + 1.3)))
    ax.barh(range(len(plot_df)), plot_df["Importance"], color="forestgreen", alpha=0.7)
    ax.set_yticks(range(len(plot_df)), plot_df["Feature"], fontsize=10)
    ax.invert_yaxis()  # most important feature on top, as in the Lasso figure
    ax.grid(visible=False, axis="y")
    ax.set_xlabel("Feature Importance (MDI)")
    ax.set_ylabel("Features")
    ax.set_title(f"Random Forest Feature Importance - Top {len(plot_df)} Features\n(n_estimators=100, max_depth=15)")
    stats = f"Features shown: {len(plot_df)} / {n_total}\nImportance threshold: {cutoff}\nR² score: {train_score:.4f}"
    ax.text(0.98, 0.02, stats, transform=ax.transAxes, ha="right", va="bottom")
    return ax


def compare_lasso_rf(
    lasso_columns: list[str], coef_df: pd.DataFrame, importance_df: pd.DataFrame, plot_df: pd.DataFrame, top_n: int
) -> None:
    """Print how many Lasso-selected features are among the top random forest features."""
    print("=" * 60)
    print("Comparison: Random Forest vs Lasso Feature Selection")
    print("=" * 60)
    print(f"Total features after preprocessing: {len(importance_df)}")
    print(f"Features selected by Lasso: {len(lasso_columns)}")
    print(f"Top {top_n} features by Random Forest importance: {len(plot_df)}")

    overlap = set(lasso_columns) & set(plot_df["Feature"])
    print(f"\nOverlap between Lasso and top {top_n} RF features: {len(overlap)}")
    print(f"Overlap percentage (of Lasso features): {len(overlap) / len(lasso_columns) * 100:.1f}%")
    print(f"Overlap percentage (of top RF features): {len(overlap) / len(plot_df) * 100:.1f}%")
    if overlap:
        print("\nCommon features (alphabetically sorted):")
        coef = coef_df.set_index("Feature")["Coefficient"]
        importance = importance_df.set_index("Feature")["Importance"]
        for feature in sorted(overlap):
            print(f"  - {feature:<20} | Lasso: {coef[feature]:7.4f} | RF: {importance[feature]:.6f}")
