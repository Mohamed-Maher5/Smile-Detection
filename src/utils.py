"""Shared plotting and metrics helper functions."""

import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ── Cohen's d ────────────────────────────────────────────────────────────────

def cohens_d(df: pd.DataFrame, feature_col: str, label_col: str = "label") -> float:
    """Compute Cohen's d for a single feature between two classes.

    d = (mean_1 − mean_0) / pooled_std
    pooled_std = sqrt(((n1−1)*var1 + (n0−1)*var0) / (n1+n0−2))
    """
    g1 = df.loc[df[label_col] == 1, feature_col].dropna()
    g0 = df.loc[df[label_col] == 0, feature_col].dropna()
    n1, n0 = len(g1), len(g0)
    if n1 < 2 or n0 < 2:
        return np.nan
    pooled_var = ((n1 - 1) * g1.var() + (n0 - 1) * g0.var()) / (n1 + n0 - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return np.nan
    return float((g1.mean() - g0.mean()) / pooled_std)


# ── Separability table ──────────────────────────────────────────────────────

_FEATURE_COLS = [
    "mouth_width",
    "inter_eye_dist",
    "mouth_eye_ratio",
    "mouth_vertical_lift",
    "mouth_nose_ratio",
]


def separability_table(df: pd.DataFrame, feature_cols: list[str] | None = None,
                       label_col: str = "label") -> pd.DataFrame:
    """Return a DataFrame of Cohen's d stats for every feature, sorted by |d|."""
    if feature_cols is None:
        feature_cols = _FEATURE_COLS
    rows = []
    for col in feature_cols:
        g1 = df.loc[df[label_col] == 1, col].dropna()
        g0 = df.loc[df[label_col] == 0, col].dropna()
        n1, n0 = len(g1), len(g0)
        pooled_var = ((n1 - 1) * g1.var() + (n0 - 1) * g0.var()) / (n1 + n0 - 2)
        pooled_std = np.sqrt(pooled_var)
        d = float((g1.mean() - g0.mean()) / pooled_std) if pooled_std > 0 else np.nan
        rows.append({
            "feature": col,
            "mean_smile": g1.mean(),
            "mean_non_smile": g0.mean(),
            "pooled_std": pooled_std,
            "cohens_d": d,
        })
    table = pd.DataFrame(rows)
    table["abs_d"] = table["cohens_d"].abs()
    table = table.sort_values("abs_d", ascending=False).drop(columns="abs_d").reset_index(drop=True)
    return table


def print_interpretation(table: pd.DataFrame) -> None:
    """Print a plain-English interpretation of the separability table."""
    print("\nInterpretation (Cohen's d):")
    print("  |d| > 0.8  → large effect")
    print("  |d| 0.5–0.8 → medium effect")
    print("  |d| 0.2–0.5 → small effect")
    print("  |d| < 0.2   → negligible\n")

    for _, row in table.iterrows():
        d = row["cohens_d"]
        ad = abs(d)
        if ad > 0.8:
            strength = "LARGE"
        elif ad >= 0.5:
            strength = "MEDIUM"
        elif ad >= 0.2:
            strength = "SMALL"
        else:
            strength = "NEGLIGIBLE"
        sign_note = ""
        if row["feature"] == "mouth_vertical_lift" and d < 0:
            sign_note = " (expected — smiles lift mouth toward eye line)"
        print(f"  {row['feature']:25s}  d = {d:+.4f}  [{strength}]{sign_note}")


# ── Box-plot + histogram separability figure ─────────────────────────────────

def plot_feature_separability(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    save_path: str = "data/eda/feature_separability.png",
) -> None:
    """Grid of boxplots + overlaid histograms, one subplot per feature.

    Each subplot shows label=0 (salmon) and label=1 (teal) distributions with
    the Cohen's d value in the title.
    """
    n = len(feature_cols)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    colors = {0: "#E07070", 1: "#5CB8A5"}
    labels = {0: "non-smile", 1: "smile"}

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        d = cohens_d(df, col, label_col)

        for cls in [0, 1]:
            vals = df.loc[df[label_col] == cls, col].dropna()
            ax.hist(vals, bins=30, alpha=0.35, color=colors[cls],
                    label=labels[cls], density=True, edgecolor="none")

        data_by_class = [df.loc[df[label_col] == c, col].dropna().values for c in [0, 1]]
        bp = ax.boxplot(data_by_class, positions=[1, 2], widths=0.45,
                        patch_artist=True, showfliers=False)
        for patch, cls in zip(bp["boxes"], [0, 1]):
            patch.set_facecolor(colors[cls])
            patch.set_alpha(0.7)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["non-smile", "smile"])
        ax.set_title(f"{col}  (d = {d:+.3f})", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Separability by Class", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_path}")


# ── Per-image quality stats (Stage 1 EDA) ────────────────────────────────────

def compute_image_stats(img: np.ndarray) -> dict:
    """Per-image stats: brightness (mean gray), Laplacian blur variance, width, height."""
    if img is None:
        return {"brightness": np.nan, "blur_val": np.nan, "width": np.nan, "height": np.nan}
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(np.mean(gray)),
        "blur_val": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "width": float(w),
        "height": float(h),
    }


_STAT_COLS = ["brightness", "blur_val", "width", "height"]


def plot_stat_histograms(
    df: pd.DataFrame,
    label_col: str = "label",
    save_path: str = "data/eda/stats_histograms.png",
) -> None:
    """2×2 grid of per-class histograms for brightness, blur, width and height."""
    cols = [c for c in _STAT_COLS if c in df.columns]
    ncols = min(len(cols), 2)
    nrows = int(np.ceil(len(cols) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    colors = {0: "#E07070", 1: "#5CB8A5"}
    labels = {0: "non-smile", 1: "smile"}

    for i, col in enumerate(cols):
        ax = axes[i]
        for cls in [0, 1]:
            vals = df.loc[df[label_col] == cls, col].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=40, alpha=0.5, color=colors[cls], label=labels[cls])
        ax.set_title(col, fontsize=11)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Image Stats (Stage 1 EDA)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_path}")


# ── Stage 3: classifier evaluation, latency benchmark, leaderboard ───────────

def evaluate_classifier(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Evaluate a binary classifier from true labels and positive-class scores.

    Parameters
    ----------
    y_true : array-like of 0/1 labels.
    y_prob : array-like of scores; either predicted probabilities of the
             positive class (``predict_proba``) or decision-function scores.
    threshold : decision threshold applied to ``y_prob`` for the confusion
                matrix (accuracy/precision/recall/f1). ROC-AUC is threshold-
                independent and uses the raw scores.

    Returns a dict with ``accuracy``, ``precision``, ``recall``, ``f1``,
    ``roc_auc``, a ``cm_display`` (sklearn ConfusionMatrixDisplay for
    plotting) and ROC curve data ``roc_fpr`` / ``roc_tpr``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "cm_display": ConfusionMatrixDisplay(
            confusion_matrix(y_true, y_pred), display_labels=["non_smile", "smile"]),
        "roc_fpr": fpr,
        "roc_tpr": tpr,
    }


def latency_benchmark(predict_fn, X_sample, n_iter: int = 10000, warmup: int = 100) -> dict:
    """Per-call latency of ``predict_fn`` on a single feature row.

    Parameters
    ----------
    predict_fn : callable taking an array of shape (1, n_features), e.g.
                 ``model.predict`` (X_sample is a single row repeated).
    n_iter : number of timed iterations after ``warmup``.

    Returns dict with ``mean_ms``, ``median_ms``, ``p95_ms``, ``p99_ms``
    measured with ``time.perf_counter``.
    """
    for _ in range(warmup):
        predict_fn(X_sample)

    ms = np.empty(n_iter)
    for i in range(n_iter):
        t0 = time.perf_counter()
        predict_fn(X_sample)
        ms[i] = (time.perf_counter() - t0) * 1e3

    return {
        "mean_ms": float(ms.mean()),
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
    }


_LB_ROWS: list[dict] = []


class Leaderboard:
    """Accumulator for CV leaderboard rows, materialized as a pandas DataFrame.

    Rows live in the module-level store so updates persist across notebook cells
    even when the same notebook object is re-imported. Standard usage:

        LEADERBOARD = cv_leaderboard()                 # reset + get handle
        LEADERBOARD.add_row(model_name=..., ...)
        LEADERBOARD.print_leaderboard_sorted()
        LEADERBOARD.dataframe()                        # the pandas DataFrame
    """

    def __init__(self, rows: list[dict] | None = None):
        self._rows = _LB_ROWS if rows is None else rows

    def add_row(self, model_name: str, params=None, cv_f1_mean: float | None = None,
                cv_f1_std: float | None = None, latency_median_ms: float | None = None,
                **extra) -> "Leaderboard":
        """Accumulate one row into the global leaderboard."""
        row = {
            "model_name": model_name,
            "params": params,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
            "latency_median_ms": latency_median_ms,
        }
        row.update(extra)
        self._rows.append(row)
        return self

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def print_leaderboard_sorted(self, by: str = "cv_f1_mean",
                                 ascending: bool = False) -> pd.DataFrame:
        """Print leaderboard sorted by ``by`` (default descending F1)."""
        df = self.dataframe()
        if df.empty:
            print("(leaderboard is empty)")
            return df
        df = df.sort_values(by=by, ascending=ascending, na_position="last").reset_index(drop=True)
        print(df.to_string(index=False))
        return df

    def __len__(self) -> int:
        return len(self._rows)


def cv_leaderboard() -> Leaderboard:
    """Reset and return the notebook-global Leaderboard accumulator.

    Also create a module-level alias ``add_row`` / ``print_leaderboard_sorted``
    if you prefer the functional style; the object's methods are the canonical
    API, e.g.:

        LEADERBOARD = cv_leaderboard()
        LEADERBOARD.add_row(model_name="lr", cv_f1_mean=0.8, cv_f1_std=0.01,
                            latency_median_ms=0.12)
        LEADERBOARD.print_leaderboard_sorted()
    """
    _LB_ROWS.clear()
    return Leaderboard()