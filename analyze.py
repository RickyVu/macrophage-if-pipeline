"""
analyze.py - PCA / t-SNE / UMAP / RF analysis of per-cell morphology & marker
intensity features extracted during segmentation.

Loads cell_stats + images tables from the SQLite DB produced by segment.py
and runs a full characterisation pipeline for each macrophage type.

Output layout
-------------
analysis_output/
  01_pca_scatter.png
  02_pca_loadings.png
  03_pca_scree.png
  04_tsne.png
  05_umap.png                (skipped if umap-learn not installed)
  06_rf_importance.png       (impurity + permutation side-by-side)
  07_feature_boxplots.png    (top-12 Kruskal-Wallis features)
  08_correlation_heatmap.png
  09_hierarchical_clustering.png
  10_marker_positivity_heatmap.png
  kruskal_wallis.csv
  feature_stats_by_type.csv
  summary.txt

Usage
-----
    python analyze.py [--db image_index.sqlite] [--config config.json]
                      [--out-dir analysis_output/]
                      [--min-cells-per-type 30]
                      [--no-umap]
"""

import argparse
import json
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from skimage.filters import threshold_otsu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_MORPH_FEATURES = [
    "cell_area", "perimeter", "eccentricity",
    "major_axis", "minor_axis", "solidity",
    "extent", "circularity", "centroid_dist",
]
RAW_INTENSITY_FEATURES = [
    f"ch{i}_{s}" for i in range(6) for s in ("mean", "std")
]
ALL_RAW_FEATURES = RAW_MORPH_FEATURES + RAW_INTENSITY_FEATURES


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(db_path: Path) -> pd.DataFrame:
    """Load cell_stats joined with macrophage_type and case_name."""
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            """
            SELECT cs.*, i.macrophage_type, i.case_name
            FROM cell_stats cs
            JOIN images i ON i.id = cs.image_id
            WHERE cs.cell_area IS NOT NULL
            """,
            conn,
        )
    finally:
        conn.close()
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the dataframe (in-place, returns same df)."""
    df = df.copy()
    df["aspect_ratio"]           = df["major_axis"] / (df["minor_axis"] + 1e-6)
    df["membrane_nuc_ratio"]     = df["ch5_mean"] / (df["ch0_mean"] + 1e-6)
    df["marker_sum"]             = (df["ch1_mean"] + df["ch2_mean"] +
                                    df["ch3_mean"] + df["ch4_mean"])
    df["marker_max"]             = df[["ch1_mean", "ch2_mean",
                                       "ch3_mean", "ch4_mean"]].max(axis=1)
    df["nuc_intensity_spread"]   = df["ch0_std"] / (df["ch0_mean"] + 1e-6)
    for i in range(6):
        df[f"ch{i}_cv"]          = df[f"ch{i}_std"] / (df[f"ch{i}_mean"] + 1e-6)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    derived = [
        "aspect_ratio", "membrane_nuc_ratio", "marker_sum", "marker_max",
        "nuc_intensity_spread",
    ] + [f"ch{i}_cv" for i in range(6)]
    cols = ALL_RAW_FEATURES + derived
    return [c for c in cols if c in df.columns]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame, feature_cols: list, min_cells_per_type: int,
                 outlier_frac: float = 0.01):
    """
    Filter, impute, winsorise, remove outliers, scale.
    Returns (X_scaled, y_labels, le, df_filtered, feature_cols_kept).
    """
    # Drop types with too few cells
    type_counts = df["macrophage_type"].value_counts()
    keep_types  = type_counts[type_counts >= min_cells_per_type].index
    df = df[df["macrophage_type"].isin(keep_types)].copy()

    if len(df) == 0:
        raise SystemExit(
            f"No cells remain after filtering for >= {min_cells_per_type} cells/type."
        )

    X = df[feature_cols].values.astype(float)

    # Drop columns with >30% NaN
    nan_frac  = np.isnan(X).mean(axis=0)
    keep_cols = nan_frac <= 0.30
    X         = X[:, keep_cols]
    kept_feat = [c for c, k in zip(feature_cols, keep_cols) if k]

    # Impute remaining NaN with column median
    imputer = SimpleImputer(strategy="median")
    X       = imputer.fit_transform(X)

    # Winsorise at 1st/99th percentile
    for j in range(X.shape[1]):
        lo, hi  = np.percentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], lo, hi)

    # Multivariate outlier removal via Isolation Forest
    if outlier_frac > 0:
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=outlier_frac, random_state=42, n_jobs=-1)
        inlier_mask = iso.fit_predict(X) == 1
        n_removed   = int((~inlier_mask).sum())
        print(f"  Outlier removal ({outlier_frac*100:.1f}%): {n_removed} cells removed")
        X  = X[inlier_mask]
        df = df[inlier_mask].copy()
        for t, cnt in df["macrophage_type"].value_counts().items():
            if cnt < min_cells_per_type:
                print(f"  [WARN] {t} dropped to {cnt} cells after outlier removal")

    # RobustScaler
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    le       = LabelEncoder()
    y_labels = le.fit_transform(df["macrophage_type"])

    return X_scaled, y_labels, le, df, kept_feat


# ---------------------------------------------------------------------------
# Balanced subsampling (visualizations only)
# ---------------------------------------------------------------------------

def balanced_subsample(X: np.ndarray, y: np.ndarray,
                       cap_per_class: int, seed: int = 42):
    """
    Subsample to at most cap_per_class cells per type.
    Returns X_bal, y_bal. Used only for plots - stats always use full data.
    """
    rng = np.random.default_rng(seed)
    idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        take    = min(len(cls_idx), cap_per_class)
        idx.append(rng.choice(cls_idx, take, replace=False))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

def type_palette(types):
    cmap = plt.get_cmap("tab10")
    return {t: cmap(i % 10) for i, t in enumerate(types)}


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(X_scaled, y_labels, X_viz, y_viz, le, feature_cols, out_dir: Path):
    """
    X_scaled / y_labels : full dataset (used for PCA fit, scree, loadings).
    X_viz    / y_viz    : balanced subsample (used for scatter only).
    """
    pca   = PCA()
    X_pca = pca.fit_transform(X_scaled)
    # Project balanced subsample into the same PC space
    X_pca_viz = pca.transform(X_viz)
    types   = le.classes_
    palette = type_palette(types)
    n_viz   = len(X_viz)

    # --- Scree ---
    fig, ax = plt.subplots(figsize=(7, 4))
    n_show  = min(20, len(pca.explained_variance_ratio_))
    ax.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show] * 100)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA scree plot")
    fig.tight_layout()
    fig.savefig(out_dir / "03_pca_scree.png", dpi=150)
    plt.close(fig)

    # --- PC1 x PC2 scatter (balanced subsample) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for t in types:
        mask = y_viz == le.transform([t])[0]
        ax.scatter(X_pca_viz[mask, 0], X_pca_viz[mask, 1],
                   c=[palette[t]], label=t, alpha=0.40, s=10, linewidths=0)
        # 1σ ellipse (fit on full PC projection for this type)
        mask_full = y_labels == le.transform([t])[0]
        pts = X_pca[mask_full, :2]
        if len(pts) > 5:
            cov    = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order  = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h   = 2 * np.sqrt(vals)
            ell    = Ellipse(xy=pts.mean(0), width=w, height=h,
                             angle=angle, edgecolor=palette[t],
                             facecolor="none", lw=1.5)
            ax.add_patch(ell)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"PCA - PC1 x PC2  (balanced subsample, n={n_viz:,})")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "01_pca_scatter.png", dpi=150)
    plt.close(fig)

    # --- Loadings arrow plot ---
    loadings = pca.components_[:2].T   # (n_features, 2)
    n_feat   = len(feature_cols)
    fig, ax  = plt.subplots(figsize=(8, 8))
    for i, feat in enumerate(feature_cols):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                 head_width=0.015, head_length=0.015, fc="steelblue", ec="steelblue", alpha=0.7)
        ax.text(loadings[i, 0] * 1.12, loadings[i, 1] * 1.12, feat,
                ha="center", va="center", fontsize=7)
    circle = plt.Circle((0, 0), 1, color="grey", fill=False, linestyle="--", lw=0.8)
    ax.add_patch(circle)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axhline(0, c="k", lw=0.5)
    ax.axvline(0, c="k", lw=0.5)
    ax.set_xlabel("PC1 loading")
    ax.set_ylabel("PC2 loading")
    ax.set_title("PCA loadings")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "02_pca_loadings.png", dpi=150)
    plt.close(fig)

    # Print top-3 loadings
    for pc_idx, pc_name in enumerate(["PC1", "PC2"]):
        order = np.argsort(np.abs(pca.components_[pc_idx]))[::-1]
        top3  = [(feature_cols[j], pca.components_[pc_idx][j]) for j in order[:3]]
        print(f"  {pc_name} top loadings: "
              + "  ".join(f"{n}={v:+.3f}" for n, v in top3))

    return pca, X_pca


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def run_tsne(X_viz, y_viz, le, out_dir: Path):
    """X_viz / y_viz are already a balanced subsample."""
    N = X_viz.shape[0]
    perplexity = min(30, max(5, N // 5))
    print(f"  t-SNE: N={N}, perplexity={perplexity}")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000,
                random_state=42, init="pca")
    X_tsne = tsne.fit_transform(X_viz)

    types   = le.classes_
    palette = type_palette(types)
    fig, ax = plt.subplots(figsize=(8, 6))
    for t in types:
        mask = y_viz == le.transform([t])[0]
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[palette[t]], label=t, alpha=0.40, s=10, linewidths=0)
    ax.set_title(f"t-SNE  (balanced subsample, n={N:,})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "04_tsne.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# UMAP (optional)
# ---------------------------------------------------------------------------

def run_umap(X_viz, y_viz, le, out_dir: Path):
    """X_viz / y_viz are already a balanced subsample."""
    try:
        import umap
    except ImportError:
        print("  UMAP skipped (umap-learn not installed).")
        return

    N       = X_viz.shape[0]
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap  = reducer.fit_transform(X_viz)

    types   = le.classes_
    palette = type_palette(types)
    fig, ax = plt.subplots(figsize=(8, 6))
    for t in types:
        mask = y_viz == le.transform([t])[0]
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                   c=[palette[t]], label=t, alpha=0.40, s=10, linewidths=0)
    ax.set_title(f"UMAP  (balanced subsample, n={N:,})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "05_umap.png", dpi=150)
    plt.close(fig)
    print("  UMAP done.")


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def run_random_forest(X_scaled, y_labels, le, feature_cols, out_dir: Path):
    from sklearn.metrics import balanced_accuracy_score, classification_report

    rf  = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 n_jobs=-1, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_acc = cross_val_score(rf, X_scaled, y_labels, cv=skf,
                             scoring="accuracy", n_jobs=-1)
    cv_bal = cross_val_score(rf, X_scaled, y_labels, cv=skf,
                             scoring="balanced_accuracy", n_jobs=-1)
    print(f"  RF 5-fold accuracy         : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"  RF 5-fold balanced accuracy: {cv_bal.mean():.3f} ± {cv_bal.std():.3f}")

    # Fit on full data for importance + report
    rf.fit(X_scaled, y_labels)
    report = classification_report(y_labels, rf.predict(X_scaled),
                                   target_names=le.classes_)
    print(report)
    (out_dir / "rf_classification_report.txt").write_text(report)

    imp_mdi    = rf.feature_importances_
    order      = np.argsort(imp_mdi)[::-1][:20]

    # Permutation importance
    perm_res   = permutation_importance(rf, X_scaled, y_labels,
                                        n_repeats=10, random_state=42, n_jobs=-1)
    imp_perm   = perm_res.importances_mean
    order_perm = np.argsort(imp_perm)[::-1][:20]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh([feature_cols[i] for i in order[::-1]],
                 imp_mdi[order[::-1]], color="steelblue")
    axes[0].set_title("RF - mean decrease impurity (top 20)")
    axes[0].set_xlabel("Importance")

    axes[1].barh([feature_cols[i] for i in order_perm[::-1]],
                 imp_perm[order_perm[::-1]], color="darkorange")
    axes[1].set_title("RF - permutation importance (top 20)")
    axes[1].set_xlabel("Mean accuracy decrease")

    fig.tight_layout()
    fig.savefig(out_dir / "06_rf_importance.png", dpi=150)
    plt.close(fig)

    return rf, cv_acc, cv_bal, imp_mdi, imp_perm, feature_cols


# ---------------------------------------------------------------------------
# Kruskal-Wallis + FDR
# ---------------------------------------------------------------------------

def run_kruskal_wallis(df_filtered: pd.DataFrame, feature_cols: list,
                       out_dir: Path):
    """Run Kruskal-Wallis one-way ANOVA per feature; BH-correct p-values."""
    types  = df_filtered["macrophage_type"].unique()
    groups = {t: df_filtered.loc[df_filtered["macrophage_type"] == t,
                                 feature_cols].values
              for t in types}

    rows = []
    for feat in feature_cols:
        feat_idx = feature_cols.index(feat)
        samples  = [g[:, feat_idx] for g in groups.values()]
        samples  = [s[~np.isnan(s)] for s in samples]
        if any(len(s) < 3 for s in samples):
            rows.append({"feature": feat, "H": np.nan, "p": 1.0})
            continue
        try:
            h_stat, p_val = kruskal(*samples)
        except Exception:
            h_stat, p_val = np.nan, 1.0
        rows.append({"feature": feat, "H": h_stat, "p": p_val})

    kw_df = pd.DataFrame(rows)
    # BH correction
    p_vals = kw_df["p"].values
    n      = len(p_vals)
    order  = np.argsort(p_vals)
    rank   = np.empty(n)
    rank[order] = np.arange(1, n + 1)
    q_vals = np.minimum(1, p_vals * n / rank)
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        q_vals[order[i]] = min(q_vals[order[i]], q_vals[order[i + 1]])
    kw_df["q_bh"] = q_vals
    kw_df = kw_df.sort_values("q_bh")
    kw_df.to_csv(out_dir / "kruskal_wallis.csv", index=False)

    sig = (kw_df["q_bh"] < 0.05).sum()
    print(f"  Kruskal-Wallis: {sig}/{len(kw_df)} features significant (q<0.05)")
    return kw_df


# ---------------------------------------------------------------------------
# Box plots (top-12 KW features)
# ---------------------------------------------------------------------------

def plot_feature_boxplots(df_filtered: pd.DataFrame, feature_cols: list,
                          kw_df: pd.DataFrame, out_dir: Path):
    top12 = kw_df.head(12)["feature"].tolist()
    top12 = [f for f in top12 if f in feature_cols]
    if not top12:
        return

    n = len(top12)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flat if n > 1 else [axes]

    types   = sorted(df_filtered["macrophage_type"].unique())
    palette = type_palette(types)

    for ax, feat in zip(axes_flat, top12):
        data = [df_filtered.loc[df_filtered["macrophage_type"] == t, feat].dropna().values
                for t in types]
        bp = ax.boxplot(data, patch_artist=True, labels=types,
                        medianprops={"color": "black"})
        for patch, t in zip(bp["boxes"], types):
            patch.set_facecolor((*palette[t][:3], 0.6))

        ax.set_title(feat, fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    # Hide unused axes
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)

    fig.suptitle("Top-12 Kruskal-Wallis features by macrophage type", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "07_feature_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df_filtered: pd.DataFrame, feature_cols: list,
                              out_dir: Path):
    X = df_filtered[feature_cols].dropna(axis=0)
    if len(X) < 10:
        return
    corr = X.corr(method="pearson")

    # Hierarchical linkage for ordering
    Z    = linkage(corr.fillna(0).values, method="ward")
    dend = dendrogram(Z, no_plot=True)
    order = dend["leaves"]
    corr  = corr.iloc[order, :]
    corr  = corr.iloc[:, order]

    n = len(feature_cols)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), max(8, n * 0.4)))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature correlation (Pearson, clustered)")
    fig.tight_layout()
    fig.savefig(out_dir / "08_correlation_heatmap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hierarchical clustering
# ---------------------------------------------------------------------------

def plot_hierarchical_clustering(X_viz, y_viz, le, out_dir: Path):
    """X_viz / y_viz are a balanced subsample - used directly for the dendrogram."""
    from sklearn.metrics import adjusted_rand_score

    # Further cap for dendrogram readability (already balanced, just limit total N)
    max_n = 500
    if len(X_viz) > max_n:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(X_viz), max_n, replace=False)
        Xs   = X_viz[idx]
        ys   = y_viz[idx]
    else:
        Xs, ys = X_viz, y_viz

    Z     = linkage(Xs, method="ward")
    types = le.classes_
    palette = type_palette(types)
    label_colors = [palette[le.classes_[y]] for y in ys]

    fig, ax = plt.subplots(figsize=(14, 5))
    dend    = dendrogram(Z, ax=ax, no_labels=True,
                         link_color_func=lambda k: "grey")

    # Colour the leaf tick marks by macrophage type
    xcoords = dend["leaves"]
    for xi, leaf_idx in enumerate(dend["leaves"]):
        ax.get_xticklabels()  # silence warning
        col = label_colors[leaf_idx]
        ax.axvline(x=(xi + 0.5) * 10, ymin=0, ymax=0.04,
                   color=col, lw=2, clip_on=False)

    handles = [mpatches.Patch(color=palette[t], label=t) for t in types]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.set_title("Ward hierarchical clustering (leaf colours = macrophage type)")
    fig.tight_layout()
    fig.savefig(out_dir / "09_hierarchical_clustering.png", dpi=150)
    plt.close(fig)

    # ARI: cut tree into as many clusters as there are types
    from scipy.cluster.hierarchy import fcluster
    n_types = len(types)
    cluster_labels = fcluster(Z, n_types, criterion="maxclust")
    ari = adjusted_rand_score(ys, cluster_labels)
    print(f"  Hierarchical clustering ARI = {ari:.3f}")
    return ari


# ---------------------------------------------------------------------------
# Marker positivity heatmap
# ---------------------------------------------------------------------------

def plot_marker_positivity(df_filtered: pd.DataFrame, out_dir: Path):
    """
    For ch1-ch4: fraction of cells per macrophage_type with ch_mean > Otsu threshold.
    """
    types   = sorted(df_filtered["macrophage_type"].unique())
    markers = [f"ch{i}_mean" for i in range(1, 5)]

    result = {}
    for col in markers:
        vals = df_filtered[col].dropna().values
        if len(vals) < 10:
            thresh = 0.0
        else:
            try:
                thresh = threshold_otsu(vals)
            except Exception:
                thresh = vals.mean()
        fracs = {}
        for t in types:
            sub = df_filtered.loc[df_filtered["macrophage_type"] == t, col].dropna()
            fracs[t] = (sub > thresh).mean() * 100 if len(sub) > 0 else np.nan
        result[col] = fracs

    heatmap = pd.DataFrame(result, index=types)
    heatmap.columns = [c.replace("_mean", "") for c in heatmap.columns]

    fig, ax = plt.subplots(figsize=(6, max(3, len(types) * 0.6)))
    im = ax.imshow(heatmap.values, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_yticks(range(len(types)))
    ax.set_xticklabels(heatmap.columns)
    ax.set_yticklabels(types)
    plt.colorbar(im, ax=ax, label="% positive cells")
    for i in range(len(types)):
        for j in range(len(heatmap.columns)):
            val = heatmap.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=9)
    ax.set_title("Marker positivity (% cells above Otsu threshold)")
    fig.tight_layout()
    fig.savefig(out_dir / "10_marker_positivity_heatmap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary stats CSV
# ---------------------------------------------------------------------------

def save_feature_stats(df_filtered: pd.DataFrame, feature_cols: list,
                       out_dir: Path):
    rows = []
    for t in sorted(df_filtered["macrophage_type"].unique()):
        sub = df_filtered.loc[df_filtered["macrophage_type"] == t, feature_cols]
        row = {"macrophage_type": t}
        for feat in feature_cols:
            row[f"{feat}_mean"] = sub[feat].mean()
            row[f"{feat}_std"]  = sub[feat].std()
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "feature_stats_by_type.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PCA/t-SNE/RF analysis of cell_stats from segment.py."
    )
    parser.add_argument("--db",      default=None, help="SQLite DB path (overrides config)")
    parser.add_argument("--config",  default=None, help="Path to config.json")
    parser.add_argument("--out-dir", default="analysis_output",
                        help="Output directory (default: analysis_output/)")
    parser.add_argument("--min-cells-per-type", type=int, default=30,
                        help="Skip macrophage types with fewer cells (default: 30)")
    parser.add_argument("--no-umap", action="store_true",
                        help="Skip UMAP even if umap-learn is installed")
    parser.add_argument("--outlier-frac", type=float, default=0.01,
                        help="Fraction of cells removed as outliers via IsolationForest "
                             "(default: 0.01 = 1%%; set 0 to disable)")
    parser.add_argument("--viz-cap", type=int, default=2000,
                        help="Max cells per type in visualisations (balanced subsample, "
                             "default: 2000)")
    args = parser.parse_args()

    # Resolve DB path
    if args.db:
        db_path = Path(args.db).resolve()
    elif args.config:
        cfg = json.loads(Path(args.config).read_text())
        db_path = Path(cfg["project"]["db_path"]).resolve()
    else:
        # Try default locations
        for candidate in ["image_index.sqlite", "image_index.db"]:
            if Path(candidate).exists():
                db_path = Path(candidate).resolve()
                break
        else:
            try:
                cfg = json.loads(Path("config.json").read_text())
                db_path = Path(cfg["project"]["db_path"]).resolve()
            except Exception:
                raise SystemExit(
                    "Cannot find database. Pass --db or --config."
                )

    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Database : {db_path}")
    print(f"Output   : {out_dir}")
    print()

    # --- Load ---
    print("Loading data …")
    df = load_data(db_path)
    if len(df) == 0:
        raise SystemExit(
            "No rows in cell_stats with non-null cell_area. "
            "Run segment.py first."
        )

    df = engineer_features(df)
    feature_cols = get_feature_cols(df)

    type_counts = df["macrophage_type"].value_counts()
    print(f"  Total cells   : {len(df):,}")
    print(f"  Types found   : {len(type_counts)}")
    for t, n in type_counts.items():
        print(f"    {t}: {n:,}")
    print()

    # --- Prepare ---
    print("Preparing features …")
    X_scaled, y_labels, le, df_filtered, kept_feat = prepare_data(
        df, feature_cols, args.min_cells_per_type, outlier_frac=args.outlier_frac
    )
    print(f"  Kept {len(kept_feat)}/{len(feature_cols)} features "
          f"({len(df_filtered):,} cells, {len(le.classes_)} types)")
    print()

    # Balanced subsample for visualisations (scatter/t-SNE/UMAP/dendrogram)
    X_viz, y_viz = balanced_subsample(X_scaled, y_labels, cap_per_class=args.viz_cap)
    print(f"  Viz subsample: {len(X_viz):,} cells (≤{args.viz_cap} per type)")
    print()

    # --- PCA ---
    print("PCA …")
    pca, X_pca = run_pca(X_scaled, y_labels, X_viz, y_viz, le, kept_feat, out_dir)
    print()

    # --- t-SNE ---
    print("t-SNE …")
    run_tsne(X_viz, y_viz, le, out_dir)
    print()

    # --- UMAP ---
    if not args.no_umap:
        print("UMAP …")
        run_umap(X_viz, y_viz, le, out_dir)
        print()

    # --- Random Forest (uses full data) ---
    print("Random Forest …")
    rf, cv_acc, cv_bal, imp_mdi, imp_perm, _ = run_random_forest(
        X_scaled, y_labels, le, kept_feat, out_dir
    )
    print()

    # --- Kruskal-Wallis (uses full data) ---
    print("Kruskal-Wallis …")
    kw_df = run_kruskal_wallis(df_filtered, kept_feat, out_dir)
    print()

    # --- Box plots (uses full data) ---
    print("Box plots …")
    plot_feature_boxplots(df_filtered, kept_feat, kw_df, out_dir)
    print()

    # --- Correlation heatmap (uses full data) ---
    print("Correlation heatmap …")
    plot_correlation_heatmap(df_filtered, kept_feat, out_dir)
    print()

    # --- Hierarchical clustering (balanced subsample) ---
    print("Hierarchical clustering …")
    ari = plot_hierarchical_clustering(X_viz, y_viz, le, out_dir)
    print()

    # --- Marker positivity (uses full data) ---
    print("Marker positivity …")
    plot_marker_positivity(df_filtered, out_dir)
    print()

    # --- Summary stats CSV ---
    save_feature_stats(df_filtered, kept_feat, out_dir)

    # --- summary.txt ---
    top_feats_mdi = sorted(range(len(kept_feat)),
                            key=lambda i: imp_mdi[i], reverse=True)[:10]
    top_feats_kw  = kw_df.head(10)["feature"].tolist()

    summary_lines = [
        f"analyze.py summary",
        f"DB                   : {db_path}",
        f"Total cells          : {len(df_filtered):,}",
        f"Types                : {', '.join(le.classes_)}",
        f"Features             : {len(kept_feat)}",
        f"Outlier frac removed : {args.outlier_frac*100:.1f}%",
        f"Viz cap per type     : {args.viz_cap:,}  (n_viz={len(X_viz):,})",
        "",
        f"RF 5-fold accuracy          : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}",
        f"RF 5-fold balanced accuracy : {cv_bal.mean():.3f} ± {cv_bal.std():.3f}",
        f"Clustering ARI              : {ari:.3f}",
        "",
        "Top-10 features (RF MDI impurity):",
    ] + [f"  {kept_feat[i]}" for i in top_feats_mdi] + [
        "",
        "Top-10 features (Kruskal-Wallis):",
    ] + [f"  {f}" for f in top_feats_kw]

    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    print("Done.")
    print(f"  Output: {out_dir}/")
    print(f"  RF accuracy          : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"  RF balanced accuracy : {cv_bal.mean():.3f} ± {cv_bal.std():.3f}")
    print(f"  ARI                  : {ari:.3f}")


if __name__ == "__main__":
    main()
