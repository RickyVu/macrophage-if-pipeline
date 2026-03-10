"""
pairwise.py - Pairwise macrophage-type analysis (features or CNN embeddings).

For each pair of macrophage types (A, B):
  - Subsample the larger class to min(N_A, N_B) so both sides are balanced.
  - Run: PCA scatter, t-SNE, UMAP (optional), binary RF (accuracy + balanced
    accuracy + ROC-AUC), ROC curve, top-discriminating features bar chart.
  - Save results to  <out-dir>/<typeA>_vs_<typeB>/
  - Append a row to  <out-dir>/master_comparison.csv

Why pairwise?
  Global multi-class analysis penalises rare types (563 cells vs 11k forces
  the balanced subsample to 563x4 = 2252 cells total, discarding 97% of the
  data for the large groups).  Pairwise uses min(N_A, N_B) per comparison, so:
    • 11k vs 10k   → 10k each  (full statistical power)
    • 11k vs 563   → 563 each  (fair to the small group)

Two modes
---------
features   : morphological + intensity features from the `cell_stats` DB table
             (same features as analyze.py)
embeddings : 512-dim ResNet-18 CNN embeddings from an embeddings.npz file
             (produced by embed.py)

Usage
-----
    # features mode
    python pairwise.py --mode features \\
           [--db image_index.sqlite] [--config config.json] \\
           [--out-dir pairwise_features/]

    # embeddings mode
    python pairwise.py --mode embeddings \\
           --embeddings embedding_output/embeddings.npz \\
           [--db image_index.sqlite] [--config config.json] \\
           [--out-dir pairwise_embeddings/]

    # shared flags (both modes)
           [--outlier-frac 0.01]
           [--no-umap]
           [--min-cells-per-type 30]
"""

import argparse
import json
import sqlite3
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse data-loading + feature helpers from analyze.py
from analyze import (
    load_data,
    engineer_features,
    get_feature_cols,
    type_palette,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data loading - features mode
# ---------------------------------------------------------------------------

def load_feature_data(db_path: Path) -> tuple:
    """
    Returns (df_meta, feature_cols) where df_meta has columns
    [image_id, macrophage_type, case_name, <feature_cols...>].
    """
    df           = load_data(db_path)
    df           = engineer_features(df)
    feature_cols = get_feature_cols(df)
    return df, feature_cols


# ---------------------------------------------------------------------------
# Data loading - embeddings mode
# ---------------------------------------------------------------------------

def load_embedding_data(npz_path: Path, db_path: Path) -> tuple:
    """
    Returns (embeddings np.ndarray float32 (N, 512),
             df_meta pd.DataFrame [image_id, macrophage_type, case_name],
             feature_names list[str])
    """
    data       = np.load(str(npz_path))
    image_ids  = data["image_ids"].astype(np.int64)
    embeddings = data["embeddings"].astype(np.float32)

    conn = sqlite3.connect(str(db_path))
    try:
        id_list = ",".join(str(x) for x in image_ids.tolist())
        df_db   = pd.read_sql_query(
            f"SELECT id AS image_id, macrophage_type, case_name "
            f"FROM images WHERE id IN ({id_list})",
            conn,
        )
    finally:
        conn.close()

    id_to_row = df_db.set_index("image_id").to_dict("index")
    rows = []
    for iid in image_ids:
        if iid in id_to_row:
            rows.append({"image_id": int(iid), **id_to_row[iid]})
        else:
            rows.append({"image_id": int(iid),
                         "macrophage_type": "unknown", "case_name": "unknown"})
    df_meta      = pd.DataFrame(rows).reset_index(drop=True)
    feature_names = [f"e{i}" for i in range(embeddings.shape[1])]
    return embeddings, df_meta, feature_names


# ---------------------------------------------------------------------------
# Per-pair data preparation
# ---------------------------------------------------------------------------

def subsample_pair(
    X_all: np.ndarray,
    df_meta: pd.DataFrame,
    name_a: str,
    name_b: str,
    seed: int = 42,
) -> tuple:
    """
    Extract rows for name_a and name_b; subsample the larger to min(N_a, N_b).

    Returns
    -------
    X_pair : (2*min_n, D) combined matrix (name_a first, then name_b)
    y_pair : (2*min_n,)  binary labels  0=name_a, 1=name_b
    N_a, N_b, min_n
    """
    mask_a   = (df_meta["macrophage_type"] == name_a).values
    mask_b   = (df_meta["macrophage_type"] == name_b).values
    idx_a    = np.where(mask_a)[0]
    idx_b    = np.where(mask_b)[0]
    N_a, N_b = len(idx_a), len(idx_b)
    min_n    = min(N_a, N_b)

    # Reproducible seed per pair (order-independent)
    pair_seed = seed ^ (hash(tuple(sorted([name_a, name_b]))) & 0xFFFFFFFF)
    rng       = np.random.default_rng(pair_seed)
    idx_a_sub = rng.choice(idx_a, min_n, replace=False)
    idx_b_sub = rng.choice(idx_b, min_n, replace=False)

    X_pair = np.vstack([X_all[idx_a_sub], X_all[idx_b_sub]])
    y_pair = np.array([0] * min_n + [1] * min_n, dtype=np.int64)
    return X_pair, y_pair, N_a, N_b, min_n


def prepare_pair(
    X_pair: np.ndarray,
    y_pair: np.ndarray,
    has_nan: bool,
    outlier_frac: float,
) -> np.ndarray:
    """
    Impute (if has_nan), winsorise, optional outlier removal, RobustScale.
    Returns X_scaled (same shape as X_pair, minus any removed outlier rows),
    and the (possibly shorter) y_pair.
    """
    X = X_pair.astype(float)
    y = y_pair.copy()

    if has_nan:
        imputer = SimpleImputer(strategy="median")
        X       = imputer.fit_transform(X)
        # Winsorise
        for j in range(X.shape[1]):
            lo, hi  = np.percentile(X[:, j], [1, 99])
            X[:, j] = np.clip(X[:, j], lo, hi)

    if outlier_frac > 0 and len(X) >= 20:
        iso         = IsolationForest(contamination=outlier_frac,
                                      random_state=42, n_jobs=-1)
        inlier      = iso.fit_predict(X) == 1
        n_removed   = int((~inlier).sum())
        if n_removed > 0:
            X = X[inlier]
            y = y[inlier]

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


# ---------------------------------------------------------------------------
# Core pairwise analysis (mode-agnostic)
# ---------------------------------------------------------------------------

def analyze_pair(
    X_scaled: np.ndarray,
    y_binary: np.ndarray,
    name_a: str,
    name_b: str,
    feature_names: list,
    out_dir: Path,
    no_umap: bool,
) -> dict:
    """
    Run full 2-class analysis and save plots.

    Parameters
    ----------
    X_scaled     : (N, D) float64, already scaled
    y_binary     : (N,)  0=name_a, 1=name_b
    feature_names: column labels (morphological names or "e0", "e1", ...)
    out_dir      : directory for this pair's outputs

    Returns metrics dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    col_a  = "steelblue"
    col_b  = "darkorange"
    N      = len(X_scaled)

    # ---- PCA ----
    pca       = PCA(random_state=42 if hasattr(PCA, "random_state") else None)
    X_pca     = pca.fit_transform(X_scaled)
    fig, ax   = plt.subplots(figsize=(7, 5))
    ax.scatter(X_pca[y_binary == 0, 0], X_pca[y_binary == 0, 1],
               c=col_a, label=name_a, alpha=0.35, s=8, linewidths=0)
    ax.scatter(X_pca[y_binary == 1, 0], X_pca[y_binary == 1, 1],
               c=col_b, label=name_b, alpha=0.35, s=8, linewidths=0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"PCA - {name_a} vs {name_b}  (n={N:,})")
    ax.legend(markerscale=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "01_pca.png", dpi=150)
    plt.close(fig)

    # ---- t-SNE ----
    perplexity = min(30, max(5, N // 10))
    n_comp     = min(50, X_scaled.shape[1])
    X_in       = PCA(n_components=n_comp).fit_transform(X_scaled)
    X_tsne     = TSNE(n_components=2, perplexity=perplexity, max_iter=1000,
                      random_state=42, init="pca").fit_transform(X_in)
    fig, ax    = plt.subplots(figsize=(7, 5))
    ax.scatter(X_tsne[y_binary == 0, 0], X_tsne[y_binary == 0, 1],
               c=col_a, label=name_a, alpha=0.35, s=8, linewidths=0)
    ax.scatter(X_tsne[y_binary == 1, 0], X_tsne[y_binary == 1, 1],
               c=col_b, label=name_b, alpha=0.35, s=8, linewidths=0)
    ax.set_title(f"t-SNE - {name_a} vs {name_b}  (n={N:,})")
    ax.legend(markerscale=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "02_tsne.png", dpi=150)
    plt.close(fig)

    # ---- UMAP ----
    if not no_umap:
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            X_umap  = reducer.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(X_umap[y_binary == 0, 0], X_umap[y_binary == 0, 1],
                       c=col_a, label=name_a, alpha=0.35, s=8, linewidths=0)
            ax.scatter(X_umap[y_binary == 1, 0], X_umap[y_binary == 1, 1],
                       c=col_b, label=name_b, alpha=0.35, s=8, linewidths=0)
            ax.set_title(f"UMAP - {name_a} vs {name_b}  (n={N:,})")
            ax.legend(markerscale=3, fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / "03_umap.png", dpi=150)
            plt.close(fig)
        except ImportError:
            pass

    # ---- Binary RF ----
    rf  = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 n_jobs=-1, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_acc  = cross_val_score(rf, X_scaled, y_binary, cv=skf,
                              scoring="accuracy", n_jobs=-1)
    cv_bal  = cross_val_score(rf, X_scaled, y_binary, cv=skf,
                              scoring="balanced_accuracy", n_jobs=-1)
    cv_auc  = cross_val_score(rf, X_scaled, y_binary, cv=skf,
                              scoring="roc_auc", n_jobs=-1)

    # Fit on full for importance + ROC curve
    rf.fit(X_scaled, y_binary)
    proba   = rf.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_binary, proba)
    auc_full    = roc_auc_score(y_binary, proba)

    report = classification_report(y_binary, rf.predict(X_scaled),
                                   target_names=[name_a, name_b])
    (out_dir / "rf_results.txt").write_text(
        f"{name_a} vs {name_b}\n"
        f"5-fold accuracy         : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}\n"
        f"5-fold balanced accuracy: {cv_bal.mean():.3f} ± {cv_bal.std():.3f}\n"
        f"5-fold ROC-AUC          : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}\n\n"
        + report
    )

    # ROC curve plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="firebrick",
            label=f"AUC={auc_full:.3f} (train)")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC - {name_a} vs {name_b}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "04_roc.png", dpi=150)
    plt.close(fig)

    # Top-20 features by MDI
    imp_mdi = rf.feature_importances_
    top20   = np.argsort(imp_mdi)[::-1][:20]
    top_names = [feature_names[i] if i < len(feature_names) else f"f{i}"
                 for i in top20]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_names[::-1], imp_mdi[top20[::-1]], color="steelblue")
    ax.set_title(f"RF MDI importance (top 20) - {name_a} vs {name_b}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_dir / "05_rf_importance.png", dpi=150)
    plt.close(fig)

    metrics = {
        "pair":              f"{name_a} vs {name_b}",
        "N_a":               int((y_binary == 0).sum()),
        "N_b":               int((y_binary == 1).sum()),
        "N_balanced":        N,
        "RF_accuracy":       round(cv_acc.mean(), 4),
        "RF_accuracy_std":   round(cv_acc.std(),  4),
        "RF_bal_accuracy":   round(cv_bal.mean(), 4),
        "RF_bal_acc_std":    round(cv_bal.std(),  4),
        "RF_AUC":            round(cv_auc.mean(), 4),
        "RF_AUC_std":        round(cv_auc.std(),  4),
        "top5_features":     ", ".join(top_names[:5]),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pairwise macrophage-type analysis (features or embeddings)."
    )
    parser.add_argument("--mode", choices=["features", "embeddings"],
                        required=True,
                        help="features: use cell_stats morphology; "
                             "embeddings: use CNN embeddings npz")
    parser.add_argument("--db",          default=None,
                        help="SQLite DB path (overrides config)")
    parser.add_argument("--config",      default=None,
                        help="Path to config.json")
    parser.add_argument("--embeddings",  default="embedding_output/embeddings.npz",
                        help="Path to embeddings.npz (embeddings mode only, "
                             "default: embedding_output/embeddings.npz)")
    parser.add_argument("--out-dir",     default=None,
                        help="Output root dir (default: pairwise_features/ "
                             "or pairwise_embeddings/)")
    parser.add_argument("--outlier-frac", type=float, default=0.01,
                        help="Fraction removed as outliers per pair "
                             "(default: 0.01; set 0 to disable)")
    parser.add_argument("--min-cells-per-type", type=int, default=30,
                        help="Skip types with fewer total cells (default: 30)")
    parser.add_argument("--no-umap",     action="store_true",
                        help="Skip UMAP")
    args = parser.parse_args()

    # --- Resolve DB path ---
    if args.db:
        db_path = Path(args.db).resolve()
    elif args.config:
        cfg     = json.loads(Path(args.config).read_text())
        db_path = Path(cfg["project"]["db_path"]).resolve()
    else:
        for candidate in ["image_index.sqlite", "image_index.db"]:
            if Path(candidate).exists():
                db_path = Path(candidate).resolve()
                break
        else:
            try:
                cfg     = json.loads(Path("config.json").read_text())
                db_path = Path(cfg["project"]["db_path"]).resolve()
            except Exception:
                raise SystemExit("Cannot find database. Pass --db or --config.")
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    out_dir = Path(args.out_dir) if args.out_dir else \
              Path(f"pairwise_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode     : {args.mode}")
    print(f"Database : {db_path}")
    print(f"Output   : {out_dir}")
    print()

    # --- Load data ---
    if args.mode == "features":
        print("Loading feature data …")
        df_meta, feature_cols = load_feature_data(db_path)
        # X_all is the raw feature matrix (may have NaN - handled per pair)
        X_all    = df_meta[feature_cols].values.astype(float)
        has_nan  = np.isnan(X_all).any()
        type_col = df_meta["macrophage_type"]
    else:
        npz_path = Path(args.embeddings)
        if not npz_path.exists():
            raise SystemExit(
                f"Embeddings file not found: {npz_path}\n"
                "Run embed.py first."
            )
        print(f"Loading embeddings from {npz_path} …")
        X_all, df_meta, feature_cols = load_embedding_data(npz_path, db_path)
        has_nan  = False
        type_col = df_meta["macrophage_type"]

    # Filter types with too few cells
    type_counts = type_col.value_counts()
    keep_types  = type_counts[type_counts >= args.min_cells_per_type].index.tolist()
    if len(keep_types) < 2:
        raise SystemExit(
            f"Need ≥ 2 types with ≥ {args.min_cells_per_type} cells for pairwise analysis."
        )

    print(f"Types to analyse ({len(keep_types)}):")
    for t in keep_types:
        print(f"  {t}: {type_counts[t]:,} cells")
    pairs = list(combinations(sorted(keep_types), 2))
    print(f"\nPairs to run: {len(pairs)}")
    print()

    all_metrics = []

    for name_a, name_b in pairs:
        pair_label = f"{name_a}_vs_{name_b}"
        print(f"─── {pair_label} ───")
        pair_dir = out_dir / pair_label

        # Subsample to balanced pair
        X_pair, y_pair, N_a, N_b, min_n = subsample_pair(
            X_all, df_meta, name_a, name_b
        )
        print(f"  N_a={N_a:,}  N_b={N_b:,}  → balanced N={min_n:,} each")

        # Prepare (impute / winsorise / outlier / scale)
        X_scaled, y_pair = prepare_pair(X_pair, y_pair, has_nan, args.outlier_frac)
        print(f"  After prep: {len(X_scaled):,} cells total")

        # Analyse
        metrics = analyze_pair(
            X_scaled, y_pair, name_a, name_b, feature_cols,
            pair_dir, args.no_umap
        )
        metrics["name_a"] = name_a
        metrics["name_b"] = name_b
        all_metrics.append(metrics)

        print(f"  RF acc={metrics['RF_accuracy']:.3f}  "
              f"bal={metrics['RF_bal_accuracy']:.3f}  "
              f"AUC={metrics['RF_AUC']:.3f}")
        print(f"  Top features: {metrics['top5_features']}")
        print()

    # --- Master comparison CSV ---
    master = pd.DataFrame(all_metrics)
    master = master.sort_values("RF_AUC", ascending=False)
    csv_path = out_dir / "master_comparison.csv"
    master.to_csv(str(csv_path), index=False)
    print(f"Master comparison saved → {csv_path}")

    # --- Master AUC bar chart ---
    fig, ax = plt.subplots(figsize=(max(6, len(pairs) * 0.8), 5))
    pairs_labels = [f"{r.name_a}\nvs\n{r.name_b}" for _, r in master.iterrows()]
    ax.bar(pairs_labels, master["RF_AUC"].values, color="steelblue")
    ax.axhline(0.5, color="k", linestyle="--", lw=0.8, label="chance")
    ax.set_ylim(0, 1)
    ax.set_ylabel("5-fold ROC-AUC")
    ax.set_title(f"Pairwise separability ({args.mode} mode)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "00_master_auc.png", dpi=150)
    plt.close(fig)
    print(f"Master AUC chart saved → {out_dir / '00_master_auc.png'}")

    print("\nDone.")
    print(master[["pair", "N_balanced", "RF_accuracy",
                  "RF_bal_accuracy", "RF_AUC"]].to_string(index=False))


if __name__ == "__main__":
    main()
