"""
embed.py - CNN embedding extraction + analysis for cell images.

Loads 6-channel cell images from HDF5 files produced by segment.py, extracts
512-dim feature vectors using a 6-channel adapted ResNet-18 (ImageNet pretrained
weights, averaged from RGB → 6 channels, no fine-tuning), then runs the same
analysis pipeline as analyze.py on those embeddings.

Why CNN embeddings?
  Scalar morphological summaries (cell_area, ch3_mean, etc.) are 1-D projections
  that discard spatial and textural information. A CNN looking at the raw image
  can capture: nuclear texture, subcellular marker localisation, co-expression
  patterns. If CNN-based separation is higher than morphological, the biology is
  real but the hand-crafted features miss the key signal.

Output layout
-------------
embedding_output/
  embeddings.npz                  (image_ids: int64, embeddings: float32 Nx512)
  01_pca_scatter.png
  02_pca_scree.png
  03_tsne.png
  04_umap.png                     (skipped if umap-learn not installed)
  05_rf_importance.png            (top 20 embedding dims by MDI)
  06_kw_pc_scores.png             (KW H-statistic on top-20 PC scores)
  09_hierarchical_clustering.png
  rf_classification_report.txt
  summary.txt

Usage
-----
    python embed.py [--db image_index.sqlite] [--config config.json]
                    [--h5-dir stacked_cells/]
                    [--out-dir embedding_output/]
                    [--batch-size 256]
                    [--img-size 64]
                    [--no-cuda]
                    [--no-umap]
                    [--outlier-frac 0.01]
                    [--viz-cap 2000]
                    [--min-cells-per-type 30]
                    [--skip-extract]
"""

import argparse
import json
import sqlite3
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

# Reuse visualisation helpers from analyze.py
from analyze import (
    balanced_subsample,
    type_palette,
    plot_hierarchical_clustering,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Model - ResNet-18 adapted for 6-channel input
# ---------------------------------------------------------------------------

def build_embedding_model(use_cuda: bool) -> tuple:
    """
    Return (model, device) where model maps (B, 6, H, W) → (B, 512).

    Weights: ImageNet pretrained ResNet-18.  The first conv layer is replaced
    with a 6-channel version; its weights are initialised by averaging the
    3-channel pretrained weights (preserves scale, reasonable starting point).
    No fine-tuning - frozen inference.
    """
    try:
        import torchvision.models as tv_models
    except ImportError:
        raise SystemExit(
            "torchvision not installed. Run: pip install torchvision"
        )

    model   = tv_models.resnet18(weights="IMAGENET1K_V1")
    old_w   = model.conv1.weight.data          # (64, 3, 7, 7)
    new_w   = old_w.mean(dim=1, keepdim=True).expand(-1, 6, -1, -1).clone()
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = new_w
    model.fc    = nn.Identity()                # output: 512-dim
    model.eval()

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    return model, device


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    h5_dir: Path,
    db_path: Path,
    model,
    device,
    batch_size: int,
    img_size: int,
) -> tuple:
    """
    Walk all cells.h5 files under h5_dir/*/*/cells.h5.
    Extract 512-dim CNN embeddings for every cell image.

    Returns
    -------
    image_ids  : np.ndarray int64  (N,)
    embeddings : np.ndarray float32 (N, 512)
    df_meta    : pd.DataFrame  columns: image_id, macrophage_type, case_name
    """
    h5_files = sorted(h5_dir.glob("*/*/cells.h5"))
    if not h5_files:
        raise SystemExit(f"No cells.h5 found under {h5_dir}")

    print(f"  Found {len(h5_files)} cells.h5 file(s)")

    all_ids  = []
    all_embs = []

    for h5_path in h5_files:
        case_name = h5_path.parent.parent.name
        mac_type  = h5_path.parent.name
        try:
            with h5py.File(str(h5_path), "r") as hf:
                imgs     = hf["images"][:]     # (N, 6, H, W) float32
                img_ids  = hf["image_ids"][:]  # (N,) int64
        except Exception as e:
            print(f"  [WARN] Failed to open {h5_path}: {e}")
            continue

        N = len(imgs)
        if N == 0:
            continue

        print(f"  {case_name}/{mac_type}: {N:,} cells  "
              f"shape={imgs.shape[1:]}  → resizing to (6,{img_size},{img_size})")

        embs = _run_model_on_images(imgs, model, device, batch_size, img_size)
        all_ids.append(img_ids)
        all_embs.append(embs)

    if not all_ids:
        raise SystemExit("No embeddings extracted - check h5_dir.")

    image_ids  = np.concatenate(all_ids,  axis=0).astype(np.int64)
    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)

    # Join with DB to get macrophage_type and case_name
    conn = sqlite3.connect(str(db_path))
    try:
        id_list  = ",".join(str(x) for x in image_ids.tolist())
        df_meta  = pd.read_sql_query(
            f"SELECT id AS image_id, macrophage_type, case_name "
            f"FROM images WHERE id IN ({id_list})",
            conn,
        )
    finally:
        conn.close()

    # Align df_meta to image_ids order
    id_to_row = {row.image_id: row for _, row in df_meta.iterrows()}
    rows = []
    for iid in image_ids:
        if iid in id_to_row:
            rows.append(id_to_row[iid])
        else:
            rows.append({"image_id": iid, "macrophage_type": "unknown",
                         "case_name": "unknown"})
    df_meta_aligned = pd.DataFrame(rows).reset_index(drop=True)

    return image_ids, embeddings, df_meta_aligned


@torch.no_grad()
def _run_model_on_images(
    imgs: np.ndarray,
    model,
    device,
    batch_size: int,
    img_size: int,
) -> np.ndarray:
    """Run model over imgs in batches. imgs shape: (N, 6, H, W)."""
    N    = len(imgs)
    embs = []

    for start in range(0, N, batch_size):
        batch = imgs[start: start + batch_size]           # (B, 6, H, W)
        t     = torch.from_numpy(batch).to(device)        # float32

        # Resize to img_size x img_size
        if t.shape[2] != img_size or t.shape[3] != img_size:
            t = F.interpolate(t, size=(img_size, img_size),
                              mode="bilinear", align_corners=False)

        out = model(t)                                     # (B, 512)
        embs.append(out.cpu().numpy())

    return np.concatenate(embs, axis=0)


# ---------------------------------------------------------------------------
# Data preparation (embeddings)
# ---------------------------------------------------------------------------

def prepare_embeddings(
    embeddings: np.ndarray,
    df_meta: pd.DataFrame,
    min_cells_per_type: int,
    outlier_frac: float,
) -> tuple:
    """
    Filter by min_cells_per_type, remove outliers, scale.
    Returns (X_scaled, y_labels, le, df_filtered).
    """
    # Filter types
    type_counts = df_meta["macrophage_type"].value_counts()
    keep_types  = type_counts[type_counts >= min_cells_per_type].index
    mask        = df_meta["macrophage_type"].isin(keep_types).values
    X           = embeddings[mask].astype(np.float32)
    df          = df_meta[mask].copy().reset_index(drop=True)

    if len(X) == 0:
        raise SystemExit(
            f"No cells remain after filtering for >= {min_cells_per_type} cells/type."
        )

    print(f"  After type filter: {len(X):,} cells, {len(keep_types)} types")

    # Outlier removal
    if outlier_frac > 0:
        iso         = IsolationForest(contamination=outlier_frac,
                                      random_state=42, n_jobs=-1)
        inlier_mask = iso.fit_predict(X) == 1
        n_removed   = int((~inlier_mask).sum())
        print(f"  Outlier removal ({outlier_frac*100:.1f}%): {n_removed} cells removed")
        X  = X[inlier_mask]
        df = df[inlier_mask].copy().reset_index(drop=True)

    # Scale
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    le       = LabelEncoder()
    y_labels = le.fit_transform(df["macrophage_type"])

    return X_scaled, y_labels, le, df


# ---------------------------------------------------------------------------
# PCA for embeddings
# ---------------------------------------------------------------------------

def run_pca_emb(X_scaled, y_labels, X_viz, y_viz, le, out_dir: Path):
    pca       = PCA()
    X_pca     = pca.fit_transform(X_scaled)
    X_pca_viz = pca.transform(X_viz)
    types     = le.classes_
    palette   = type_palette(types)
    n_viz     = len(X_viz)

    # Scree
    fig, ax  = plt.subplots(figsize=(7, 4))
    n_show   = min(30, len(pca.explained_variance_ratio_))
    ax.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show] * 100)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("CNN embedding - PCA scree")
    fig.tight_layout()
    fig.savefig(out_dir / "02_pca_scree.png", dpi=150)
    plt.close(fig)

    # Scatter (balanced subsample)
    fig, ax = plt.subplots(figsize=(8, 6))
    for t in types:
        m = y_viz == le.transform([t])[0]
        ax.scatter(X_pca_viz[m, 0], X_pca_viz[m, 1],
                   c=[palette[t]], label=t, alpha=0.40, s=10, linewidths=0)
        # Ellipse from full data
        m_full = y_labels == le.transform([t])[0]
        pts    = X_pca[m_full, :2]
        if len(pts) > 5:
            cov       = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order     = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle     = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h      = 2 * np.sqrt(vals)
            ell = Ellipse(xy=pts.mean(0), width=w, height=h,
                          angle=angle, edgecolor=palette[t],
                          facecolor="none", lw=1.5)
            ax.add_patch(ell)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"CNN embeddings - PCA  (balanced subsample, n={n_viz:,})")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "01_pca_scatter.png", dpi=150)
    plt.close(fig)

    return pca, X_pca


# ---------------------------------------------------------------------------
# t-SNE for embeddings
# ---------------------------------------------------------------------------

def run_tsne_emb(X_viz, y_viz, le, out_dir: Path):
    N          = X_viz.shape[0]
    perplexity = min(30, max(5, N // 5))
    print(f"  t-SNE: N={N}, perplexity={perplexity}")

    # Use top-50 PCA components as input (faster, less noisy for high-dim)
    n_comp = min(50, X_viz.shape[1])
    X_in   = PCA(n_components=n_comp, random_state=42).fit_transform(X_viz)

    tsne   = TSNE(n_components=2, perplexity=perplexity, n_iter=1000,
                  random_state=42, init="pca")
    X_tsne = tsne.fit_transform(X_in)

    types   = le.classes_
    palette = type_palette(types)
    fig, ax = plt.subplots(figsize=(8, 6))
    for t in types:
        mask = y_viz == le.transform([t])[0]
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[palette[t]], label=t, alpha=0.40, s=10, linewidths=0)
    ax.set_title(f"CNN embeddings - t-SNE  (balanced subsample, n={N:,})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "03_tsne.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# UMAP for embeddings
# ---------------------------------------------------------------------------

def run_umap_emb(X_viz, y_viz, le, out_dir: Path):
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
    ax.set_title(f"CNN embeddings - UMAP  (balanced subsample, n={N:,})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "04_umap.png", dpi=150)
    plt.close(fig)
    print("  UMAP done.")


# ---------------------------------------------------------------------------
# Random Forest on embeddings
# ---------------------------------------------------------------------------

def run_rf_emb(X_scaled, y_labels, le, out_dir: Path):
    rf  = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 n_jobs=-1, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_acc = cross_val_score(rf, X_scaled, y_labels, cv=skf,
                             scoring="accuracy", n_jobs=-1)
    cv_bal = cross_val_score(rf, X_scaled, y_labels, cv=skf,
                             scoring="balanced_accuracy", n_jobs=-1)
    print(f"  RF 5-fold accuracy         : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"  RF 5-fold balanced accuracy: {cv_bal.mean():.3f} ± {cv_bal.std():.3f}")

    rf.fit(X_scaled, y_labels)
    report = classification_report(y_labels, rf.predict(X_scaled),
                                   target_names=le.classes_)
    print(report)
    (out_dir / "rf_classification_report.txt").write_text(report)

    # MDI importance: top-20 embedding dimensions
    imp_mdi  = rf.feature_importances_           # length 512
    order    = np.argsort(imp_mdi)[::-1][:20]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([f"e{i}" for i in order[::-1]], imp_mdi[order[::-1]], color="steelblue")
    ax.set_title("CNN embedding - RF MDI importance (top 20 dims)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_dir / "05_rf_importance.png", dpi=150)
    plt.close(fig)

    return rf, cv_acc, cv_bal


# ---------------------------------------------------------------------------
# KW on top-20 PC scores
# ---------------------------------------------------------------------------

def run_kw_pc_scores(X_scaled, y_labels, le, pca, out_dir: Path):
    """
    Kruskal-Wallis on the scores of the top-20 PCA components.
    More interpretable than running KW on all 512 embedding dims.
    """
    n_pc   = min(20, X_scaled.shape[1])
    X_pca  = pca.transform(X_scaled)[:, :n_pc]
    types  = le.classes_

    h_stats = []
    p_vals  = []
    labels  = [f"PC{i+1}" for i in range(n_pc)]

    for j in range(n_pc):
        samples = [X_pca[y_labels == le.transform([t])[0], j] for t in types]
        try:
            h, p = kruskal(*samples)
        except Exception:
            h, p = 0.0, 1.0
        h_stats.append(h)
        p_vals.append(p)

    order = np.argsort(h_stats)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ["steelblue" if p_vals[i] < 0.05 else "lightgrey" for i in order]
    ax.barh([labels[i] for i in order[::-1]], [h_stats[i] for i in order[::-1]],
            color=colors[::-1])
    ax.set_title("KW H-statistic on top-20 PC scores\n(blue = p<0.05)")
    ax.set_xlabel("Kruskal-Wallis H")
    fig.tight_layout()
    fig.savefig(out_dir / "06_kw_pc_scores.png", dpi=150)
    plt.close(fig)

    sig = sum(p < 0.05 for p in p_vals)
    print(f"  KW on PC scores: {sig}/{n_pc} significant (p<0.05)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CNN embedding extraction + analysis for cell images."
    )
    parser.add_argument("--db",      default=None, help="SQLite DB path (overrides config)")
    parser.add_argument("--config",  default=None, help="Path to config.json")
    parser.add_argument("--h5-dir",  default="stacked_cells",
                        help="Root directory of cells.h5 files (default: stacked_cells/)")
    parser.add_argument("--out-dir", default="embedding_output",
                        help="Output directory (default: embedding_output/)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Images per inference batch (default: 256)")
    parser.add_argument("--img-size",   type=int, default=64,
                        help="Resize cell images to this square size (default: 64)")
    parser.add_argument("--no-cuda",    action="store_true")
    parser.add_argument("--no-umap",    action="store_true")
    parser.add_argument("--outlier-frac", type=float, default=0.01,
                        help="Fraction removed as outliers via IsolationForest "
                             "(default: 0.01; set 0 to disable)")
    parser.add_argument("--viz-cap",    type=int, default=2000,
                        help="Max cells per type in visualisations (default: 2000)")
    parser.add_argument("--min-cells-per-type", type=int, default=30,
                        help="Skip types with fewer cells (default: 30)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Load existing embeddings.npz instead of re-running model")
    args = parser.parse_args()

    # --- Resolve paths ---
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

    h5_dir  = Path(args.h5_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "embeddings.npz"

    use_cuda = not args.no_cuda

    print(f"Database : {db_path}")
    print(f"H5 dir   : {h5_dir}")
    print(f"Output   : {out_dir}")
    print()

    # --- Extract or load embeddings ---
    if args.skip_extract and npz_path.exists():
        print(f"Loading existing embeddings from {npz_path} …")
        data       = np.load(str(npz_path))
        image_ids  = data["image_ids"]
        embeddings = data["embeddings"]

        # Rebuild df_meta from DB
        conn = sqlite3.connect(str(db_path))
        try:
            id_list = ",".join(str(x) for x in image_ids.tolist())
            df_meta = pd.read_sql_query(
                f"SELECT id AS image_id, macrophage_type, case_name "
                f"FROM images WHERE id IN ({id_list})",
                conn,
            )
        finally:
            conn.close()
        id_to_row = {row.image_id: row for _, row in df_meta.iterrows()}
        rows = []
        for iid in image_ids:
            if iid in id_to_row:
                rows.append(id_to_row[iid])
            else:
                rows.append({"image_id": int(iid), "macrophage_type": "unknown",
                             "case_name": "unknown"})
        df_meta = pd.DataFrame(rows).reset_index(drop=True)
        print(f"  Loaded {len(embeddings):,} embeddings (dim={embeddings.shape[1]})")
    else:
        print("Building embedding model …")
        model, device = build_embedding_model(use_cuda)
        print(f"  Model on {device}")
        print()

        print("Extracting embeddings …")
        image_ids, embeddings, df_meta = extract_embeddings(
            h5_dir, db_path, model, device, args.batch_size, args.img_size
        )
        np.savez(str(npz_path), image_ids=image_ids, embeddings=embeddings)
        print(f"  Saved {len(embeddings):,} embeddings → {npz_path}")

    print()
    type_counts = df_meta["macrophage_type"].value_counts()
    print(f"Types found: {len(type_counts)}")
    for t, n in type_counts.items():
        print(f"  {t}: {n:,}")
    print()

    # --- Prepare ---
    print("Preparing embeddings …")
    X_scaled, y_labels, le, df_filtered = prepare_embeddings(
        embeddings, df_meta, args.min_cells_per_type, args.outlier_frac
    )
    print(f"  Final: {len(X_scaled):,} cells, {len(le.classes_)} types")
    print()

    # Balanced subsample for visualisations
    X_viz, y_viz = balanced_subsample(X_scaled, y_labels, cap_per_class=args.viz_cap)
    print(f"  Viz subsample: {len(X_viz):,} cells (≤{args.viz_cap} per type)")
    print()

    # --- PCA ---
    print("PCA …")
    pca, X_pca = run_pca_emb(X_scaled, y_labels, X_viz, y_viz, le, out_dir)
    print()

    # --- t-SNE ---
    print("t-SNE …")
    run_tsne_emb(X_viz, y_viz, le, out_dir)
    print()

    # --- UMAP ---
    if not args.no_umap:
        print("UMAP …")
        run_umap_emb(X_viz, y_viz, le, out_dir)
        print()

    # --- Random Forest ---
    print("Random Forest …")
    rf, cv_acc, cv_bal = run_rf_emb(X_scaled, y_labels, le, out_dir)
    print()

    # --- KW on PC scores ---
    print("KW on PC scores …")
    run_kw_pc_scores(X_scaled, y_labels, le, pca, out_dir)
    print()

    # --- Hierarchical clustering (balanced subsample) ---
    print("Hierarchical clustering …")
    ari = plot_hierarchical_clustering(X_viz, y_viz, le, out_dir)
    print()

    # --- summary.txt ---
    summary_lines = [
        "embed.py summary",
        f"DB                   : {db_path}",
        f"H5 dir               : {h5_dir}",
        f"Embedding dim        : {embeddings.shape[1]}",
        f"Image resize         : {args.img_size}x{args.img_size}",
        f"Total cells          : {len(X_scaled):,}",
        f"Types                : {', '.join(le.classes_)}",
        f"Outlier frac removed : {args.outlier_frac*100:.1f}%",
        f"Viz cap per type     : {args.viz_cap:,}  (n_viz={len(X_viz):,})",
        "",
        f"RF 5-fold accuracy          : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}",
        f"RF 5-fold balanced accuracy : {cv_bal.mean():.3f} ± {cv_bal.std():.3f}",
        f"Clustering ARI              : {ari:.3f}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    print("Done.")
    print(f"  Output: {out_dir}/")
    print(f"  RF accuracy          : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"  RF balanced accuracy : {cv_bal.mean():.3f} ± {cv_bal.std():.3f}")
    print(f"  ARI                  : {ari:.3f}")


if __name__ == "__main__":
    main()
