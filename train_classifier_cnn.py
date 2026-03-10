"""
train_classifier_cnn.py - Train a 6-channel CNN quality classifier.

Reads good/bad labels from `inspected_images`, loads the corresponding
OME-TIFF files from disk, and trains a small CNN that operates directly
on all 6 image channels.  Saves the best checkpoint to `cnn_model.pt`.

Why CNN over scalar features:
  The sklearn classifier uses 6 hand-crafted scalars (SNR, Laplacian, etc.)
  that discard spatial structure.  A CNN can learn to detect:
  - two overlapping cells
  - cell truncated at image edge
  - out-of-focus blur vs sharp background
  - unexpected artefacts in specific regions
  all without manual feature engineering.

Architecture: CellQualityCNN
  6 → 32 → 64 → 128 channels, two MaxPool layers,
  AdaptiveAvgPool2d(4) so any input size ≥ 16x16 works,
  binary output via FocalLoss.

Loss / imbalance strategy (labels ~97 good / ~719 bad):
  - FocalLoss(γ=2): suppresses easy-negative gradient so hard examples near
    the decision boundary dominate training.  α = n_bad/(n_good+n_bad) gives
    the rare good class adequate gradient signal.
  - Label smoothing (ε=0.05): prevents the model from fitting individual
    mislabelled images with extreme confidence.
  - WeightedRandomSampler at 3:1 bad:good: good images appear regularly
    without the overfitting risk of 1:1 oversampling.

Stopping metric: F_β with β=0.5 (weights precision 4x more than recall).

After training a threshold sweep finds the operating point that satisfies
precision_good ≥ target_precision while maximising recall.

Usage:
    python train_classifier_cnn.py [--db image_index.sqlite]
                                    [--model-out cnn_model.pt]
                                    [--epochs 60] [--batch-size 32]
                                    [--lr 1e-3] [--no-cuda] [--config config.json]
                                    [--target-precision 0.90]
"""

import argparse
import json
import random
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path = None) -> dict:
    defaults = {"project": {"root_dir": None, "db_path": "image_index.sqlite"}}
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        defaults.update(cfg)
    return defaults


# ---------------------------------------------------------------------------
# Image loading (mirrors filter.py)
# ---------------------------------------------------------------------------

def load_image_6ch(full_path: Path) -> np.ndarray:
    """
    Load an OME-TIFF and return a float32 array of shape (6, H, W),
    each channel normalised to [0, 1].
    """
    with tifffile.TiffFile(str(full_path)) as tif:
        series = tif.series[0]
        pages = series.pages
        C = len(pages)
        channels = []
        for i in range(C):
            ch = pages[i].asarray().astype(np.float32)
            mx = ch.max()
            if mx > 0:
                ch /= mx
            channels.append(ch)

    # Ensure exactly 6 channels (pad with zeros if fewer, truncate if more)
    while len(channels) < 6:
        channels.append(np.zeros_like(channels[0]))
    channels = channels[:6]

    return np.stack(channels, axis=0)  # (6, H, W)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CellDataset(Dataset):
    """
    PyTorch Dataset for cell quality classification.

    Args:
        records   : list of (image_id, file_path, label_int)  label_int: 1=good, 0=bad
        root_dir  : Path to image root directory
        augment   : apply random flips / rotations / brightness jitter
    """

    def __init__(self, records: list, root_dir: Path, augment: bool = False):
        self.records  = records
        self.root_dir = root_dir
        self.augment  = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image_id, file_path, label = self.records[idx]
        img = load_image_6ch(self.root_dir / file_path)  # (6, H, W) float32

        if self.augment:
            img = self._augment(img)

        tensor = torch.from_numpy(img)           # (6, H, W)
        target = torch.tensor(float(label))      # scalar
        return tensor, target, image_id

    @staticmethod
    def _augment(img: np.ndarray) -> np.ndarray:
        """In-place compatible augmentation - returns modified copy."""
        # Random 90° rotation (k = 0/1/2/3)
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k, axes=(1, 2)).copy()

        # Random horizontal flip
        if random.random() < 0.5:
            img = img[:, :, ::-1].copy()

        # Random vertical flip
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy()

        # Per-channel brightness jitter ±10 %
        factors = np.random.uniform(0.9, 1.1, size=(6, 1, 1)).astype(np.float32)
        img = np.clip(img * factors, 0.0, 1.0)

        return img


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CellQualityCNN(nn.Module):
    """
    Lightweight 6-channel CNN for binary quality classification.

    AdaptiveAvgPool2d(4) makes the model image-size-agnostic.
    The architecture is intentionally small to train well from ~4k samples.
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.block1 = _conv_block(in_channels, 32)
        self.pool1  = nn.MaxPool2d(2)
        self.drop1  = nn.Dropout2d(0.1)

        self.block2 = _conv_block(32, 64)
        self.pool2  = nn.MaxPool2d(2)
        self.drop2  = nn.Dropout2d(0.1)

        self.block3 = _conv_block(64, 128)
        self.gap    = nn.AdaptiveAvgPool2d(4)   # → (128, 4, 4) = 2048 features

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),                  # raw logit, no sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.pool1(self.block1(x)))
        x = self.drop2(self.pool2(self.block2(x)))
        x = self.gap(self.block3(x))
        return self.classifier(x).squeeze(1)    # (B,)


# ---------------------------------------------------------------------------
# Data loading from DB
# ---------------------------------------------------------------------------

def load_labelled_records(db_path: Path) -> list:
    """
    Return list of (image_id, file_path, label_int) for all good/bad labels.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT ii.image_id, i.file_path, ii.label
            FROM inspected_images ii
            JOIN images i ON i.id = ii.image_id
            WHERE ii.label IN ('good', 'bad')
            LIMIT 1250
            """
        ).fetchall()
    finally:
        conn.close()

    return [(r[0], r[1], 1 if r[2] == "good" else 0) for r in rows]


def stratified_split(records: list, val_frac: float = 0.2, seed: int = 42):
    """Stratified 80/20 split preserving class ratio."""
    rng = random.Random(seed)
    good = [r for r in records if r[2] == 1]
    bad  = [r for r in records if r[2] == 0]
    rng.shuffle(good)
    rng.shuffle(bad)

    def split_class(lst):
        n_val = max(1, int(len(lst) * val_frac))
        return lst[n_val:], lst[:n_val]

    tr_good, va_good = split_class(good)
    tr_bad,  va_bad  = split_class(bad)
    return tr_good + tr_bad, va_good + va_bad


# ---------------------------------------------------------------------------
# Loss: Focal Loss with label smoothing
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary Focal Loss with label smoothing.

    FL = -α_t · (1 − p_t)^γ · BCE(p, ŷ_smooth)

    Args:
        alpha        : weight for the positive (good) class [0, 1].
                       Typically set to n_bad / (n_good + n_bad) so the rare
                       good class receives adequate gradient signal.
        gamma        : focusing exponent.  γ=2 strongly down-weights easy
                       negatives (the model quickly assigns high confidence to
                       most bad images; those contribute little to the loss).
        label_smooth : ε for label smoothing; replaces 1→(1-ε), 0→ε.
                       Prevents overfitting to individual mislabelled images.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 label_smooth: float = 0.05):
        super().__init__()
        self.alpha        = alpha
        self.gamma        = gamma
        self.label_smooth = label_smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Soft targets via label smoothing
        eps = self.label_smooth
        targets_smooth = targets * (1.0 - eps) + (1.0 - targets) * eps

        p = torch.sigmoid(logits)

        # p_t: probability assigned to the correct (hard-label) class
        p_t     = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_w = (1.0 - p_t) ** self.gamma

        # Binary cross-entropy on smooth targets
        bce = -(
            targets_smooth * torch.log(p + 1e-8)
            + (1.0 - targets_smooth) * torch.log(1.0 - p + 1e-8)
        )

        return (alpha_t * focal_w * bce).mean()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_weighted_sampler(records: list) -> WeightedRandomSampler:
    """
    Over-sample so each batch has ~3x more bad than good images.

    Per-item weights: w_good = 1/n_good, w_bad = 3/n_bad.
    This gives total sampling weight ratio bad:good = 3:1 (75% bad, 25% good),
    a compromise between the natural 7:1 skew and aggressive 1:1 oversampling.
    """
    labels = [r[2] for r in records]
    n_good = sum(labels)
    n_bad  = len(labels) - n_good
    w_good = 1.0 / n_good if n_good else 1.0
    w_bad  = 3.0 / n_bad  if n_bad  else 1.0
    weights = [w_good if l == 1 else w_bad for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def compute_fbeta(preds: list, targets: list,
                  threshold: float = 0.5, beta: float = 0.5):
    """
    Compute F_β, precision, and recall for the positive (good) class.

    β < 1 weights precision more than recall.
    β = 0.5: F_0.5 = 1.25 · (P·R) / (0.25·P + R)
    """
    tp = fp = fn = 0
    for p, t in zip(preds, targets):
        pred_pos = p >= threshold
        if pred_pos and t == 1:
            tp += 1
        elif pred_pos and t == 0:
            fp += 1
        elif not pred_pos and t == 1:
            fn += 1
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2    = beta * beta
    fbeta = (1.0 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) > 0 else 0.0
    return fbeta, prec, rec


def threshold_sweep(val_preds: list, val_tgts: list) -> list:
    """
    Sweep thresholds from 0.10 to 0.95 (step 0.05).

    Returns list of (threshold, prec_good, rec_good, fbeta, kept_pct) where
    kept_pct is the fraction of val images classified as good at that threshold.
    """
    thresholds = np.round(np.arange(0.10, 1.00, 0.05), 2)
    n_total = len(val_preds)
    results = []
    for thr in thresholds:
        t = float(thr)
        fbeta, prec, rec = compute_fbeta(val_preds, val_tgts, threshold=t)
        kept     = sum(1 for p in val_preds if p >= t)
        kept_pct = 100.0 * kept / n_total if n_total > 0 else 0.0
        results.append((t, prec, rec, fbeta, kept_pct))
    return results


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(db_path: Path, root_dir: Path, model_out: Path,
          epochs: int, batch_size: int, lr: float, use_cuda: bool,
          target_precision: float = 0.90) -> None:

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    if device.type == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(0)}")

    # --- Data ---
    records = load_labelled_records(db_path)
    if not records:
        raise SystemExit("No labelled data found. Run inspect.py first.")

    n_good = sum(1 for r in records if r[2] == 1)
    n_bad  = len(records) - n_good
    print(f"Labels          : {len(records):,}  (good={n_good}, bad={n_bad})")
    if len(records) < 50:
        print("[WARN] Very few labels - results may be unreliable.")

    train_rec, val_rec = stratified_split(records, val_frac=0.2)
    print(f"Split           : {len(train_rec)} train / {len(val_rec)} val")

    alpha = n_bad / (n_good + n_bad) if (n_good + n_bad) > 0 else 0.75
    print(f"FocalLoss α     : {alpha:.3f}  γ=2  label_smooth=0.05")
    print(f"Sampler ratio   : 3:1 bad:good")
    print(f"Target precision: {target_precision:.2f}")
    print()

    train_ds = CellDataset(train_rec, root_dir, augment=True)
    val_ds   = CellDataset(val_rec,   root_dir, augment=False)

    sampler    = make_weighted_sampler(train_rec)
    train_dl   = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                            num_workers=0, pin_memory=(device.type == "cuda"))
    val_dl     = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    # --- Model ---
    model     = CellQualityCNN(in_channels=6).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0, label_smooth=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Training loop ---
    best_val_fbeta  = -1.0
    best_epoch      = 0
    patience        = 10
    patience_ctr    = 0
    history         = []
    best_val_preds  = []
    best_val_tgts   = []

    print(f"{'Epoch':>5}  {'TrainLoss':>9}  {'ValLoss':>7}  {'Prec(good)':>10}  {'Rec(good)':>9}  {'F0.5':>6}")
    print("-" * 62)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for imgs, targets, _ in train_dl:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_ds)
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss  = 0.0
        val_preds = []
        val_tgts  = []
        with torch.no_grad():
            for imgs, targets, _ in val_dl:
                imgs, targets = imgs.to(device), targets.to(device)
                logits = model(imgs)
                loss   = criterion(logits, targets)
                val_loss += loss.item() * len(imgs)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs.tolist())
                val_tgts.extend(targets.cpu().numpy().tolist())
        val_loss /= len(val_ds)

        val_fbeta, val_prec, val_rec = compute_fbeta(val_preds, val_tgts)

        marker = "  ← best" if val_fbeta > best_val_fbeta else ""
        print(f"{epoch:5d}  {train_loss:9.4f}  {val_loss:7.4f}  "
              f"{val_prec:10.3f}  {val_rec:9.3f}  {val_fbeta:6.3f}{marker}")

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "val_loss":   round(val_loss, 5),
            "val_fbeta":  round(val_fbeta, 4),
            "val_prec":   round(val_prec, 4),
            "val_rec":    round(val_rec, 4),
        })

        # --- Checkpoint best model ---
        if val_fbeta > best_val_fbeta:
            best_val_fbeta  = val_fbeta
            best_epoch      = epoch
            patience_ctr    = 0
            best_val_preds  = val_preds[:]
            best_val_tgts   = val_tgts[:]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config":     {"in_channels": 6},
                "threshold":        0.5,          # placeholder; updated after sweep
                "target_precision": target_precision,
                "val_precision_good": val_prec,
                "val_recall_good":    val_rec,
                "val_fbeta":          val_fbeta,
                "val_loss":           val_loss,
                "epoch":              epoch,
                "class_counts":       {"good": n_good, "bad": n_bad},
            }, model_out)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best F0.5={best_val_fbeta:.3f} at epoch {best_epoch})")
                break

    print(f"\nBest val F0.5 : {best_val_fbeta:.3f}  (epoch {best_epoch})")

    # --- Threshold sweep on best-epoch val predictions ---
    print("\nThreshold sweep on validation set:")
    print(f"{'Threshold':>9}  {'Prec(good)':>10}  {'Rec(good)':>9}  {'F0.5':>6}  {'Kept%':>7}")
    print("-" * 52)

    sweep = threshold_sweep(best_val_preds, best_val_tgts)

    # Select: highest F0.5 among thresholds with precision >= target_precision
    candidates = [(thr, prec, rec, fbeta, kp)
                  for thr, prec, rec, fbeta, kp in sweep
                  if prec >= target_precision]

    if candidates:
        best_row = max(candidates, key=lambda x: x[3])  # max F0.5
    else:
        # Fall back to highest precision point if target never reached
        best_row = max(sweep, key=lambda x: x[1])
        print(f"[WARN] No threshold reached precision ≥ {target_precision:.2f}; "
              f"falling back to highest-precision point.")

    chosen_thr, chosen_prec, chosen_rec, chosen_fbeta, chosen_kp = best_row

    for thr, prec, rec, fbeta, kp in sweep:
        marker = "  ← selected" if thr == chosen_thr else ""
        print(f"{thr:9.2f}  {prec:10.3f}  {rec:9.3f}  {fbeta:6.3f}  {kp:6.1f}%{marker}")

    print(f"\nChosen threshold : {chosen_thr:.2f}  "
          f"(prec={chosen_prec:.3f}, rec={chosen_rec:.3f}, F0.5={chosen_fbeta:.3f})")

    # Update checkpoint with chosen threshold
    checkpoint = torch.load(str(model_out), map_location="cpu")
    checkpoint["threshold"]          = chosen_thr
    checkpoint["val_precision_good"] = chosen_prec
    checkpoint["val_recall_good"]    = chosen_rec
    checkpoint["val_fbeta"]          = chosen_fbeta
    torch.save(checkpoint, model_out)

    print(f"Model saved : {model_out}")

    # --- Save metrics ---
    metrics_path = model_out.with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump({
            "best_epoch":        best_epoch,
            "best_val_fbeta":    round(best_val_fbeta, 4),
            "threshold":         chosen_thr,
            "target_precision":  target_precision,
            "val_precision_good": round(chosen_prec, 4),
            "val_recall_good":    round(chosen_rec, 4),
            "n_train":           len(train_rec),
            "n_val":             round(val_rec, 4),
            "class_counts":      {"good": n_good, "bad": n_bad},
            "threshold_sweep":   [
                {"threshold": t, "prec": round(p, 4), "rec": round(r, 4),
                 "fbeta": round(f, 4), "kept_pct": round(k, 2)}
                for t, p, r, f, k in sweep
            ],
            "history": history,
        }, f, indent=2)
    print(f"Metrics     : {metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a 6-channel CNN quality classifier for cell images."
    )
    parser.add_argument("--db",        help="SQLite DB path (overrides config)")
    parser.add_argument("--config",    help="Path to config.json")
    parser.add_argument("--model-out", default="cnn_model.pt",
                        help="Output path for model checkpoint (default: cnn_model.pt)")
    parser.add_argument("--epochs",           type=int,   default=60)
    parser.add_argument("--batch-size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--no-cuda",          action="store_true",
                        help="Disable GPU even if available")
    parser.add_argument("--target-precision", type=float, default=0.90,
                        help="Minimum precision on 'good' class when selecting "
                             "operating threshold (default: 0.90)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    root_dir = Path(cfg["project"]["root_dir"])
    if not root_dir.exists():
        raise SystemExit(f"Root directory does not exist: {root_dir}. Set it in config.json.")

    db_path_raw = Path(args.db) if args.db else Path(cfg["project"]["db_path"])
    db_path = (Path.cwd() / db_path_raw if not db_path_raw.is_absolute() else db_path_raw).resolve()
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}. Run index_images.py first.")

    model_out = Path(args.model_out)
    if not model_out.is_absolute():
        model_out = Path.cwd() / model_out

    train(
        db_path=db_path,
        root_dir=root_dir,
        model_out=model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_cuda=not args.no_cuda,
        target_precision=args.target_precision,
    )


if __name__ == "__main__":
    main()
