"""
segment.py - Combined CNN classify + Cellpose segment pipeline.

Reads all images that passed the rule-based filter (`filtered_images` table),
classifies each with the CNN, and for every "good" image runs Cellpose to
produce a single-cell segmentation mask.

Results are written to:
  - SQLite `classified_images` table  (CNN result for every image)
  - SQLite `segmented_images` table   (Cellpose result for every CNN-good image)
  - SQLite `cell_stats` table         (per-cell morphology + channel intensities)
  - HDF5 `cells.h5` per case/macrophage directory (stacked images + masks)

Why combined (not two separate scripts):
  NTFS with millions of small files is the bottleneck. Two separate passes
  (classify then segment) would traverse the filesystem twice. This script
  reads each file exactly once.

HDF5 layout (one file per case/macrophage directory):
  /images     (N, 6, H, W)  float32   normalised [0, 1]
  /masks      (N, H, W)     uint16    single-cell binary mask
  /image_ids  (N,)          int64     SQLite images.id

Pipeline architecture:
  [N I/O+CNN workers] → cp_queue → [1 Cellpose worker] → write_queue
                      → cnn_queue                       → [1 writer thread]

  I/O+CNN workers  : directory-exclusive (same NTFS optimisation as filter.py).
                     CNN-good images buffered per-directory, sent as a batch.
  Cellpose worker  : single GPU instance shared by all; runs one batch per
                     directory to keep memory bounded.
  Writer thread    : drains both queues, writes HDF5 + three DB tables.

Channel composition (Rule B):
  - ch0 (nucleus): always the nuclear channel
  - ch5 (membrane) + clean marker channels (ch1-ch4) max-projected → cyto channel
  - "Noisy" = largest connected component after Otsu < min_area_frac of image
  - All-noisy fallback → nuclei mode

Middle-cell selection (Rule A):
  If Cellpose finds multiple cells, keep the one whose centroid is closest to
  image centre AND within max_distance_ratio x image_diagonal of the centre.
  If no cell meets the distance criterion → empty mask.

Two-phase parallelisation:
  Phase 1 - small dirs (< large_dir_threshold images): each dir owned by one
             worker, all dirs processed in parallel.
  Phase 2 - large dirs (≥ large_dir_threshold): processed one at a time, but
             the dir's rows are split across all workers for intra-dir parallelism.

Usage:
    python segment.py [--db image_index.sqlite] [--config config.json]
                      [--cnn-model cnn_model.pt]
                      [--out-dir stacked_cells/]
                      [--workers 8]
                      [--cp-batch 16]
                      [--model cyto3]
                      [--diameter 0]
                      [--min-area-frac 0.03]
                      [--no-cuda]
                      [--resume]
                      [--large-dir-threshold 80000]
"""

import argparse
import json
import math
import queue
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from skimage.filters import threshold_otsu
from skimage.measure import label as sk_label, regionprops

from train_classifier_cnn import CellQualityCNN, load_config, load_image_6ch


# ---------------------------------------------------------------------------
# DB schemas
# ---------------------------------------------------------------------------

CLASSIFIED_SCHEMA = """
CREATE TABLE IF NOT EXISTS classified_images (
    image_id         INTEGER PRIMARY KEY REFERENCES images(id),
    predicted_label  TEXT,
    confidence       REAL,
    model_version    TEXT,
    classified_at    TEXT
);
"""

SEGMENTED_SCHEMA = """
CREATE TABLE IF NOT EXISTS segmented_images (
    image_id      INTEGER PRIMARY KEY REFERENCES images(id),
    stack_path    TEXT,
    stack_index   INTEGER,
    mask_empty    INTEGER,
    segmented_at  TEXT
);
"""

CELL_STATS_SCHEMA = """
CREATE TABLE IF NOT EXISTS cell_stats (
    image_id      INTEGER PRIMARY KEY REFERENCES images(id),
    cell_area     REAL, perimeter    REAL, eccentricity REAL,
    major_axis    REAL, minor_axis   REAL, solidity     REAL,
    extent        REAL, circularity  REAL, centroid_dist REAL,
    ch0_mean REAL, ch0_std REAL,
    ch1_mean REAL, ch1_std REAL,
    ch2_mean REAL, ch2_std REAL,
    ch3_mean REAL, ch3_std REAL,
    ch4_mean REAL, ch4_std REAL,
    ch5_mean REAL, ch5_std REAL
);
"""

CLASSIFIED_INSERT = """
    INSERT OR REPLACE INTO classified_images
        (image_id, predicted_label, confidence, model_version, classified_at)
    VALUES (?, ?, ?, ?, ?)
"""

SEGMENTED_INSERT = """
    INSERT OR REPLACE INTO segmented_images
        (image_id, stack_path, stack_index, mask_empty, segmented_at)
    VALUES (?, ?, ?, ?, ?)
"""

CELL_STATS_INSERT = """
    INSERT OR REPLACE INTO cell_stats
        (image_id,
         cell_area, perimeter, eccentricity, major_axis, minor_axis,
         solidity, extent, circularity, centroid_dist,
         ch0_mean, ch0_std, ch1_mean, ch1_std,
         ch2_mean, ch2_std, ch3_mean, ch3_std,
         ch4_mean, ch4_std, ch5_mean, ch5_std)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# ---------------------------------------------------------------------------
# Noise detection + channel composition (mirrors test_cellpose.py)
# ---------------------------------------------------------------------------

def is_noisy_channel(ch: np.ndarray, min_area_frac: float = 0.03) -> bool:
    """True if no single component covers ≥ min_area_frac of image pixels."""
    try:
        thresh = threshold_otsu(ch)
    except Exception:
        return True
    binary = ch >= thresh
    if not binary.any():
        return True
    props = regionprops(sk_label(binary))
    if not props:
        return True
    largest_area = max(p.area for p in props)
    return (largest_area / ch.size) < min_area_frac


def build_cellpose_input(img6ch: np.ndarray, min_area_frac: float = 0.03):
    """
    Compose Cellpose input from a 6-channel image.

    Returns (cp_img, cp_channels).
    cp_img: (H,W) for nuclei mode or (H,W,2) for cyto mode.
    cp_channels: [0,0] or [1,2].
    """
    nuc_ch = img6ch[0]
    C      = img6ch.shape[0]

    cyto_chans = []

    if C > 5 and not is_noisy_channel(img6ch[5], min_area_frac):
        cyto_chans.append(img6ch[5])

    for i in range(1, min(5, C)):
        if not is_noisy_channel(img6ch[i], min_area_frac):
            cyto_chans.append(img6ch[i])

    if not cyto_chans:
        return nuc_ch, [0, 0]

    cyto_composite = np.max(np.stack(cyto_chans, axis=0), axis=0)
    return np.stack([cyto_composite, nuc_ch], axis=-1), [1, 2]


def select_middle_cell(mask: np.ndarray, max_distance_ratio: float = 0.2) -> np.ndarray:
    """Return uint16 binary mask for the cell closest to image centre.

    The cell centroid must lie within max_distance_ratio x image_diagonal of
    the centre.  If no cell meets the criterion, returns an all-zero mask.
    """
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]

    if len(unique_ids) == 0:
        return np.zeros(mask.shape, dtype=np.uint16)

    H, W = mask.shape
    cy, cx = H / 2.0, W / 2.0
    max_dist = max_distance_ratio * math.sqrt(H ** 2 + W ** 2)

    best_id   = None
    best_dist = float("inf")

    for cell_id in unique_ids:
        ys, xs = np.where(mask == cell_id)
        if len(ys) == 0:
            continue
        d = math.sqrt((ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2)
        if d <= max_dist and d < best_dist:
            best_dist = d
            best_id   = cell_id

    if best_id is None:
        return np.zeros(mask.shape, dtype=np.uint16)

    return (mask == best_id).astype(np.uint16)


# ---------------------------------------------------------------------------
# Per-cell statistics extraction
# ---------------------------------------------------------------------------

def extract_cell_stats(img6ch: np.ndarray, mask: np.ndarray):
    """
    Compute morphology + per-channel intensity stats for the selected cell.

    Parameters
    ----------
    img6ch : (6, H, W) float32 normalised [0,1]
    mask   : (H, W) uint16 binary mask (1=cell, 0=background)

    Returns
    -------
    dict with all cell_stats columns, or None-filled dict if mask is empty.
    """
    null_row = {
        "cell_area": None, "perimeter": None, "eccentricity": None,
        "major_axis": None, "minor_axis": None, "solidity": None,
        "extent": None, "circularity": None, "centroid_dist": None,
        **{f"ch{i}_mean": None for i in range(6)},
        **{f"ch{i}_std":  None for i in range(6)},
    }

    if mask.max() == 0:
        return null_row

    binary = (mask > 0).astype(np.uint8)
    labeled = sk_label(binary)
    props_list = regionprops(labeled)
    if not props_list:
        return null_row
    p = props_list[0]

    H, W = mask.shape
    half_diag = math.sqrt(H ** 2 + W ** 2) / 2.0
    cy_centroid, cx_centroid = p.centroid
    centroid_dist = math.sqrt((cy_centroid - H / 2.0) ** 2 +
                               (cx_centroid - W / 2.0) ** 2) / (half_diag + 1e-9)

    perim = p.perimeter
    area  = float(p.area)
    if perim > 0:
        circularity = 4.0 * math.pi * area / (perim ** 2)
    else:
        circularity = 0.0

    stats: dict = {
        "cell_area":     area,
        "perimeter":     float(perim),
        "eccentricity":  float(p.eccentricity),
        "major_axis":    float(p.major_axis_length),
        "minor_axis":    float(p.minor_axis_length),
        "solidity":      float(p.solidity),
        "extent":        float(p.extent),
        "circularity":   circularity,
        "centroid_dist": centroid_dist,
    }

    mask_bool = binary.astype(bool)
    for i in range(min(6, img6ch.shape[0])):
        ch_pixels = img6ch[i][mask_bool]
        stats[f"ch{i}_mean"] = float(ch_pixels.mean())
        stats[f"ch{i}_std"]  = float(ch_pixels.std())

    # Fill any missing channels (if img6ch has fewer than 6)
    for i in range(img6ch.shape[0], 6):
        stats[f"ch{i}_mean"] = None
        stats[f"ch{i}_std"]  = None

    return stats


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def init_tables(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(CLASSIFIED_SCHEMA)
        conn.execute(SEGMENTED_SCHEMA)
        conn.execute(CELL_STATS_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def fetch_candidates(db_path: Path, out_dir: Path, resume: bool) -> list:
    """
    Return (id, file_path, case_name, macrophage_type) for all pass_filter=1 images.
    With --resume, skip directories that already have a cells.h5 output file.
    """
    skip_dirs: set = set()
    if resume and out_dir.exists():
        for h5_path in out_dir.glob("*/*/cells.h5"):
            case_name      = h5_path.parent.parent.name
            macrophage_type = h5_path.parent.name
            skip_dirs.add((case_name, macrophage_type))
        if skip_dirs:
            print(f"Resume: skipping {len(skip_dirs)} already-completed director(ies).")

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT i.id, i.file_path, i.case_name, i.macrophage_type
            FROM images i
            JOIN filtered_images f ON f.image_id = i.id
            WHERE f.pass_filter = 1
            ORDER BY i.case_name, i.macrophage_type, i.id
            """
        ).fetchall()
    finally:
        conn.close()

    if skip_dirs:
        rows = [r for r in rows if (r[2], r[3]) not in skip_dirs]

    return rows


def distribute_by_directory(rows: list, num_workers: int) -> list:
    """Assign whole (case/macrophage_type) directories to workers round-robin."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row[2], row[3])].append(row)

    sorted_groups  = sorted(groups.values(), key=len, reverse=True)
    actual_workers = min(num_workers, len(sorted_groups))
    assignments    = [[] for _ in range(actual_workers)]

    for i, group in enumerate(sorted_groups):
        assignments[i % actual_workers].extend(group)

    return assignments


# ---------------------------------------------------------------------------
# I/O + CNN worker
# ---------------------------------------------------------------------------

def io_cnn_worker(row_list: list, root_str: str, model_state: dict,
                  device_str: str, cnn_batch_size: int, min_area_frac: float,
                  cp_queue: queue.Queue, cnn_queue: queue.Queue) -> None:
    """
    Processes all rows assigned to this worker (exclusive directory ownership).

    For each directory:
      1. Load images + run CNN classify in mini-batches (GPU efficiency).
      2. Push ALL CNN results to cnn_queue (for classified_images DB write).
      3. Collect CNN-good images in memory.
      4. Push (good_imgs, good_ids, dir_key) to cp_queue for Cellpose.

    Sends ("DONE",) to cp_queue when finished.
    """
    device    = torch.device(device_str)
    cnn_model = CellQualityCNN(in_channels=model_state["model_config"]["in_channels"])
    cnn_model.load_state_dict(model_state["model_state_dict"])
    cnn_model.to(device)
    cnn_model.eval()
    threshold     = model_state.get("threshold", 0.5)
    model_version = model_state.get("model_version", "cnn_model")
    now           = datetime.now().isoformat(timespec="seconds")

    # Group rows by directory (already exclusive, but explicit for clarity)
    dir_groups: dict = defaultdict(list)
    for row in row_list:
        dir_groups[(row[2], row[3])].append(row)

    for dir_key, dir_rows in dir_groups.items():
        dir_good_imgs: list = []
        dir_good_ids:  list = []

        for batch_start in range(0, len(dir_rows), cnn_batch_size):
            batch = dir_rows[batch_start: batch_start + cnn_batch_size]

            imgs:     list = []
            ids:      list = []

            for row in batch:
                image_id, file_path_str, _, _ = row
                try:
                    img = load_image_6ch(Path(root_str) / file_path_str)
                    imgs.append(img)
                    ids.append(image_id)
                except Exception:
                    cnn_queue.put((image_id, "bad", 0.0, model_version, now))

            if not imgs:
                continue

            batch_t = torch.from_numpy(np.stack(imgs, axis=0)).to(device)
            with torch.no_grad():
                logits = cnn_model(batch_t)
                probs  = torch.sigmoid(logits).cpu().numpy()

            for img, image_id, prob in zip(imgs, ids, probs):
                prob_f = float(prob)
                is_good = prob_f >= threshold
                label_str  = "good" if is_good else "bad"
                confidence = prob_f if is_good else (1.0 - prob_f)
                cnn_queue.put((image_id, label_str, confidence, model_version, now))

                if is_good:
                    dir_good_imgs.append(img)
                    dir_good_ids.append(image_id)

        # Send directory batch (possibly empty) to Cellpose worker
        cp_queue.put((dir_good_imgs, dir_good_ids, dir_key))

    cp_queue.put(("DONE",))


# ---------------------------------------------------------------------------
# Cellpose worker
# ---------------------------------------------------------------------------

def cellpose_worker(cp_queue: queue.Queue, write_queue: queue.Queue,
                    model_type: str, diameter: float, min_area_frac: float,
                    use_cuda: bool, num_io_workers: int,
                    cp_batch_size: int) -> None:
    """
    Single Cellpose GPU instance (v4.x API). Drains cp_queue, segments good
    images, pushes segmentation results to write_queue.

    Signals done by putting None on write_queue.
    """
    try:
        from cellpose import models as cp_models
    except ImportError:
        raise SystemExit("cellpose not installed: pip install cellpose")

    use_gpu  = use_cuda and torch.cuda.is_available()
    cp_model = cp_models.CellposeModel(gpu=use_gpu, model_type=model_type)
    cp_diam  = diameter if diameter > 0 else None
    now      = datetime.now().isoformat(timespec="seconds")
    done_ctr = 0

    while done_ctr < num_io_workers:
        item = cp_queue.get()

        if item[0] == "DONE":
            done_ctr += 1
            continue

        dir_good_imgs, dir_good_ids, dir_key = item

        if not dir_good_imgs:
            continue

        seg_results: list = []   # list of (img6ch, mask, image_id, stats_dict)

        for batch_start in range(0, len(dir_good_imgs), cp_batch_size):
            batch_imgs = dir_good_imgs[batch_start: batch_start + cp_batch_size]
            batch_ids  = dir_good_ids[batch_start: batch_start + cp_batch_size]

            # Build Cellpose inputs - shape carries channel info; channels=None
            cp_inputs = [build_cellpose_input(img6ch, min_area_frac)[0]
                         for img6ch in batch_imgs]

            try:
                # v4.x: 3-tuple (masks, flows, styles), channels=None
                masks_sub, _, _ = cp_model.eval(
                    cp_inputs,
                    diameter=cp_diam,
                    channels=None,
                    do_3D=False,
                )
            except Exception:
                empty_shape = batch_imgs[0].shape[1:]
                masks_sub = [np.zeros(empty_shape, dtype=np.uint16)
                             for _ in batch_imgs]

            for img6ch, raw_mask, image_id in zip(batch_imgs, masks_sub, batch_ids):
                mask  = select_middle_cell(raw_mask, max_distance_ratio=0.2)
                stats = extract_cell_stats(img6ch, mask)
                seg_results.append((img6ch, mask, image_id, stats))

        write_queue.put((seg_results, dir_key, now))

    write_queue.put(None)


# ---------------------------------------------------------------------------
# Writer thread (HDF5 + DB)
# ---------------------------------------------------------------------------

def writer_thread_fn(db_path: Path, out_dir: Path,
                     cnn_queue: queue.Queue, write_queue: queue.Queue,
                     cnn_db_batch: int, stats: dict) -> None:
    """
    Two input queues:
      cnn_queue   : (image_id, label, confidence, model_version, ts)  → classified_images
      write_queue : (seg_results, dir_key, ts) | None  → HDF5 + segmented_images
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA cache_size=-32000;")

    cnn_buf: list = []
    seg_done      = False

    def _flush_cnn():
        if cnn_buf:
            conn.executemany(CLASSIFIED_INSERT, cnn_buf)
            conn.commit()
            stats["cnn_written"] += len(cnn_buf)
            cnn_buf.clear()

    def _write_hdf5(seg_results, dir_key, ts):
        case_name, macrophage_type = dir_key
        h5_dir = out_dir / case_name / macrophage_type
        h5_dir.mkdir(parents=True, exist_ok=True)
        h5_path = h5_dir / "cells.h5"

        good_imgs, good_masks, good_ids = [], [], []
        seg_rows:       list = []
        cell_stats_rows: list = []

        for img6ch, mask, image_id, cell_st in seg_results:
            empty     = int(mask.max() == 0)
            stack_idx = len(good_ids)
            rel_path  = str(
                (h5_path).relative_to(out_dir)
                if h5_path.is_relative_to(out_dir)
                else h5_path
            )
            seg_rows.append((image_id, str(rel_path), stack_idx, empty, ts))
            if not empty:
                good_imgs.append(img6ch)
                good_masks.append(mask)
                good_ids.append(image_id)
            stats["seg_total"] += 1
            if not empty:
                stats["seg_ok"] += 1

            # cell_stats row (NULL-filled for empty masks, which have None values)
            if cell_st is not None and cell_st.get("cell_area") is not None:
                cell_stats_rows.append((
                    image_id,
                    cell_st["cell_area"],  cell_st["perimeter"],
                    cell_st["eccentricity"], cell_st["major_axis"],
                    cell_st["minor_axis"],  cell_st["solidity"],
                    cell_st["extent"],      cell_st["circularity"],
                    cell_st["centroid_dist"],
                    cell_st["ch0_mean"], cell_st["ch0_std"],
                    cell_st["ch1_mean"], cell_st["ch1_std"],
                    cell_st["ch2_mean"], cell_st["ch2_std"],
                    cell_st["ch3_mean"], cell_st["ch3_std"],
                    cell_st["ch4_mean"], cell_st["ch4_std"],
                    cell_st["ch5_mean"], cell_st["ch5_std"],
                ))

        # Write HDF5 only if there are non-empty masks
        if good_imgs:
            imgs_arr  = np.stack(good_imgs,  axis=0)  # (N, 6, H, W) float32
            masks_arr = np.stack(good_masks, axis=0)  # (N, H, W)    uint16
            ids_arr   = np.array(good_ids,   dtype=np.int64)

            mode = "a" if h5_path.exists() else "w"
            with h5py.File(str(h5_path), mode) as hf:
                if "images" in hf:
                    # Append to existing datasets
                    old_n = hf["images"].shape[0]
                    new_n = old_n + len(good_imgs)
                    hf["images"].resize(new_n, axis=0)
                    hf["masks"].resize(new_n, axis=0)
                    hf["image_ids"].resize(new_n, axis=0)
                    hf["images"][old_n:]    = imgs_arr
                    hf["masks"][old_n:]     = masks_arr
                    hf["image_ids"][old_n:] = ids_arr
                else:
                    H, W = imgs_arr.shape[2], imgs_arr.shape[3]
                    hf.create_dataset(
                        "images",
                        data=imgs_arr,
                        maxshape=(None, 6, H, W),
                        chunks=(1, 6, H, W),
                        compression="lz4" if _has_lz4() else "gzip",
                    )
                    hf.create_dataset(
                        "masks",
                        data=masks_arr,
                        maxshape=(None, H, W),
                        chunks=(1, H, W),
                        compression="lz4" if _has_lz4() else "gzip",
                    )
                    hf.create_dataset(
                        "image_ids",
                        data=ids_arr,
                        maxshape=(None,),
                        chunks=(min(1024, len(ids_arr)),),
                    )
            stats["hdf5_written"] += len(good_imgs)

        # DB: segmented_images
        if seg_rows:
            conn.executemany(SEGMENTED_INSERT, seg_rows)
            conn.commit()

        # DB: cell_stats (only for non-empty masks)
        if cell_stats_rows:
            conn.executemany(CELL_STATS_INSERT, cell_stats_rows)
            conn.commit()

    try:
        while not seg_done:
            # Drain CNN queue first (non-blocking)
            while True:
                try:
                    item = cnn_queue.get_nowait()
                    cnn_buf.append(item)
                    if len(cnn_buf) >= cnn_db_batch:
                        _flush_cnn()
                except queue.Empty:
                    break

            # Check segmentation queue (blocking 0.5s)
            try:
                seg_item = write_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if seg_item is None:
                seg_done = True
            else:
                seg_results, dir_key, ts = seg_item
                _write_hdf5(seg_results, dir_key, ts)

        # Drain remaining CNN items
        while True:
            try:
                cnn_buf.append(cnn_queue.get_nowait())
                if len(cnn_buf) >= cnn_db_batch:
                    _flush_cnn()
            except queue.Empty:
                break
        _flush_cnn()

    finally:
        conn.close()


def _has_lz4() -> bool:
    """Check if hdf5plugin/lz4 compression is available."""
    try:
        import hdf5plugin  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

def group_by_directory(rows: list) -> dict:
    """Return {(case_name, macrophage_type): [rows]} dict."""
    groups: dict = defaultdict(list)
    for row in rows:
        groups[(row[2], row[3])].append(row)
    return dict(groups)


def _run_pipeline(
    assignments: list,
    root_dir: Path,
    model_state: dict,
    device_str: str,
    cnn_batch_size: int,
    cp_batch_size: int,
    model_type: str,
    diameter: float,
    min_area_frac: float,
    use_cuda: bool,
    out_dir: Path,
    db_path: Path,
    stats: dict,
    total_label: str = "",
) -> None:
    """
    Spin up the full I/O+CNN → Cellpose → writer pipeline and run to completion.

    `assignments` is a list of row_lists, one per I/O+CNN worker.
    Called once for phase-1 (all small dirs) and once per large dir in phase-2.
    """
    n_workers = len(assignments)
    total_rows = sum(len(a) for a in assignments)

    cp_queue:    queue.Queue = queue.Queue(maxsize=n_workers * 4)
    write_queue: queue.Queue = queue.Queue(maxsize=n_workers * 4)
    cnn_queue:   queue.Queue = queue.Queue(maxsize=n_workers * 512)

    writer = threading.Thread(
        target=writer_thread_fn,
        args=(db_path, out_dir, cnn_queue, write_queue, 2000, stats),
        daemon=True,
    )
    writer.start()

    cellpose_th = threading.Thread(
        target=cellpose_worker,
        args=(cp_queue, write_queue, model_type, diameter, min_area_frac,
              use_cuda, n_workers, cp_batch_size),
        daemon=True,
    )
    cellpose_th.start()

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                io_cnn_worker,
                assignment, str(root_dir), model_state, device_str,
                cnn_batch_size, min_area_frac, cp_queue, cnn_queue,
            )
            for assignment in assignments
        ]

        prefix = f"[{total_label}] " if total_label else ""
        done_count = 0
        while done_count < n_workers:
            done_count = sum(1 for f in futures if f.done())
            elapsed = time.time() - start_time
            rate    = (stats["cnn_written"] / elapsed) if elapsed > 0 else 0
            print(
                f"  {prefix}CNN {stats['cnn_written']:,}/{total_rows:,} @ {rate:.0f} img/s  "
                f"Seg OK {stats['seg_ok']:,}  "
                f"[{done_count}/{n_workers} workers done]",
                end="\r", flush=True,
            )
            time.sleep(2.0)

        for f in futures:
            exc = f.exception()
            if exc:
                print(f"\n[WARN] Worker raised: {exc}")

    cellpose_th.join()
    writer.join()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_segment(db_path: Path, root_dir: Path, model_path: Path,
                out_dir: Path, workers: int, cnn_batch_size: int,
                cp_batch_size: int, model_type: str, diameter: float,
                min_area_frac: float, use_cuda: bool, resume: bool,
                large_dir_threshold: int = 80_000) -> None:

    device_str = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    model_state = torch.load(str(model_path), map_location="cpu")
    model_state["model_version"] = model_path.stem

    print(f"CNN model    : {model_path.name}  "
          f"(threshold={model_state.get('threshold', 0.5):.3f})")
    print(f"Cellpose     : {model_type}  diameter={diameter or 'auto'}")
    print(f"Device       : {device_str}")
    if device_str == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"Output dir   : {out_dir}")

    init_tables(db_path)

    rows = fetch_candidates(db_path, out_dir, resume=resume)
    total = len(rows)
    if total == 0:
        print("Nothing to do.")
        return

    dir_groups = group_by_directory(rows)
    small_dirs = {k: v for k, v in dir_groups.items() if len(v) <  large_dir_threshold}
    large_dirs = {k: v for k, v in dir_groups.items() if len(v) >= large_dir_threshold}

    n_dirs = len(dir_groups)
    print(f"Images       : {total:,}  ({'resume' if resume else 'full'} mode)")
    print(f"Directories  : {n_dirs}  "
          f"(small={len(small_dirs)}, large={len(large_dirs)}, "
          f"threshold={large_dir_threshold:,})")
    print(f"I/O workers  : {workers}")
    print()

    stats = {
        "cnn_written":  0,
        "seg_total":    0,
        "seg_ok":       0,
        "hdf5_written": 0,
    }

    start_time = time.time()

    # ------------------------------------------------------------------
    # Phase 1: all small directories in parallel
    # ------------------------------------------------------------------
    if small_dirs:
        small_rows = [r for rows_list in small_dirs.values() for r in rows_list]
        assignments = distribute_by_directory(small_rows, workers)
        n_w = len(assignments)
        print(f"Phase 1: {len(small_dirs)} small dir(s), {len(small_rows):,} images, "
              f"{n_w} worker(s)")
        _run_pipeline(
            assignments, root_dir, model_state, device_str,
            cnn_batch_size, cp_batch_size,
            model_type, diameter, min_area_frac, use_cuda,
            out_dir, db_path, stats,
            total_label=f"phase1/{len(small_rows):,}",
        )
        print()

    # ------------------------------------------------------------------
    # Phase 2: large directories, one at a time, all workers per dir
    # ------------------------------------------------------------------
    for phase_idx, (dir_key, dir_rows) in enumerate(
        sorted(large_dirs.items(), key=lambda x: -len(x[1]))
    ):
        case_name, mac_type = dir_key
        n_w = min(workers, len(dir_rows))
        # Split rows into n_w interleaved chunks (worker i gets rows i, i+n_w, i+2n_w…)
        chunks = [dir_rows[i::n_w] for i in range(n_w)]
        chunks = [c for c in chunks if c]

        print(f"Phase 2 [{phase_idx+1}/{len(large_dirs)}]: "
              f"{case_name}/{mac_type}  {len(dir_rows):,} images  "
              f"{len(chunks)} worker(s)")
        _run_pipeline(
            chunks, root_dir, model_state, device_str,
            cnn_batch_size, cp_batch_size,
            model_type, diameter, min_area_frac, use_cuda,
            out_dir, db_path, stats,
            total_label=f"{case_name}/{mac_type}",
        )
        print()

    elapsed = time.time() - start_time
    rate    = total / elapsed if elapsed > 0 else 0

    print(f"Done. {total:,} images in {elapsed:.1f}s ({rate:.0f} img/s).")
    print(f"  CNN good          : {stats['cnn_written']:,} classified  "
          f"(~{stats['seg_total']:,} sent to Cellpose)")
    print(f"  Segmented OK      : {stats['seg_ok']:,}")
    print(f"  HDF5 cells stored : {stats['hdf5_written']:,}")

    _print_hdf5_summary(out_dir)


def _print_hdf5_summary(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    h5_files = sorted(out_dir.glob("*/*/cells.h5"))
    if not h5_files:
        return

    print(f"\nHDF5 summary ({len(h5_files)} file(s)):")
    print("-" * 60)
    for h5_path in h5_files:
        rel = h5_path.relative_to(out_dir)
        try:
            with h5py.File(str(h5_path), "r") as hf:
                n = hf["images"].shape[0] if "images" in hf else 0
        except Exception:
            n = "?"
        print(f"  {str(rel):<45}  {n} cells")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Combined CNN classify + Cellpose segment pipeline."
    )
    parser.add_argument("--db",            help="SQLite DB path (overrides config)")
    parser.add_argument("--config",        help="Path to config.json")
    parser.add_argument("--cnn-model",     default="cnn_model.pt",
                        help="CNN model checkpoint (default: cnn_model.pt)")
    parser.add_argument("--out-dir",       default="stacked_cells",
                        help="Root for HDF5 output (default: stacked_cells/)")
    parser.add_argument("--workers",       type=int,   default=8,
                        help="I/O + CNN worker threads (default: 8)")
    parser.add_argument("--cnn-batch",     type=int,   default=64,
                        help="Images per CNN inference batch (default: 64)")
    parser.add_argument("--cp-batch",      type=int,   default=16,
                        help="Images per Cellpose batch (default: 16)")
    parser.add_argument("--model",         default="cyto3",
                        help="Cellpose model: cyto3, cyto2, nuclei (default: cyto3)")
    parser.add_argument("--diameter",      type=float, default=0,
                        help="Cell diameter in pixels; 0 = auto (default: 0)")
    parser.add_argument("--min-area-frac", type=float, default=0.03,
                        help="Noise threshold for channel skip (default: 0.03)")
    parser.add_argument("--no-cuda",       action="store_true")
    parser.add_argument("--resume",        action="store_true",
                        help="Skip directories whose cells.h5 already exists")
    parser.add_argument("--large-dir-threshold", type=int, default=80_000,
                        help="Dirs with >= this many images use phase-2 "
                             "intra-dir parallelism (default: 80000)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    root_dir = Path(cfg["project"]["root_dir"])
    if not root_dir.exists():
        raise SystemExit(f"Root directory not found: {root_dir}")

    db_raw  = Path(args.db) if args.db else Path(cfg["project"]["db_path"])
    db_path = (Path.cwd() / db_raw if not db_raw.is_absolute() else db_raw).resolve()
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    model_path = Path(args.cnn_model)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        raise SystemExit(
            f"CNN model not found: {model_path}. Run train_classifier_cnn.py first."
        )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir

    run_segment(
        db_path=db_path,
        root_dir=root_dir,
        model_path=model_path,
        out_dir=out_dir,
        workers=args.workers,
        cnn_batch_size=args.cnn_batch,
        cp_batch_size=args.cp_batch,
        model_type=args.model,
        diameter=args.diameter,
        min_area_frac=args.min_area_frac,
        use_cuda=not args.no_cuda,
        resume=args.resume,
        large_dir_threshold=args.large_dir_threshold,
    )


if __name__ == "__main__":
    main()
