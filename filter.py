"""
filter.py - Rule-based quality filter for OME-TIFF cell images.

Reads all images from the `images` table, computes per-image quality metrics,
and writes results to the `filtered_images` table.

All thresholds and enable/disable toggles are read from config.json under the
"filtering" key. Edit config.json to tune thresholds without touching this code.

Performance design
------------------
- Work is partitioned by directory (case / macrophage_type).
  Each worker thread is assigned EXCLUSIVE ownership of a set of directories
  and processes them sequentially. No two workers ever compete for the same
  directory, eliminating NTFS directory-lock contention that causes the
  progressive slowdown observed on Windows.
  Total time ≈ max(worker_time_1, worker_time_2, …) instead of degrading.
- ThreadPoolExecutor (default 16 threads): no spawn overhead on Windows,
  GIL released during file I/O and numpy/cv2/skimage operations.
- Dedicated writer thread with bounded result queue:
  workers push dicts into a queue.Queue; a single writer thread holds one
  persistent SQLite connection (PRAGMA set once) and bulk-inserts batches.
  Bounded queue (maxsize = workers * 8) provides backpressure.
- Selective channel loading: only channels required by enabled rules are read.

Usage:
    python filter.py [--db image_index.sqlite] [--config config.json]
                     [--workers N] [--resume]

    --resume    Skip images already present in filtered_images (continue interrupted run)
"""

import argparse
import json
import math
import queue
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tifffile
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "project": {
        "root_dir": None,
        "db_path": "image_index.sqlite",
    },
    "filtering": {
        "workers": 16,
        "batch_size": 500,
        "rules": {
            "nucleus_snr": {"enabled": True, "min_snr": 5.0},
            "focus": {"enabled": True, "min_laplacian_var": 20.0},
            "cell_completeness": {
                "enabled": True,
                "max_centroid_border_frac": 0.30,
                "min_nucleus_area_frac": 0.02,
                "max_nucleus_area_frac": 0.50,
            },
            "membrane_signal": {"enabled": True, "min_mean": 0.01},
            "marker_signal": {"enabled": True, "min_max_mean": 0.01},
        },
    },
}


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

    if config_path.exists():
        with open(config_path, "r") as f:
            user_cfg = json.load(f)
        for key, val in user_cfg.items():
            if key in cfg and isinstance(cfg[key], dict) and isinstance(val, dict):
                cfg[key].update(val)
            else:
                cfg[key] = val
        if "filtering" in user_cfg and "rules" in user_cfg["filtering"]:
            for rule, rule_cfg in user_cfg["filtering"]["rules"].items():
                if rule in cfg["filtering"]["rules"] and isinstance(rule_cfg, dict):
                    cfg["filtering"]["rules"][rule].update(rule_cfg)
                else:
                    cfg["filtering"]["rules"][rule] = rule_cfg

    return cfg


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS filtered_images (
    image_id          INTEGER PRIMARY KEY REFERENCES images(id),
    pass_filter       INTEGER,
    fail_reasons      TEXT,
    snr_nucleus       REAL,
    laplacian_var     REAL,
    centroid_offset   REAL,
    nucleus_area_frac REAL,
    membrane_mean     REAL,
    marker_max_mean   REAL,
    filtered_at       TEXT
);
"""

INSERT_SQL = """
    INSERT OR REPLACE INTO filtered_images
        (image_id, pass_filter, fail_reasons,
         snr_nucleus, laplacian_var, centroid_offset,
         nucleus_area_frac, membrane_mean, marker_max_mean, filtered_at)
    VALUES
        (:image_id, :pass_filter, :fail_reasons,
         :snr_nucleus, :laplacian_var, :centroid_offset,
         :nucleus_area_frac, :membrane_mean, :marker_max_mean, :filtered_at)
"""


def init_filter_table(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def fetch_image_rows(db_path: Path, resume: bool) -> list:
    """
    Return list of (id, file_path, case_name, macrophage_type).
    Ordered by case_name, macrophage_type so directory-local rows are contiguous.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        order = "ORDER BY case_name, macrophage_type, id"
        if resume:
            rows = conn.execute(
                f"""
                SELECT id, file_path, case_name, macrophage_type FROM images
                WHERE id NOT IN (SELECT image_id FROM filtered_images)
                {order}
                """
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT id, file_path, case_name, macrophage_type FROM images {order}"
            ).fetchall()
        return rows
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Directory-based work distribution
# ---------------------------------------------------------------------------

def distribute_by_directory(rows: list, num_workers: int) -> list:
    """
    Group rows by (case_name, macrophage_type), then assign whole groups to
    workers round-robin.

    Result: list of worker_row_lists. No directory is split across workers,
    so workers never compete for the same NTFS directory entries.

    Load balancing: groups are sorted largest-first before assignment (LPT
    heuristic) to keep worker finish times roughly equal.
    """
    groups = defaultdict(list)
    for row in rows:
        key = (row[2], row[3])  # (case_name, macrophage_type)
        groups[key].append(row)

    # Largest groups first → better load balance across workers
    sorted_groups = sorted(groups.values(), key=len, reverse=True)

    actual_workers = min(num_workers, len(sorted_groups))
    assignments = [[] for _ in range(actual_workers)]

    # Round-robin: assign each group entirely to one worker
    for i, group in enumerate(sorted_groups):
        assignments[i % actual_workers].extend(group)

    return assignments


# ---------------------------------------------------------------------------
# Dedicated writer thread
# ---------------------------------------------------------------------------

def writer_thread_fn(db_path: Path, result_queue: queue.Queue, batch_size: int,
                     stats: dict) -> None:
    """
    Single persistent SQLite connection. Drains result_queue and bulk-inserts.
    Terminates on receiving None sentinel.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA cache_size=-32000;")

    buf = []
    try:
        while True:
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                if buf:
                    conn.executemany(INSERT_SQL, buf)
                    conn.commit()
                    stats["written"] += len(buf)
                    buf.clear()
                continue

            if result is None:
                break

            buf.append(result)
            if result["pass_filter"] == 1:
                stats["passed"] += 1
            else:
                stats["failed"] += 1

            if len(buf) >= batch_size:
                conn.executemany(INSERT_SQL, buf)
                conn.commit()
                stats["written"] += len(buf)
                buf.clear()

        if buf:
            conn.executemany(INSERT_SQL, buf)
            conn.commit()
            stats["written"] += len(buf)

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Channel loading (selective)
# ---------------------------------------------------------------------------

def load_channel(pages, ch_idx: int) -> np.ndarray:
    """Load one channel as float32 normalised to [0, 1]."""
    raw = pages[ch_idx].asarray().astype(np.float32)
    mx = raw.max()
    if mx > 0:
        raw /= mx
    return raw


def open_tiff_pages(full_path: Path):
    """Open TiffFile, return (tif_handle, pages, num_channels). Caller must close."""
    tif = tifffile.TiffFile(str(full_path))
    series = tif.series[0]
    pages = series.pages
    return tif, pages, len(pages)


# ---------------------------------------------------------------------------
# Metric computation (pure functions, no I/O)
# ---------------------------------------------------------------------------

def compute_nucleus_snr(ch0: np.ndarray) -> float:
    try:
        thresh = threshold_otsu(ch0)
        fg = ch0[ch0 >= thresh]
        bg = ch0[ch0 < thresh]
        if fg.size == 0 or bg.size == 0:
            return 0.0
        bg_std = float(bg.std())
        if bg_std < 1e-9:
            bg_std = 1e-4
        return float(fg.mean()) / bg_std
    except Exception:
        return 0.0


def compute_laplacian_var(ch0: np.ndarray) -> float:
    img_u8 = (ch0 * 255).astype(np.uint8)
    lap = cv2.Laplacian(img_u8, cv2.CV_64F)
    return float(lap.var())


def compute_cell_completeness(ch0: np.ndarray):
    """
    Find the nucleus component closest to image centre.
    Returns (centroid_offset_normalised, nucleus_area_frac) or (None, None).
    """
    H, W = ch0.shape
    cx_img, cy_img = W / 2.0, H / 2.0
    half_diag = math.sqrt(cx_img ** 2 + cy_img ** 2)

    try:
        thresh = threshold_otsu(ch0)
    except Exception:
        return None, None

    binary = ch0 >= thresh
    if not binary.any():
        return None, None

    labeled = label(binary)
    props = regionprops(labeled)
    if not props:
        return None, None

    best = min(
        props,
        key=lambda p: math.sqrt((p.centroid[1] - cx_img) ** 2 + (p.centroid[0] - cy_img) ** 2),
    )

    dy = best.centroid[0] - cy_img
    dx = best.centroid[1] - cx_img
    offset = math.sqrt(dx ** 2 + dy ** 2) / half_diag
    area_frac = best.area / float(H * W)
    return float(offset), float(area_frac)


# ---------------------------------------------------------------------------
# Single-image processing (called by worker, returns result dict)
# ---------------------------------------------------------------------------

def _process_single_image(image_id: int, file_path_str: str, root_dir_str: str,
                           rules_cfg: dict) -> dict:
    """Compute all quality metrics for one image. Returns a result dict."""
    now = datetime.now().isoformat(timespec="seconds")
    full_path = Path(root_dir_str) / file_path_str

    result = {
        "image_id": image_id,
        "pass_filter": 1,
        "fail_reasons": "[]",
        "snr_nucleus": None,
        "laplacian_var": None,
        "centroid_offset": None,
        "nucleus_area_frac": None,
        "membrane_mean": None,
        "marker_max_mean": None,
        "filtered_at": now,
    }

    snr_cfg  = rules_cfg.get("nucleus_snr", {})
    focus_cfg = rules_cfg.get("focus", {})
    cc_cfg   = rules_cfg.get("cell_completeness", {})
    mem_cfg  = rules_cfg.get("membrane_signal", {})
    mk_cfg   = rules_cfg.get("marker_signal", {})

    need_membrane = mem_cfg.get("enabled", True)
    need_markers  = mk_cfg.get("enabled", True)

    fail_reasons = []
    tif = None
    try:
        tif, pages, C = open_tiff_pages(full_path)
        ch0 = load_channel(pages, 0)

        # Nucleus SNR
        snr = compute_nucleus_snr(ch0)
        result["snr_nucleus"] = snr
        if snr_cfg.get("enabled", True) and snr < snr_cfg.get("min_snr", 5.0):
            fail_reasons.append("low_nucleus_snr")

        # Focus
        lap_var = compute_laplacian_var(ch0)
        result["laplacian_var"] = lap_var
        if focus_cfg.get("enabled", True) and lap_var < focus_cfg.get("min_laplacian_var", 20.0):
            fail_reasons.append("blurry")

        # Cell completeness
        offset, area_frac = compute_cell_completeness(ch0)
        result["centroid_offset"] = offset
        result["nucleus_area_frac"] = area_frac
        if cc_cfg.get("enabled", True):
            if offset is None:
                fail_reasons.append("no_nucleus_detected")
            else:
                if offset > cc_cfg.get("max_centroid_border_frac", 0.30):
                    fail_reasons.append("no_center_cell")
                if area_frac is not None:
                    if area_frac < cc_cfg.get("min_nucleus_area_frac", 0.02):
                        fail_reasons.append("nucleus_too_small")
                    elif area_frac > cc_cfg.get("max_nucleus_area_frac", 0.50):
                        fail_reasons.append("nucleus_too_large")

        # Membrane (ch5)
        if C > 5 and need_membrane:
            ch5 = load_channel(pages, 5)
            mem_mean = float(ch5.mean())
            result["membrane_mean"] = mem_mean
            if mem_cfg.get("enabled", True) and mem_mean < mem_cfg.get("min_mean", 0.01):
                fail_reasons.append("no_membrane_signal")

        # Markers (ch1-4)
        if need_markers:
            marker_means = [
                float(load_channel(pages, ch).mean())
                for ch in range(1, 5) if ch < C
            ]
            if marker_means:
                marker_max = max(marker_means)
                result["marker_max_mean"] = marker_max
                if mk_cfg.get("enabled", True) and marker_max < mk_cfg.get("min_max_mean", 0.01):
                    fail_reasons.append("no_marker_signal")

    except Exception as e:
        fail_reasons = [f"load_error: {e}"]
    finally:
        if tif is not None:
            try:
                tif.close()
            except Exception:
                pass

    if fail_reasons:
        result["pass_filter"] = 0
    result["fail_reasons"] = json.dumps(fail_reasons)
    return result


# ---------------------------------------------------------------------------
# Worker: processes a list of rows belonging to exclusive directories
# ---------------------------------------------------------------------------

def process_worker_batch(row_list: list, root_str: str, rules_cfg: dict,
                         result_queue: queue.Queue) -> None:
    """
    Worker function assigned to one thread. Processes all images in row_list
    sequentially. Because every directory in row_list is exclusively owned by
    this worker, there is no NTFS directory-lock contention with other workers.
    """
    for row in row_list:
        image_id      = row[0]
        file_path_str = row[1]
        try:
            result = _process_single_image(image_id, file_path_str, root_str, rules_cfg)
        except Exception as e:
            now = datetime.now().isoformat(timespec="seconds")
            result = {
                "image_id": image_id,
                "pass_filter": 0,
                "fail_reasons": json.dumps([f"unhandled_error: {e}"]),
                "snr_nucleus": None, "laplacian_var": None,
                "centroid_offset": None, "nucleus_area_frac": None,
                "membrane_mean": None, "marker_max_mean": None,
                "filtered_at": now,
            }
        result_queue.put(result)  # blocks if queue is full (backpressure)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_filter(root_dir: Path, db_path: Path, rules_cfg: dict,
               workers: int, batch_size: int, resume: bool, verbose: bool = True) -> None:
    init_filter_table(db_path)
    rows = fetch_image_rows(db_path, resume=resume)
    total = len(rows)

    if total == 0:
        print("Nothing to do.")
        return

    # Partition rows by directory - each worker owns exclusive directories
    assignments = distribute_by_directory(rows, workers)
    actual_workers = len(assignments)

    if verbose:
        mode = "resume" if resume else "full"
        groups = defaultdict(list)
        for row in rows:
            groups[(row[2], row[3])].append(row)
        print(f"Images to process : {total:,} ({mode} mode)")
        print(f"Directories       : {len(groups)}  ({', '.join(f'{k[0]}/{k[1]}' for k in sorted(groups)[:6])}{'…' if len(groups) > 6 else ''})")
        print(f"Worker threads    : {actual_workers}  (requested {workers}, capped at {len(groups)} directories)")
        print(f"Writer batch size : {batch_size}")
        print(f"Root dir          : {root_dir}")
        print()
        for i, assignment in enumerate(assignments):
            assignment_dirs = set((row[2], row[3]) for row in assignment)
            print(f"  Worker {i+1:2d}: {len(assignment):>7,} images across {len(assignment_dirs)} dir(s)")
        print()

    root_str = str(root_dir)
    result_queue: queue.Queue = queue.Queue(maxsize=actual_workers * 8)
    stats = {"passed": 0, "failed": 0, "written": 0}

    writer = threading.Thread(
        target=writer_thread_fn,
        args=(db_path, result_queue, batch_size, stats),
        daemon=True,
    )
    writer.start()

    start = time.time()

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = [
            executor.submit(process_worker_batch, assignment, root_str, rules_cfg, result_queue)
            for assignment in assignments
        ]

        # Progress monitor: poll stats from writer thread
        done_count = 0
        while done_count < actual_workers:
            done_count = sum(1 for f in futures if f.done())
            if verbose:
                written = stats["written"]
                elapsed = time.time() - start
                rate = written / elapsed if elapsed > 0 else 0
                print(
                    f"  Written {written:,}/{total:,}  "
                    f"({stats['passed']:,} pass / {stats['failed']:,} fail)  "
                    f"@ {rate:.0f} img/s  "
                    f"[{done_count}/{actual_workers} workers done]",
                    end="\r",
                    flush=True,
                )
            time.sleep(2.0)

        # Collect any worker exceptions
        for f in futures:
            exc = f.exception()
            if exc:
                print(f"\n[WARN] Worker raised: {exc}")

    result_queue.put(None)
    writer.join()

    elapsed = time.time() - start
    rate = total / elapsed if elapsed > 0 else 0
    passed = stats["passed"]
    failed = stats["failed"]

    print(f"\nDone. {total:,} images in {elapsed:.1f}s ({rate:.0f} img/s).")
    if total:
        print(f"  Passed : {passed:,} ({100 * passed / total:.1f}%)")
        print(f"  Failed : {failed:,} ({100 * failed / total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Rule-based quality filter for OME-TIFF cell images."
    )
    parser.add_argument("--db", help="SQLite DB path (overrides config)")
    parser.add_argument("--config", help="Path to config.json")
    parser.add_argument("--workers", type=int, help="Number of worker threads (overrides config)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already in filtered_images table",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    root_dir = Path(cfg["project"]["root_dir"])
    if not root_dir.exists():
        raise SystemExit(f"Root directory does not exist: {root_dir}. Set it in config.json.")

    db_path_raw = Path(args.db) if args.db else Path(cfg["project"]["db_path"])
    db_path = (Path.cwd() / db_path_raw if not db_path_raw.is_absolute() else db_path_raw).resolve()

    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}. Run index_images.py first.")

    filter_cfg = cfg["filtering"]
    workers    = args.workers if args.workers else filter_cfg.get("workers", 16)
    batch_size = filter_cfg.get("batch_size", 500)
    rules_cfg  = filter_cfg.get("rules", {})

    run_filter(
        root_dir=root_dir,
        db_path=db_path,
        rules_cfg=rules_cfg,
        workers=workers,
        batch_size=batch_size,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
