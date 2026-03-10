"""
inspect.py - Manual image quality annotation tool.

Presents images that passed the rule-based filter one at a time for human
review. Each image is labelled "good", "bad", or "skip". Labels are saved
to the `inspected_images` table in the SQLite database immediately after
each keypress so progress is never lost.

Images are drawn in a stratified random order (balanced across the 4
macrophage types) so that each type receives roughly equal annotation
effort. The session ends when the target count of labelled images per type
is reached, or when the user presses Q to quit.

Controls:
    G          - label "good" and advance
    B          - label "bad" and advance
    S          - skip (no label stored) and advance
    LEFT / P   - go back one image (label is NOT changed)
    RIGHT / N  - advance without labelling
    1-5        - switch viewing mode (same as viewer.py)
    Q          - quit (all labels already saved)

Usage:
    python inspect.py [--target 1000] [--db image_index.sqlite] [--config config.json]
"""

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Reuse image-loading and visualisation from viewer.py
from viewer import (
    ensure_channels_first,
    load_config,
    make_all_mosaic,
    make_channels_grid,
    make_if_nuc_mem_mac_merged,
    make_if_nuc_mem_rgb,
    make_if_style_composite,
    normalize_channels,
    read_ome_tiff_file,
    resize_for_display,
)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS inspected_images (
    image_id     INTEGER PRIMARY KEY REFERENCES images(id),
    label        TEXT,
    inspected_at TEXT
);
"""


def init_inspect_table(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def fetch_candidates(db_path: Path) -> list:
    """
    Return rows for images that passed the filter and have not yet been labelled.
    Each row: (id, file_path, case_name, macrophage_type)
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT i.id, i.file_path, i.case_name, i.macrophage_type
            FROM images i
            JOIN filtered_images f ON i.id = f.image_id
            WHERE f.pass_filter = 1
              AND i.id NOT IN (SELECT image_id FROM inspected_images)
            ORDER BY i.macrophage_type, RANDOM()
            """
        ).fetchall()
        return rows
    finally:
        conn.close()


def fetch_label_counts(db_path: Path) -> dict:
    """Return {macrophage_type: {"good": N, "bad": N}} from inspected_images."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT i.macrophage_type, ii.label, COUNT(*)
            FROM inspected_images ii
            JOIN images i ON i.id = ii.image_id
            WHERE ii.label IN ('good', 'bad')
            GROUP BY i.macrophage_type, ii.label
            """
        ).fetchall()
        counts = defaultdict(lambda: {"good": 0, "bad": 0})
        for mtype, label, cnt in rows:
            counts[mtype][label] = cnt
        return dict(counts)
    finally:
        conn.close()


def save_label(db_path: Path, image_id: int, label: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            "INSERT OR REPLACE INTO inspected_images (image_id, label, inspected_at) VALUES (?, ?, ?)",
            (image_id, label, now),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def build_stratified_queue(candidates: list, target_per_type: int, label_counts: dict) -> list:
    """
    Build an interleaved queue from candidates, aiming to balance across types.
    Types that already have more labels are deprioritised.
    """
    by_type = defaultdict(list)
    for row in candidates:
        by_type[row[3]].append(row)  # row[3] = macrophage_type
    for lst in by_type.values():
        random.shuffle(lst)

    queue = []
    # Round-robin across types until each type hits target or runs out
    types = list(by_type.keys())
    while types:
        next_types = []
        for mtype in types:
            lc = label_counts.get(mtype, {"good": 0, "bad": 0})
            labelled = lc["good"] + lc["bad"]
            if labelled >= target_per_type:
                continue  # already at target - skip but don't remove from types yet to allow overflow
            if by_type[mtype]:
                queue.append(by_type[mtype].pop(0))
                next_types.append(mtype)
        if not next_types:
            break
        types = next_types

    return queue


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

MODES = [
    ("IF nuc+mem only", make_if_nuc_mem_rgb),
    ("IF nuc+mem + macrophage merged", make_if_nuc_mem_mac_merged),
    ("IF full composite (per-marker)", make_if_style_composite),
    ("single-channel grid (colored)", make_channels_grid),
    ("ALL mosaic (1+2+3+4)", make_all_mosaic),
]


def build_ui(rgb_image: np.ndarray, row: tuple, mode_name: str,
             queue_idx: int, queue_total: int, label_counts: dict,
             target: int) -> np.ndarray:
    h, w = rgb_image.shape[:2]

    panel_h = 110
    ui_w = max(w, 900)
    ui = np.zeros((h + panel_h, ui_w, 3), dtype=np.uint8)
    ui[:] = (30, 30, 30)

    x_off = (ui_w - w) // 2
    ui[0:h, x_off: x_off + w] = rgb_image

    font = cv2.FONT_HERSHEY_SIMPLEX
    y0 = h + 6

    # Line 1: case / type
    img_id, _, case_name, mtype = row
    cv2.putText(
        ui,
        f"Case: {case_name}   Type: {mtype}   ID: {img_id}",
        (10, y0 + 16), font, 0.45, (200, 200, 200), 1,
    )

    # Line 2: view mode + queue position
    cv2.putText(
        ui,
        f"View: {mode_name}   [{queue_idx + 1}/{queue_total}]",
        (10, y0 + 34), font, 0.45, (180, 180, 255), 1,
    )

    # Line 3: per-type label progress
    types_sorted = sorted(label_counts.keys())
    progress_parts = []
    for t in types_sorted:
        lc = label_counts.get(t, {"good": 0, "bad": 0})
        total_lc = lc["good"] + lc["bad"]
        short = t[:12]
        progress_parts.append(f"{short}: {total_lc}/{target}")
    cv2.putText(
        ui,
        "  |  ".join(progress_parts) if progress_parts else "No labels yet",
        (10, y0 + 52), font, 0.38, (100, 200, 100), 1,
    )

    # Line 4: controls
    cv2.putText(
        ui,
        "G=good  B=bad  S=skip  LEFT/P=back  RIGHT/N=fwd  1-5=view  Q=quit",
        (10, y0 + 70), font, 0.40, (150, 150, 255), 1,
    )

    # Line 5: keyboard reminder for current type target
    lc_cur = label_counts.get(mtype, {"good": 0, "bad": 0})
    cv2.putText(
        ui,
        f"Current type labelled: {lc_cur['good']+lc_cur['bad']} / {target}  "
        f"(good={lc_cur['good']}, bad={lc_cur['bad']})",
        (10, y0 + 88), font, 0.40, (200, 200, 100), 1,
    )

    return ui


# ---------------------------------------------------------------------------
# Main viewer loop
# ---------------------------------------------------------------------------

def run_inspect(root_dir: Path, db_path: Path, target: int, max_upscale: int) -> None:
    init_inspect_table(db_path)

    candidates = fetch_candidates(db_path)
    if not candidates:
        print("No candidates found. Make sure filter.py has been run and images have passed.")
        return

    label_counts = fetch_label_counts(db_path)
    queue = build_stratified_queue(candidates, target_per_type=target, label_counts=label_counts)

    if not queue:
        print("All target labels already reached for all types. Nothing to do.")
        return

    print(f"Queue size : {len(queue):,}")
    print(f"Target     : {target} good+bad labels per cell type")
    print()

    current_idx = 0
    current_mode = 0
    window_name = "Inspect: G=good  B=bad  S=skip  Q=quit"
    window_created = False
    last_win_size = None
    img_cache = {}  # image_id → (img_norm, error_str)

    def load_cached(row):
        image_id, file_path_str, _, _ = row
        if image_id in img_cache:
            return img_cache[image_id]
        try:
            full_path = root_dir / file_path_str
            raw = read_ome_tiff_file(full_path)
            img_c = ensure_channels_first(raw)
            img_norm = normalize_channels(img_c)
            img_cache[image_id] = (img_norm, None)
        except Exception as e:
            img_cache[image_id] = (None, str(e))
        # Keep cache small
        if len(img_cache) > 20:
            oldest = next(iter(img_cache))
            del img_cache[oldest]
        return img_cache[image_id]

    while 0 <= current_idx < len(queue):
        row = queue[current_idx]
        image_id = row[0]

        img_norm, err = load_cached(row)
        if img_norm is None:
            print(f"  [WARN] Could not load image {image_id}: {err} - skipping")
            current_idx += 1
            continue

        mode_name, mode_fn = MODES[current_mode]
        view_float = mode_fn(img_norm)
        rgb = (np.clip(view_float, 0, 1) * 255).astype(np.uint8)
        rgb = resize_for_display(rgb, max_upscale=max_upscale)

        label_counts = fetch_label_counts(db_path)
        ui = build_ui(rgb, row, mode_name, current_idx, len(queue), label_counts, target)

        h, w = ui.shape[:2]
        if not window_created:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            window_created = True
        if last_win_size != (w, h):
            cv2.resizeWindow(window_name, w, h)
            last_win_size = (w, h)

        cv2.imshow(window_name, ui)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q")):
            print("\nQuitting - all labels saved.")
            break
        elif key in (ord("g"), ord("G")):
            save_label(db_path, image_id, "good")
            current_idx += 1
        elif key in (ord("b"), ord("B")):
            save_label(db_path, image_id, "bad")
            current_idx += 1
        elif key in (ord("s"), ord("S")):
            current_idx += 1  # skip: no label written
        elif key in (ord("n"), ord("N"), ord(" "), 83):  # 83 = right arrow
            current_idx += 1
        elif key in (ord("p"), ord("P"), 81):  # 81 = left arrow
            current_idx = max(0, current_idx - 1)
        elif ord("1") <= key <= ord("5"):
            current_mode = key - ord("1")
        # else: any other key → redraw same image

    cv2.destroyAllWindows()

    # Final summary
    label_counts = fetch_label_counts(db_path)
    print("\nFinal label counts:")
    for mtype in sorted(label_counts):
        lc = label_counts[mtype]
        print(f"  {mtype}: good={lc['good']}, bad={lc['bad']}, total={lc['good']+lc['bad']}")


def main():
    parser = argparse.ArgumentParser(
        description="Manual quality annotation tool for macrophage cell images."
    )
    parser.add_argument("--target", type=int, default=1000,
                        help="Target number of good+bad labels per macrophage type (default: 1000)")
    parser.add_argument("--db", help="SQLite DB path (overrides config)")
    parser.add_argument("--config", help="Path to config.json")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    root_dir = Path(cfg["project"]["root_dir"])
    if not root_dir.exists():
        raise SystemExit(f"Root directory does not exist: {root_dir}. Set it in config.json.")

    db_path_raw = Path(args.db) if args.db else Path(cfg["project"]["db_path"])
    if not db_path_raw.is_absolute():
        db_path = Path.cwd() / db_path_raw
    else:
        db_path = db_path_raw
    db_path = db_path.resolve()

    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}. Run index_images.py and filter.py first.")

    max_upscale = cfg.get("viewer", {}).get("max_upscale", 3)
    run_inspect(root_dir=root_dir, db_path=db_path, target=args.target, max_upscale=max_upscale)


if __name__ == "__main__":
    main()
