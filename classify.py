"""
classify.py - Apply the trained quality classifier to all filtered images.

Reads pre-computed metrics from `filtered_images`, runs inference with the
saved model, and writes predictions to the `classified_images` table.

No image re-loading is needed - all features come from the DB.

Usage:
    python classify.py [--db image_index.sqlite] [--model quality_model.pkl]
                       [--config config.json] [--resume]

    --resume    Skip images already present in classified_images
"""

import argparse
import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path = None) -> dict:
    defaults = {"project": {"db_path": "image_index.sqlite"}}
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        defaults.update(cfg)
    return defaults


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS classified_images (
    image_id         INTEGER PRIMARY KEY REFERENCES images(id),
    predicted_label  TEXT,
    confidence       REAL,
    model_version    TEXT,
    classified_at    TEXT
);
"""


def init_classify_table(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def fetch_candidates(db_path: Path, feature_names: list, resume: bool) -> list:
    """
    Return rows (image_id, macrophage_type, *features) for images that passed
    the filter and (if resume=True) haven't been classified yet.
    """
    col_list = ", ".join(f"f.{c}" for c in feature_names)
    where_resume = "AND i.id NOT IN (SELECT image_id FROM classified_images)" if resume else ""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT i.id, i.macrophage_type, {col_list}
            FROM images i
            JOIN filtered_images f ON f.image_id = i.id
            WHERE f.pass_filter = 1
            {where_resume}
            ORDER BY i.id
            """
        ).fetchall()
        return rows
    finally:
        conn.close()


def insert_predictions_batch(db_path: Path, records: list) -> None:
    sql = """
        INSERT OR REPLACE INTO classified_images
            (image_id, predicted_label, confidence, model_version, classified_at)
        VALUES (?, ?, ?, ?, ?)
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executemany(sql, records)
        conn.commit()
    finally:
        conn.close()


def print_summary(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT i.macrophage_type, c.predicted_label, COUNT(*)
            FROM classified_images c
            JOIN images i ON i.id = c.image_id
            GROUP BY i.macrophage_type, c.predicted_label
            ORDER BY i.macrophage_type, c.predicted_label
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return

    from collections import defaultdict
    by_type = defaultdict(dict)
    for mtype, label, cnt in rows:
        by_type[mtype][label] = cnt

    print("\nPrediction summary by macrophage type:")
    print("-" * 60)
    for mtype in sorted(by_type):
        lc = by_type[mtype]
        good = lc.get("good", 0)
        bad = lc.get("bad", 0)
        total = good + bad
        pct = 100 * good / total if total > 0 else 0
        print(f"  {mtype:<35} good={good:>6,}  bad={bad:>6,}  ({pct:.1f}% good)")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_classify(db_path: Path, model_path: Path, resume: bool, batch_size: int = 10000) -> None:
    import numpy as np

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    pipe = model_data["model"]
    feature_names = model_data["feature_names"]
    model_version = model_path.stem

    init_classify_table(db_path)

    rows = fetch_candidates(db_path, feature_names, resume=resume)
    total = len(rows)
    print(f"Images to classify : {total:,}")

    if total == 0:
        print("Nothing to do.")
        print_summary(db_path)
        return

    now = datetime.now().isoformat(timespec="seconds")
    processed = 0
    records_buf = []

    # Process in batches
    for start in range(0, total, batch_size):
        chunk = rows[start: start + batch_size]
        image_ids = [r[0] for r in chunk]
        X = np.array(
            [[float(v) if v is not None else 0.0 for v in r[2:]] for r in chunk],
            dtype=np.float32,
        )

        preds = pipe.predict(X)
        probas = pipe.predict_proba(X)  # shape (n, 2); class order: [bad=0, good=1]

        for i, (img_id, pred_int) in enumerate(zip(image_ids, preds)):
            label = "good" if pred_int == 1 else "bad"
            confidence = float(probas[i][pred_int])
            records_buf.append((img_id, label, confidence, model_version, now))

        insert_predictions_batch(db_path, records_buf)
        records_buf.clear()

        processed += len(chunk)
        print(f"  Classified {processed:,}/{total:,}", end="\r", flush=True)

    print(f"\nDone. Classified {processed:,} images.")
    print_summary(db_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply quality classifier to all filtered images."
    )
    parser.add_argument("--db", help="SQLite DB path (overrides config)")
    parser.add_argument("--config", help="Path to config.json")
    parser.add_argument(
        "--model",
        default="quality_model.pkl",
        help="Path to trained model pickle (default: quality_model.pkl)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already in classified_images",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    db_path_raw = Path(args.db) if args.db else Path(cfg["project"]["db_path"])
    db_path = (Path.cwd() / db_path_raw).resolve() if not db_path_raw.is_absolute() else db_path_raw.resolve()

    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model_path = model_path.resolve()

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}. Run train_classifier.py first.")

    run_classify(db_path=db_path, model_path=model_path, resume=args.resume)


if __name__ == "__main__":
    main()
