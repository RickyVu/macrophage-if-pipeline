import argparse
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path


def iter_ome_tiffs(root_dir: Path):
    """
    Yield (relative_path, case, macrophage_type, filename) for each OME-TIFF.
    
    Uses os.walk() instead of rglob() for much better performance on large directory trees.

    Assumes structure: <root>/<case>/<macrophage_type>/.../<filename>.ome.tif[f]
    """
    root_str = str(root_dir.resolve())
    
    # Use os.walk() which is much faster than Path.rglob() for huge directory trees
    for dirpath, dirnames, filenames in os.walk(root_str):
        for filename in filenames:
            # Fast check: only process files ending in .ome.tif[f]
            if not (filename.lower().endswith(".ome.tiff") or filename.lower().endswith(".ome.tif")):
                continue
            
            # Build relative path
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root_str)
            
            # Parse path components
            parts = Path(rel_path).parts
            if len(parts) < 3:
                continue
            
            case = parts[0]
            macrophage_type = parts[1]
            rel_str = rel_path.replace(os.sep, "/")  # normalize to forward slashes
            
            yield rel_str, case, macrophage_type, filename


def init_db(db_path: Path, bulk_insert_mode: bool = False):
    """
    Initialize SQLite DB with an `images` table for indexing files.

    Schema:
      - id: INTEGER PRIMARY KEY AUTOINCREMENT
      - file_path: TEXT UNIQUE (relative path like "case/type/..../file.ome.tiff")
      - case_name: TEXT (renamed from 'case' because 'case' is a SQL reserved keyword)
      - macrophage_type: TEXT
      - filename: TEXT
      - created_at: TEXT (ISO timestamp when record was (last) created/updated)
    
    Args:
        bulk_insert_mode: If True, use faster settings (synchronous=OFF, no WAL) for bulk inserts
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        
        if bulk_insert_mode:
            # Faster settings for bulk inserts
            cur.execute("PRAGMA journal_mode=DELETE;")  # DELETE is faster than WAL for bulk inserts
            cur.execute("PRAGMA synchronous=OFF;")  # Fastest, but less safe (acceptable for indexing)
            cur.execute("PRAGMA cache_size=-64000;")  # 64MB cache
            cur.execute("PRAGMA temp_store=MEMORY;")
        else:
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                case_name TEXT,
                macrophage_type TEXT,
                filename TEXT,
                created_at TEXT
            )
            """
        )

        conn.commit()
    finally:
        conn.close()


def index_images(root_dir: Path, db_path: Path, batch_size: int = 10000, verbose: bool = True):
    """
    Walk the folder structure under `root_dir`, and populate / update the `images`
    table in `db_path` with (relative_path, case, macrophage_type, filename).
    
    Optimized for millions of files with:
    - Fast os.walk() traversal
    - Large batch sizes (default 10k)
    - Bulk insert optimizations (synchronous=OFF during insert)
    - INSERT OR IGNORE (faster than ON CONFLICT DO UPDATE)
    """
    root_dir = root_dir.resolve()
    # If db_path is relative, make it relative to root_dir
    if not db_path.is_absolute():
        db_path = root_dir / db_path
    db_path = db_path.resolve()

    if verbose:
        print(f"Root directory: {root_dir}")
        print(f"SQLite DB     : {db_path}")
        print(f"Batch size    : {batch_size}")
        print("Starting bulk indexing (this may take a while for millions of files)...")

    # Initialize DB with bulk insert optimizations
    init_db(db_path, bulk_insert_mode=True)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Re-apply fast settings (init_db already did this, but ensure they're set)
        cur.execute("PRAGMA journal_mode=DELETE;")
        cur.execute("PRAGMA synchronous=OFF;")
        cur.execute("PRAGMA cache_size=-64000;")
        cur.execute("PRAGMA temp_store=MEMORY;")

        # Use INSERT OR IGNORE instead of ON CONFLICT DO UPDATE - much faster
        # (We can add an update mode later if needed)
        insert_sql = """
            INSERT OR IGNORE INTO images (file_path, case_name, macrophage_type, filename, created_at)
            VALUES (?, ?, ?, ?, ?)
        """

        count = 0
        batch = []
        now = datetime.now().isoformat(timespec="seconds")
        last_report = 0
        import time
        start_time = time.time()

        for rel_path, case, mtype, filename in iter_ome_tiffs(root_dir):
            batch.append((rel_path, case, mtype, filename, now))
            count += 1

            if len(batch) >= batch_size:
                cur.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()
                
                if verbose:
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    print(f"Indexed {count:,} images ({rate:.0f} images/sec)...", end="\r", flush=True)
                last_report = count

        # Final batch
        if batch:
            cur.executemany(insert_sql, batch)
            conn.commit()
            batch.clear()

        # Re-enable safer settings
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        conn.commit()

        if verbose:
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            print(f"\nIndexed {count:,} images total in {elapsed:.1f}s ({rate:.0f} images/sec)")

    finally:
        conn.close()


def load_config(config_path: Path = None):
    """Load config from JSON file, with sensible defaults."""
    if config_path is None:
        # Look for config.json in the same directory as this script
        config_path = Path(__file__).parent / "config.json"
    
    defaults = {
        "project": {
            "root_dir": None,
            "db_path": "image_index.sqlite"
        },
        "indexing": {
            "batch_size": 10000
        }
    }
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Merge with defaults
        defaults.update(config)
        return defaults
    return defaults


def main():
    parser = argparse.ArgumentParser(
        description="Index OME-TIFF images under a root folder into a SQLite DB."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        help="Root directory containing case/macrophage_type/.../*.ome.tif[f] files (overrides config)",
    )
    parser.add_argument(
        "--db",
        help="Path to SQLite database file (overrides config)",
    )
    parser.add_argument(
        "--config",
        help="Path to config.json file (default: <script_dir>/config.json)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config) if args.config else None)
    
    # Use command-line args if provided, otherwise use config
    root_dir = Path(args.root_dir) if args.root_dir else Path(config["project"]["root_dir"])
    db_path = Path(args.db) if args.db else Path(config["project"]["db_path"])
    batch_size = config["indexing"]["batch_size"]
    
    if root_dir is None or not root_dir.exists():
        raise SystemExit(f"Root directory does not exist: {root_dir}. Set it in config.json or pass as argument.")

    index_images(root_dir, db_path, batch_size=batch_size, verbose=True)


if __name__ == "__main__":
    main()


