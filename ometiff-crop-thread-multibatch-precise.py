import os
import sys
import pathlib
import pandas as pd
import numpy as np
import shutil
import tifffile as tiff
from pyometiff import OMETIFFReader
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed





IS_TEST = False



# USER SETTINGS
# List of (OME-TIFF, CSV) pairs to process
if IS_TEST:
    DATASETS = [
        (r"D:\data\wsi\combined\Batch 2\SAMPLE_001_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_001_final_assignment.csv")
    ]
    TARGET_CELL_TYPES = {
        "PD1+ CD68+ Macrophage": (0, 5, 18, 19),
    }
else:
    DATASETS = [
        # Batch 1
        (r"D:\data\wsi\combined\Batch 1\SAMPLE_001_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_001_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 1\SAMPLE_002_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_002_final_assignment.csv"),
        #(r"D:\data\wsi\combined\Batch 1\SAMPLE_003_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_003_final_assignment.csv"),

        # Batch 2
        (r"D:\data\wsi\combined\Batch 2\SAMPLE_004_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_004_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 2\SAMPLE_005_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_005_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 2\SAMPLE_006_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_006_final_assignment.csv"),

        # Batch 3
        #(r"D:\data\wsi\combined\Batch 3\SAMPLE_007_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_007_final_assignment.csv"),
        #(r"D:\data\wsi\combined\Batch 3\SAMPLE_008_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_008_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 3\SAMPLE_009_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_009_final_assignment.csv"),
        #(r"D:\data\wsi\combined\Batch 3\SAMPLE_010_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_010_final_assignment.csv"),

        # Batch 4
        (r"D:\data\wsi\combined\Batch 4\SAMPLE_011_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_011_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 4\SAMPLE_011_R001.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_012_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 4\SAMPLE_013_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_013_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 4\SAMPLE_014_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_014_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 4\SAMPLE_015_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_015_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 4\SAMPLE_016_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_016_final_assignment.csv"),

        # Batch 5
        (r"D:\data\wsi\combined\Batch 5\SAMPLE_017_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_017_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 5\SAMPLE_018_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_018_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 5\SAMPLE_018_R001.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_019_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 5\SAMPLE_020_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_020_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 5\SAMPLE_021_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_021_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 5\SAMPLE_022_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_022_final_assignment.csv"),

        # Batch 6
        #(r"D:\data\wsi\combined\Batch 6\SAMPLE_023_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_023_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 6\SAMPLE_024_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_024_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 6\SAMPLE_025_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_025_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 6\SAMPLE_026_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_026_final_assignment.csv"),
        (r"D:\data\wsi\combined\Batch 6\SAMPLE_027_R000.ome.tif", r"D:\data\celesta\final_assignments\SAMPLE_027_final_assignment.csv"),
    ]

    TARGET_CELL_TYPES = {
        "PD1+ CD163+ Macrophage": (0, 5, 7, 12, 18, 19),#(0, 5, 7, 19),
        "PD-L1+ CD163+ Macrophage": (0, 5, 7, 12, 18, 19),#(0, 7, 12, 19),
        "PD1+ CD68+ Macrophage": (0, 5, 7, 12, 18, 19),#(0, 5, 18, 19),
        "PD-L1+ CD68+ Macrophage": (0, 5, 7, 12, 18, 19),#(0, 12, 18, 19),
    }
OUTPUT_ROOT = r"D:\data\output\cropped_cells"          # parent output directory
XSIZE = 50                       # crop width
YSIZE = 50                       # crop height



X_col = "X"
Y_col = "Y"

MAX_WORKERS = 24
BATCH_SIZE = 50


# INTERNAL GLOBALS

img_array = None
xs = ys = None

clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

# Utility functions

def zip_result(source_folder):
    shutil.make_archive(source_folder, 'zip', source_folder)
    print(f"Folder '{source_folder}' successfully zipped to '{source_folder}.zip'")

def combine_channels(img_arrs):
    return np.maximum.reduce(img_arrs) if img_arrs else None


def save_batch(batch):
    for filepath, arr in batch:
        tiff.imwrite(filepath, arr)


def flush_and_clear(buffer):
    if buffer:
        save_batch(buffer)
        buffer.clear()

def enhance_mif(img16):
    img = img16.astype("float32")
    background = cv2.GaussianBlur(img, (51, 51), 0)
    img = img - background
    img = np.clip(img, 0, None)

    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)
    img = (img - p1) / (p99 - p1 + 1e-6)
    img = np.clip(img, 0, 1)

    img8 = (img * 255).astype("uint8")
    img8 = clahe.apply(img8)

    return img8

def process_one(task):
    """
    Thread worker: crop channels + nuc channel.
    Returns two items to be batch-written.
    """
    global img_array, xs, ys

    xp, yp, channels, idx, outdir, stem, suffix = task

    y0 = int(yp - ys/2)
    y1 = int(yp + ys/2)
    x0 = int(xp - xs/2)
    x1 = int(xp + xs/2)

    # Crop each channel
    crop_array = [
        enhance_mif(img_array[y0:y1, x0:x1, ch])
        for ch in channels
    ]
    
    outfile = os.path.join(outdir, f"{stem}_{idx}{suffix}")
    
    #nuc_array = img_array[y0:y1, x0:x1, 0]

    #outfile_chan = os.path.join(outdir, f"{stem}_{idx}_chan{suffix}")
    #outfile_nuc  = os.path.join(outdir, f"{stem}_{idx}_nuc{suffix}")

    #return (outfile_chan, enhance_mif(combine_channels(crop_array))), \
    #       (outfile_nuc, enhance_mif(nuc_array))
    return outfile, crop_array



# PROCESS ONE DATASET

def process_dataset(ome_path, csv_path):
    global img_array, xs, ys
    xs, ys = XSIZE, YSIZE

    ome_path = pathlib.Path(ome_path)
    csv_path = pathlib.Path(csv_path)

    ome_name = ome_path.stem.replace(".ome", "")  # remove .ome if present

    # Create output folder for this dataset
    dataset_outdir = pathlib.Path(OUTPUT_ROOT) / ome_name
    dataset_outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================")
    print(f"PROCESSING DATASET: {ome_name}")
    print("OME:", ome_path)
    print("CSV:", csv_path)
    print("Output folder:", dataset_outdir)

    df = pd.read_csv(csv_path)

    # Check columns
    req = {X_col, Y_col, "Final_cell_type"}
    if not req.issubset(df.columns):
        print(f"CSV missing required columns: {req}")
        return

    # Filter target types
    df = df[df["Final_cell_type"].isin(TARGET_CELL_TYPES)]

    if df.empty:
        print("  No rows match TARGET_CELL_TYPES; skipping dataset.")
        return

    # Create cell-type subfolders        
    cell_types = TARGET_CELL_TYPES.keys()      #sorted(df["Final_cell_type"].unique()) 
    type_to_dir = {}

    for ct in cell_types:
        safe = ct.replace(" ", "_").replace("+", "p")
        folder = dataset_outdir / safe
        folder.mkdir(exist_ok=True)
        type_to_dir[ct] = str(folder)

    print("Processing cell types:")
    for ct in cell_types:
        print("  -", ct)

    # Load OME into RAM
    print("  Loading OME-TIFF...")
    reader = OMETIFFReader(fpath=str(ome_path))
    img_array, metadata, xml_metadata = reader.read()
    img_array = np.asarray(img_array)

    # Enforce XYC order
    if img_array.ndim == 3:
        if img_array.shape[0] < img_array.shape[1] and img_array.shape[0] < img_array.shape[2]:
            img_array = np.transpose(img_array, (1, 2, 0))
        elif img_array.shape[1] < img_array.shape[0] and img_array.shape[1] < img_array.shape[2]:
            img_array = np.transpose(img_array, (0, 2, 1))

    img_h, img_w, _ = img_array.shape

    # Build tasks
    tasks = []
    skipped = 0

    for i, row in df.iterrows():
        ct = row["Final_cell_type"]
        if ct in TARGET_CELL_TYPES:
            xp = row[X_col]
            yp = row[Y_col]

            y0 = int(yp - ys/2)
            y1 = int(yp + ys/2)
            x0 = int(xp - xs/2)
            x1 = int(xp + xs/2)

            if x0 < 0 or y0 < 0 or x1 > img_w or y1 > img_h:
                skipped += 1
                continue
            outdir = type_to_dir[ct]
            stem = ct.replace(" ", "_").replace("+", "p")
            channels = TARGET_CELL_TYPES[ct]

            tasks.append((xp, yp, channels, i, outdir, stem, ".ome.tiff"))

    print(f"  Valid crops: {len(tasks)} (Skipped: {skipped})")

    if not tasks:
        print("  No valid tasks; nothing to do.")
        return

    # Threaded + batched saving
    print(f"  Cropping with {MAX_WORKERS} workers...")

    buffers = [[] for _ in range(MAX_WORKERS)]
    futures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        for idx, task in enumerate(tasks):
            fut = exe.submit(process_one, task)
            futures.append(fut)

        for i, fut in enumerate(as_completed(futures)):
            #chan_item, nuc_item = fut.result()
            chan_item = fut.result()
            wid = i % MAX_WORKERS
            buffers[wid].append(chan_item)
            #buffers[wid].append(nuc_item)

            if len(buffers[wid]) >= BATCH_SIZE:
                flush_and_clear(buffers[wid])

    # Final flush
    for buf in buffers:
        flush_and_clear(buf)

    print(f"  Finished dataset: {ome_name}")



# MAIN ENTRY POINT

if __name__ == "__main__":
    if not DATASETS:
        print("ERROR: DATASETS list is empty. Edit the script and add your OME + CSV pairs.")
        sys.exit(1)

    OUTPUT_ROOT = pathlib.Path(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(exist_ok=True)

    for ome, csv in DATASETS:
        process_dataset(ome, csv)

    print("\nALL DATASETS COMPLETED.")
    
    zip_result(OUTPUT_ROOT)
