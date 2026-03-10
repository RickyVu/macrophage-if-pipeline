import zipfile
import json
import numpy as np
import cv2
from pathlib import Path
import tifffile
import io
import os
import random

def get_sample_images(root_dir):
    """Get ONE sample image from each case and macrophage type on disk."""
    samples = []
    
    print("Scanning folder for sample images (one per case/type)...")
    
    root = Path(root_dir)
    structure = {}
    
    # Walk the extracted folder and find OME-TIFF files
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        rel_str = str(rel)
        if not rel_str.endswith((".ome.tiff", ".ome.tif")):
            continue
        
        parts = rel.parts
        if len(parts) < 3:
            continue
        
        case = parts[0]
        mtype = parts[1]
        
        if case not in structure:
            structure[case] = {}
        if mtype not in structure[case]:
            structure[case][mtype] = []
        
        structure[case][mtype].append(rel_str)
    
    # Pick ONE random image from each combination
    for case, types in structure.items():
        for mtype, files in types.items():
            if files:
                selected_file = random.choice(files)
                samples.append({
                    'case': case,
                    'macrophage_type': mtype,
                    'file_path': selected_file,
                    'filename': Path(selected_file).name
                })
    
    return samples, structure


def read_ome_tiff_from_zip(root_dir, file_path):
    """Read OME-TIFF from extracted folder (kept name for compatibility)."""
    full_path = Path(root_dir) / file_path
    if not full_path.exists():
        raise FileNotFoundError(f"Image not found on disk: {full_path}")
    
    with tifffile.TiffFile(str(full_path)) as tif:
        series = tif.series[0]
        
        if len(series.pages) > 1:
            pages = []
            for page in series.pages:
                pages.append(page.asarray())
            image = np.stack(pages, axis=0)
        else:
            image = series.asarray()
        
        return image


def read_ome_tiff_file(file_path):
    """Read OME-TIFF from a direct file path."""
    full_path = Path(file_path)
    if not full_path.exists():
        raise FileNotFoundError(f"Image not found: {full_path}")
    
    with tifffile.TiffFile(str(full_path)) as tif:
        series = tif.series[0]
        
        if len(series.pages) > 1:
            pages = []
            for page in series.pages:
                pages.append(page.asarray())
            image = np.stack(pages, axis=0)
        else:
            image = series.asarray()
        
        return image

def create_rgb_from_channels(image, r_channel=0, g_channel=5, b_channels=[1,2,3]):
    """[Deprecated] Old flexible channel-mapping RGB compositor (kept for reference)."""
    if image is None:
        return None
    
    # Handle different array shapes
    if len(image.shape) == 3:
        if image.shape[0] <= 10:  # Assuming channels first (C, H, W)
            image = np.transpose(image, (1, 2, 0))
    
    h, w, c = image.shape
    
    # Initialize RGB array
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    # Process each channel with better normalization
    def normalize_channel(data):
        if data.max() == 0:
            return data
        # Use percentile for better contrast
        p_low, p_high = np.percentile(data[data > 0], [2, 98]) if np.any(data > 0) else (0, data.max())
        if p_high > p_low:
            return np.clip((data - p_low) / (p_high - p_low) * 255, 0, 255)
        return data / data.max() * 255 if data.max() > 0 else data
    
    # Red channel
    if r_channel < c:
        r_data = image[:, :, r_channel].astype(np.float32)
        rgb[:, :, 2] = normalize_channel(r_data)
    
    # Green channel
    if g_channel < c:
        g_data = image[:, :, g_channel].astype(np.float32)
        rgb[:, :, 1] = normalize_channel(g_data)
    
    # Blue channel - average of specified channels
    if b_channels:
        b_data = np.zeros((h, w), dtype=np.float32)
        count = 0
        for ch in b_channels:
            if ch < c:
                b_data += image[:, :, ch].astype(np.float32)
                count += 1
        if count > 0:
            b_data = b_data / count
            rgb[:, :, 0] = normalize_channel(b_data)
    
    return np.clip(rgb, 0, 255).astype(np.uint8)

def load_config(config_path: Path = None):
    """Load config from JSON file, with sensible defaults."""
    if config_path is None:
        # Look for config.json in the same directory as this script
        config_path = Path(__file__).parent / "config.json"
    
    defaults = {
        "project": {
            "root_dir": None
        },
        "viewer": {
            "window_name": "Macrophage Channel Viewer",
            "max_upscale": 3
        }
    }
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Merge with defaults
        for key in defaults:
            if key in config:
                defaults[key].update(config[key])
        return defaults
    return defaults


def resize_for_display(image, max_width=1100, max_height=700, max_upscale=1.5):
    """Resize image to fit nicely in the UI while preserving aspect ratio.

    We only allow *limited* upscaling (up to `max_upscale` times) to avoid
    over-interpolation that could distort visual appearance.
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # Compute scale to fit within max_width/max_height
    scale = min(max_width / w, max_height / h)

    # Clamp upscaling to avoid deviating too far from native resolution
    if scale > 1.0:
        scale = min(scale, max_upscale)
    elif scale <= 0:
        # Fallback in pathological cases
        scale = 1.0

    # If scale is effectively 1, keep original size
    if abs(scale - 1.0) < 1e-3:
        return image
    
    # Otherwise resize (upscale small images, downscale huge ones)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Choose interpolation based on scaling direction
    interp = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
    
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def ensure_channels_first(image):
    """Ensure image is in CxHxW format."""
    if image is None:
        return None
    
    if image.ndim == 2:
        # Single channel HxW → 1xHxW
        return image[np.newaxis, ...]
    
    if image.ndim == 3:
        # Heuristic: small first dim → channels-first, small last dim → channels-last
        if image.shape[0] <= 10:
            return image  # already CxHxW
        if image.shape[-1] <= 10:
            # HxWxC → CxHxW
            return np.transpose(image, (2, 0, 1))
    
    # Fallback: leave as-is
    return image


def normalize_channels(img_c):
    """Normalize each channel to [0, 1] like in sample_tiff_utils.py."""
    img = img_c.astype(float)
    C = img.shape[0]
    for i in range(C):
        maxv = img[i].max()
        if maxv > 0:
            img[i] /= maxv
    return img


# Channel → (marker_name, (R, G, B) in [0,1]) using common IF-style colors
CHANNEL_MAP = {
    0: ("DAPI (nucleus)",          (0.0, 0.0, 1.0)),  # blue
    1: ("PD-1",                    (1.0, 0.0, 1.0)),  # magenta
    2: ("CD-163",                  (0.0, 1.0, 0.0)),  # green
    3: ("PD-L1",                   (1.0, 1.0, 0.0)),  # yellow
    4: ("CD-68",                   (0.0, 1.0, 1.0)),  # cyan
    5: ("NaKATPase (membrane)",    (1.0, 0.0, 0.0)),  # red
}


def make_if_style_composite(img_norm):
    """
    Build an immunofluorescence-style RGB composite from normalized channels [0,1].

    Channel semantics:
      0: DAPI (nucleus)       → blue
      1: PD-1                 → magenta
      2: CD-163               → green
      3: PD-L1                → yellow
      4: CD-68                → cyan
      5: NaKATPase (membrane) → red
    """
    C, H, W = img_norm.shape
    rgb = np.zeros((H, W, 3), dtype=float)

    for ch in range(C):
        if ch not in CHANNEL_MAP:
            continue
        _, (r, g, b) = CHANNEL_MAP[ch]
        rgb[:, :, 0] += img_norm[ch] * r
        rgb[:, :, 1] += img_norm[ch] * g
        rgb[:, :, 2] += img_norm[ch] * b

    return np.clip(rgb, 0, 1)


def make_if_nuc_mem_rgb(img_norm):
    """IF-style: nucleus (DAPI, blue) + membrane (NaKATPase, red) only."""
    C, H, W = img_norm.shape
    rgb = np.zeros((H, W, 3), dtype=float)

    # Nucleus (channel 0) → blue
    if C > 0 and 0 in CHANNEL_MAP:
        _, (r, g, b) = CHANNEL_MAP[0]
        rgb[:, :, 0] += img_norm[0] * r
        rgb[:, :, 1] += img_norm[0] * g
        rgb[:, :, 2] += img_norm[0] * b

    # Membrane (channel 5) → red
    if C > 5 and 5 in CHANNEL_MAP:
        _, (r, g, b) = CHANNEL_MAP[5]
        rgb[:, :, 0] += img_norm[5] * r
        rgb[:, :, 1] += img_norm[5] * g
        rgb[:, :, 2] += img_norm[5] * b

    return np.clip(rgb, 0, 1)


def make_if_nuc_mem_mac_merged(img_norm):
    """
    IF-style simplified: nucleus + membrane + all 4 macrophage markers merged
    into a single color (green) to reduce visual clutter.
    """
    C, H, W = img_norm.shape
    rgb = np.zeros((H, W, 3), dtype=float)

    # Base: nucleus + membrane using IF colors
    rgb += make_if_nuc_mem_rgb(img_norm)

    # Merge macrophage markers (channels 1-4) into one green channel
    mac = np.zeros((H, W), dtype=float)
    count = 0
    for ch in [1, 2, 3, 4]:
        if ch < C:
            mac += img_norm[ch]
            count += 1
    if count > 0:
        mac /= count
        mac = np.clip(mac, 0, 1)
        # Use green for merged macrophage signal
        rgb[:, :, 1] += mac

    return np.clip(rgb, 0, 1)


def make_all_chan_rgb(img_norm):
    """all_chan style composite (requires at least 6 channels)."""
    C, H, W = img_norm.shape
    rgb = np.zeros((H, W, 3), dtype=float)
    
    # chan0 → Green
    rgb[:, :, 1] = img_norm[0]
    
    # chan1-4 merged → Blue
    if C >= 5:
        rgb[:, :, 2] = (img_norm[1] + img_norm[2] + img_norm[3] + img_norm[4]) / 4
    
    # chan5 (if present) → Red
    if C >= 6:
        rgb[:, :, 0] = img_norm[5]
    
    return np.clip(rgb, 0, 1)


def make_channels_grid(img_norm):
    """Grid of individual channels as colored images using IF mapping."""
    C, H, W = img_norm.shape
    cols = min(C, 3)
    rows = (C + cols - 1) // cols
    
    grid_h = rows * H
    grid_w = cols * W
    grid = np.zeros((grid_h, grid_w, 3), dtype=float)
    
    for idx in range(C):
        r = idx // cols
        c = idx % cols
        ch = img_norm[idx]
        y0, y1 = r * H, (r + 1) * H
        x0, x1 = c * W, (c + 1) * W

        # Color this channel using IF mapping if available; otherwise grayscale
        tile = np.zeros((H, W, 3), dtype=float)
        if idx in CHANNEL_MAP:
            _, (cr, cg, cb) = CHANNEL_MAP[idx]
            tile[:, :, 0] = ch * cr
            tile[:, :, 1] = ch * cg
            tile[:, :, 2] = ch * cb
        else:
            tile[:, :, :] = ch[..., None]

        grid[y0:y1, x0:x1, :] = tile
    
    return np.clip(grid, 0, 1)


def make_single_channel_tile(img_norm, ch_index):
    """Create a single-channel HxWx3 tile colored using IF mapping for one channel."""
    C, H, W = img_norm.shape
    tile = np.zeros((H, W, 3), dtype=float)
    if ch_index >= C:
        return tile

    ch = img_norm[ch_index]

    if ch_index in CHANNEL_MAP:
        _, (cr, cg, cb) = CHANNEL_MAP[ch_index]
        tile[:, :, 0] = ch * cr
        tile[:, :, 1] = ch * cg
        tile[:, :, 2] = ch * cb
    else:
        tile[:, :, :] = ch[..., None]

    return np.clip(tile, 0, 1)


def make_all_mosaic(img_norm):
    """'ALL' view: 3x3 layout combining IF-style composites and single channels.

    Layout (rowsxcols = 3x3):
      Row 0: [ IF nuc+mem , IF nuc+mem+mac merged , IF full composite ]
      Row 1: [ ch0 , ch1 , ch2 ]  (colored single-channel tiles)
      Row 2: [ ch3 , ch4 , ch5 ]
    """
    # Base composites (all HxW)
    nm = make_if_nuc_mem_rgb(img_norm)
    nm_mac = make_if_nuc_mem_mac_merged(img_norm)
    if_full = make_if_style_composite(img_norm)

    H, W, _ = nm.shape

    # Prepare 3x3 grid canvas
    mosaic = np.zeros((3 * H, 3 * W, 3), dtype=float)

    # Top row: three composite views
    mosaic[0:H, 0:W, :] = nm
    mosaic[0:H, W:2*W, :] = nm_mac
    mosaic[0:H, 2*W:3*W, :] = if_full

    # Bottom 2 rows: six single-channel tiles (0-5)
    # Row 1
    for c in range(3):
        tile = make_single_channel_tile(img_norm, c)
        mosaic[H:2*H, c*W:(c+1)*W, :] = tile

    # Row 2
    for c in range(3):
        ch_idx = 3 + c
        tile = make_single_channel_tile(img_norm, ch_idx)
        mosaic[2*H:3*H, c*W:(c+1)*W, :] = tile

    return np.clip(mosaic, 0, 1)

def create_ui_frame(image, info, mode_name, current_idx, total_idx):
    """Create a proper UI with image and control panel.

    Channel semantics (IF-style):
      - 0: DAPI (nucleus)
      - 1: PD-1
      - 2: CD-163
      - 3: PD-L1
      - 4: CD-68
      - 5: NaKATPase (membrane)
    """
    h, w = image.shape[:2]
    
    # Create a larger canvas for UI (image + control panel)
    ui_height = h + 120  # Add space for controls at bottom
    ui_width = max(w, 800)  # Minimum width for controls
    
    ui = np.zeros((ui_height, ui_width, 3), dtype=np.uint8)
    ui[:] = (40, 40, 40)  # Dark gray background
    
    # Place image in the center top
    x_offset = (ui_width - w) // 2
    ui[20:20+h, x_offset:x_offset+w] = image
    
    # Create control panel at bottom
    panel_y = h + 30
    panel_height = 70
    
    # Draw panel background
    cv2.rectangle(ui, (10, panel_y), (ui_width-10, panel_y+panel_height), (60, 60, 60), -1)
    cv2.rectangle(ui, (10, panel_y), (ui_width-10, panel_y+panel_height), (100, 100, 100), 2)
    
    # File info at top
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(ui, f"Case: {info['case']} | Type: {info['macrophage_type']}", 
                (20, 15), font, 0.5, (200, 200, 200), 1)
    cv2.putText(ui, f"Image: {current_idx}/{total_idx} | File: {info['filename'][:40]}", 
                (ui_width-400, 15), font, 0.5, (200, 200, 200), 1)
    
    # Current view / mode
    cv2.putText(ui, f"View: {mode_name}", (20, panel_y+25), font, 0.6, (255, 255, 255), 2)
    
    # Controls
    controls = [
        "Next: SPACE/N | Prev: B/P | Quit: Q",
        "1:nuc+mem  2:nuc+mem+mac  3:IF  4:single  5:ALL"
    ]
    cv2.putText(ui, controls[0], (20, panel_y+45), font, 0.45, (150, 150, 255), 1)
    cv2.putText(ui, controls[1], (20, panel_y+65), font, 0.45, (150, 150, 255), 1)
    
    return ui


def create_minimal_ui_frame(image, filename, mode_name):
    """Create a minimal UI frame for single-file viewer."""
    h, w = image.shape[:2]
    
    # Create a larger canvas for UI (image + control panel)
    ui_height = h + 100  # Add space for controls at bottom
    ui_width = max(w, 700)  # Minimum width for controls
    
    ui = np.zeros((ui_height, ui_width, 3), dtype=np.uint8)
    ui[:] = (40, 40, 40)  # Dark gray background
    
    # Place image in the center top
    x_offset = (ui_width - w) // 2
    ui[20:20+h, x_offset:x_offset+w] = image
    
    # Create control panel at bottom
    panel_y = h + 20
    panel_height = 60
    
    # Draw panel background
    cv2.rectangle(ui, (10, panel_y), (ui_width-10, panel_y+panel_height), (60, 60, 60), -1)
    cv2.rectangle(ui, (10, panel_y), (ui_width-10, panel_y+panel_height), (100, 100, 100), 2)
    
    # File info at top
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(ui, f"File: {Path(filename).name[:60]}", 
                (20, 15), font, 0.5, (200, 200, 200), 1)
    
    # Current view / mode
    cv2.putText(ui, f"View: {mode_name}", (20, panel_y+25), font, 0.6, (255, 255, 255), 2)
    
    # Controls
    controls = [
        "1:nuc+mem  2:nuc+mem+mac  3:IF  4:single  5:ALL  Q:Quit"
    ]
    cv2.putText(ui, controls[0], (20, panel_y+50), font, 0.45, (150, 150, 255), 1)
    
    return ui


def view_single_file(file_path, max_upscale=3):
    """Minimal viewer for a single image file.
    
    Args:
        file_path: Path to the OME-TIFF image file
        max_upscale: Maximum upscaling factor for display (default: 3)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")
    
    print(f"Loading: {file_path.name}")
    
    # Viewing modes
    modes = [
        ("IF nuc+mem only", make_if_nuc_mem_rgb),
        ("IF nuc+mem + macrophage merged", make_if_nuc_mem_mac_merged),
        ("IF full composite (per-marker)", make_if_style_composite),
        ("single-channel grid (colored)", make_channels_grid),
        ("ALL mosaic (1+2+3+4)", make_all_mosaic),
    ]
    current_mode = 0
    
    window_name = "Image Viewer"
    window_created = False
    last_window_size = None
    
    print("\nControls:")
    print("- 1: IF nucleus+membrane only")
    print("- 2: IF nuc+mem + macrophage merged (single-color)")
    print("- 3: IF full composite (per-marker colors)")
    print("- 4: single-channel grid (colored)")
    print("- 5: ALL mosaic (1+2+3+4 stacked)")
    print("- Q: Quit")
    
    try:
        # Load and normalize the image
        raw = read_ome_tiff_file(file_path)
        img_c = ensure_channels_first(raw)
        img_norm = normalize_channels(img_c)
        
        while True:
            mode_name, mode_fn = modes[current_mode]
            view_img = mode_fn(img_norm)  # float in [0, 1], HxWx3
            # Convert to uint8 and resize for display
            rgb_image = (np.clip(view_img, 0, 1) * 255).astype(np.uint8)
            rgb_image = resize_for_display(rgb_image, max_upscale=max_upscale)
            
            if rgb_image is None:
                print(f"Error: Could not process image")
                break
            
            # Create minimal UI
            ui = create_minimal_ui_frame(rgb_image, file_path.name, mode_name)
            
            h, w = ui.shape[:2]
            
            # Lazily create the window
            if not window_created:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                window_created = True
            
            # Always sync the window size to the current UI size to avoid stretching
            if last_window_size != (w, h):
                cv2.resizeWindow(window_name, w, h)
                last_window_size = (w, h)
            
            cv2.imshow(window_name, ui)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break
            # Mode switching
            elif key == ord('1'):
                current_mode = 0
                print("View: IF nuc+mem only")
            elif key == ord('2'):
                current_mode = 1
                print("View: IF nuc+mem + macrophage merged")
            elif key == ord('3'):
                current_mode = 2
                print("View: IF full composite (per-marker)")
            elif key == ord('4'):
                current_mode = 3
                print("View: channels grid")
            elif key == ord('5'):
                current_mode = 4
                print("View: ALL mosaic (1+2+3)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("\nDone!")

def main():
    # Load config
    config = load_config()
    root_dir = Path(config["project"]["root_dir"])
    window_name = config["viewer"]["window_name"]
    max_upscale = config["viewer"]["max_upscale"]
    
    if root_dir is None or not root_dir.exists():
        raise SystemExit(f"Root directory not set or does not exist. Set 'project.root_dir' in config.json")
    
    # Get sample images
    samples, full_structure = get_sample_images(root_dir)
    
    print(f"\nFound: {len(samples)} sample images ({len(full_structure)} cases)")
    
    if not samples:
        print("No images found!")
        return
    
    print("\nControls:")
    print("- SPACE/N: Next image")
    print("- B/P: Previous image")
    print("- 1: IF nucleus+membrane only")
    print("- 2: IF nuc+mem + macrophage merged (single-color)")
    print("- 3: IF full composite (per-marker colors)")
    print("- 4: single-channel grid (colored)")
    print("- 5: ALL mosaic (1+2+3+4 stacked)")
    print("- Q: Quit")

    current_idx = 0
    window_created = False
    last_window_size = None  # (w, h)

    # Viewing modes
    modes = [
        ("IF nuc+mem only", make_if_nuc_mem_rgb),
        ("IF nuc+mem + macrophage merged", make_if_nuc_mem_mac_merged),
        ("IF full composite (per-marker)", make_if_style_composite),
        ("single-channel grid (colored)", make_channels_grid),
        ("ALL mosaic (1+2+3+4)", make_all_mosaic),
    ]
    current_mode = 0
    
    while 0 <= current_idx < len(samples):
        img_info = samples[current_idx]
        
        print(f"\n[{current_idx + 1}/{len(samples)}] Loading: {img_info['case']} - {img_info['macrophage_type']}")
        
        try:
            # Load and normalize the current image from disk
            raw = read_ome_tiff_from_zip(root_dir, img_info['file_path'])
            img_c = ensure_channels_first(raw)
            img_norm = normalize_channels(img_c)
            
            mode_name, mode_fn = modes[current_mode]
            view_img = mode_fn(img_norm)  # float in [0, 1], HxWx3
            # Convert to uint8 and gently resize for better on-screen visibility
            rgb_image = (np.clip(view_img, 0, 1) * 255).astype(np.uint8)
            rgb_image = resize_for_display(rgb_image, max_upscale=max_upscale)
            
            if rgb_image is None:
                print(f"Error: Could not process image")
                current_idx += 1
                continue
            
            # Create UI
            ui = create_ui_frame(
                rgb_image, img_info,
                mode_name,
                current_idx + 1, len(samples)
            )

            h, w = ui.shape[:2]

            # Lazily create the window
            if not window_created:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                window_created = True

            # Always sync the window size to the current UI size to avoid stretching
            if last_window_size != (w, h):
                cv2.resizeWindow(window_name, w, h)
                last_window_size = (w, h)

            cv2.imshow(window_name, ui)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break
            elif key == ord(' ') or key == ord('n') or key == ord('N'):
                current_idx += 1
            elif key == ord('b') or key == ord('p') or key == ord('B') or key == ord('P'):
                current_idx = max(0, current_idx - 1)
            # Mode switching (does not advance image index)
            elif key == ord('1'):
                current_mode = 0
                print("View: IF nuc+mem only")
            elif key == ord('2'):
                current_mode = 1
                print("View: IF nuc+mem + macrophage merged")
            elif key == ord('3'):
                current_mode = 2
                print("View: IF full composite (per-marker)")
            elif key == ord('4'):
                current_mode = 3
                print("View: channels grid")
            elif key == ord('5'):
                current_mode = 4
                print("View: ALL mosaic (1+2+3)")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            current_idx += 1
    
    cv2.destroyAllWindows()
    print("\nDone!")

if __name__ == "__main__":
    try:
        import tifffile
    except ImportError:
        print("Installing tifffile...")
        os.system("pip install tifffile")
        import tifffile
    
    import sys
    
    # Check if a file path is provided as command-line argument
    if len(sys.argv) > 1:
        # Single-file mode
        file_path = sys.argv[1]
        view_single_file(file_path)
    else:
        # Original directory scanning mode
        main()