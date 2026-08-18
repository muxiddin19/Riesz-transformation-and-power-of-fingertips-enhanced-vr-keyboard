from pathlib import Path

import cv2
import numpy as np


def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    
    mask = (depth_mm > 0).astype(np.uint8)
    norm = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, mask=mask)
    display = norm.astype(np.uint8)
    colored = cv2.applyColorMap(display, cv2.COLORMAP_JET)
    colored[mask == 0] = (0, 0, 0)
    return colored


def visualize_session_depth(session_dir: str) -> dict:
    
    depth_dir = Path(session_dir) / "depth"
    if not depth_dir.is_dir():
        return {"converted": 0, "empty_frames": 0, "skipped": True}

    png_files = sorted(depth_dir.glob("*.png"))
    if not png_files:
        return {"converted": 0, "empty_frames": 0, "skipped": True}

    out_dir = Path(session_dir) / "depth_visualized"
    out_dir.mkdir(exist_ok=True)

    converted = 0
    empty_frames = 0

    for png_path in png_files:
        out_path = out_dir / png_path.name
        if out_path.exists():
            continue  # already converted (e.g. session partially processed before)

        depth_mm = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            continue

        if (depth_mm > 0).sum() == 0:
            empty_frames += 1

        colored = colorize_depth(depth_mm)
        cv2.imwrite(str(out_path), colored)
        converted += 1

    return {
        "converted": converted,
        "empty_frames": empty_frames,
        "skipped": False,
        "out_dir": str(out_dir),
    }