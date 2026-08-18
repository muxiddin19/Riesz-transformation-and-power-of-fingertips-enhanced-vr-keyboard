"""
check_depth_frame.py
======================
Sanity-check + visualize a saved depth PNG. Depth frames are saved as
raw uint16 millimeters, so a normal image viewer just shows near-black
(300-800mm out of a 0-65535 range) — that's expected, NOT corrupted
data. This script proves it's fine and gives you something you can
actually look at.

    python check_depth_frame.py data/P01/P01_wood_grain_typing_194158/depth/000002.png
"""

import sys

import cv2
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_depth_frame.py <path_to_depth_png>")
        raise SystemExit(1)

    path = sys.argv[1]

    depth_mm = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        print(f"Could not read {path}")
        raise SystemExit(1)

    print(f"dtype: {depth_mm.dtype}")
    print(f"shape: {depth_mm.shape}")

    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        print("WARNING: every pixel is 0 — depth camera may not have been "
              "seeing a valid surface when this frame was captured.")
        raise SystemExit(1)

    print(f"non-zero pixels: {valid.size} / {depth_mm.size}")
    print(f"min depth: {valid.min()} mm")
    print(f"max depth: {valid.max()} mm")
    print(f"median depth: {int(np.median(valid))} mm")

    display = np.zeros_like(depth_mm, dtype=np.uint8)
    norm = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX,
                          mask=(depth_mm > 0).astype(np.uint8))
    display = norm.astype(np.uint8)
    colored = cv2.applyColorMap(display, cv2.COLORMAP_JET)

    out_path = path.replace(".png", "_visualized.png")
    cv2.imwrite(out_path, colored)
    print(f"\nSaved a viewable, colorized version to: {out_path}")
    print("(closer = one color, farther = another — check it looks like "
          "your keyboard/hand shape, not noise)")


if __name__ == "__main__":
    main()