

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=str)
    parser.add_argument("--camera-name", required=True,
                         help="e.g. phone_droidcam")
    parser.add_argument("--csv-out", type=str, default=None,
                         help="Write every sample (not just the console preview) to this CSV.")
    parser.add_argument("--patch", type=int, default=3,
                         help="Sample a (2*patch+1)-pixel square median around each "
                              "fingertip pixel instead of a single pixel -- same "
                              "anti-noise idea as inspect_fingertip_depths.py's sampler.")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    ann_dir = session_dir / f"annotations_{args.camera_name}"
    depth_dir = session_dir / f"depth_{args.camera_name}"

    if not ann_dir.exists():
        print(f"ERROR: {ann_dir} not found")
        raise SystemExit(1)
    if not depth_dir.exists():
        print(f"ERROR: {depth_dir} not found -- run estimate_phone_depth.py first")
        raise SystemExit(1)

    ann_files = sorted(ann_dir.glob("*.json"))
    print(f"Found {len(ann_files)} annotation frames for '{args.camera_name}'\n")

    all_depths_mm = []
    csv_rows = []

    print(f"{'frame':>8}  {'hand':<8}  {'finger':<8}  {'depth_mm':>10}")

    for ann_file in ann_files:
        with open(ann_file) as f:
            ann = json.load(f)

        depth_png = depth_dir / ann_file.name.replace(".json", ".png")
        if not depth_png.exists():
            continue
        depth_mm_img = cv2.imread(str(depth_png), cv2.IMREAD_UNCHANGED)
        if depth_mm_img is None:
            continue
        h, w = depth_mm_img.shape[:2]

        for hand in ann.get("hands", []):
            handedness = hand.get("handedness", "?")
            for finger, px_py in hand.get("fingertip_pixels", {}).items():
                px, py = int(px_py[0]), int(px_py[1])
                px = max(0, min(px, w - 1))
                py = max(0, min(py, h - 1))
                x1, x2 = max(0, px - args.patch), min(w, px + args.patch + 1)
                y1, y2 = max(0, py - args.patch), min(h, py + args.patch + 1)
                patch = depth_mm_img[y1:y2, x1:x2]
                valid = patch[patch > 0]
                if valid.size == 0:
                    continue
                depth_mm = float(np.median(valid))
                all_depths_mm.append(depth_mm)

                frame_id = ann.get("frame_id", 0)
                csv_rows.append({
                    "frame_id": frame_id,
                    "handedness": handedness,
                    "finger": finger,
                    "depth_mm": round(depth_mm, 2),
                })

                if int(frame_id) % 20 == 0:
                    print(f"{frame_id:>8}  {handedness:<8}  {finger:<8}  {depth_mm:>10.1f}")

    if not all_depths_mm:
        print("\nNo valid depth samples found at any fingertip position.")
        return

    arr = np.array(all_depths_mm)
    print(f"\n{'=' * 60}")
    print(f"Phone estimated fingertip depth stats across {len(arr)} samples:")
    print(f"  min:    {arr.min():.1f}mm")
    print(f"  median: {np.median(arr):.1f}mm")
    print(f"  max:    {arr.max():.1f}mm")
    print(f"  std:    {arr.std():.1f}mm")
    print(f"{'=' * 60}")
    print("What to look for: values should sit in a plausible hand-to-phone-camera")
    print("range and vary meaningfully across frames as the hand moved -- NOT")
    print("collapse to one near-constant value, and not spike to 0 or 65535 (clipping).")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_id", "handedness", "finger", "depth_mm"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote {len(csv_rows)} row(s) to {args.csv_out}")


if __name__ == "__main__":
    main()