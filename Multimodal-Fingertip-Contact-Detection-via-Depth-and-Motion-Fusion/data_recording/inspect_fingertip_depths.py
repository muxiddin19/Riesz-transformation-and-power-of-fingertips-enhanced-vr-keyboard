

import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=str)
    parser.add_argument("--csv-out", type=str, default=None,
                         help="Write ALL samples (not just the console preview) "
                              "to this CSV path.")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    annotations_dir = session_dir / "annotations"
    meta_path = session_dir / "metadata.json"

    if not annotations_dir.exists():
        print(f"ERROR: {annotations_dir} not found")
        raise SystemExit(1)

    avg_surface_depth_m = None
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        calib = meta.get("surface_calibration")
        if calib:
            avg_surface_depth_m = calib.get("avg_surface_depth_m")

    if avg_surface_depth_m is not None:
        print(f"Calibrated surface depth: {avg_surface_depth_m * 1000:.1f}mm "
              f"(from metadata.json's surface_calibration)\n")
    else:
        print("No surface_calibration found in metadata.json -- "
              "will still show raw fingertip depth stats, just can't "
              "compare against surface distance.\n")

    ann_files = sorted(annotations_dir.glob("*.json"))
    print(f"Found {len(ann_files)} annotation frames\n")

    all_depths_mm = []
    near_surface_count = 0  # frames where a fingertip reads within 3mm of the surface
    checked_count = 0
    csv_rows = []  # every sample, regardless of the console's 1-in-20 preview

    print(f"{'frame':>8}  {'hand':<8}  {'finger':<8}  {'depth_mm':>10}  "
          f"{'dist_to_surface_mm':>20}")

    for ann_file in ann_files:
        with open(ann_file) as f:
            ann = json.load(f)

        for hand in ann.get("hands", []):
            handedness = hand.get("handedness", "?")
            for finger, depth_m in hand.get("fingertip_depths_m", {}).items():
                if depth_m is None:
                    continue
                depth_mm = depth_m * 1000.0
                all_depths_mm.append(depth_mm)
                checked_count += 1

                dist_mm_val = None
                dist_str = ""
                if avg_surface_depth_m is not None:
                    dist_mm_val = (avg_surface_depth_m - depth_m) * 1000.0
                    dist_str = f"{dist_mm_val:.1f}"
                    if abs(dist_mm_val) < 3.0:
                        near_surface_count += 1

                csv_rows.append({
                    "frame_id": ann["frame_id"],
                    "handedness": handedness,
                    "finger": finger,
                    "depth_mm": round(depth_mm, 2),
                    "dist_to_surface_mm": round(dist_mm_val, 2) if dist_mm_val is not None else "",
                })

                # only print a sample so this doesn't spam thousands of lines
                # (the CSV, if requested, still gets every single sample)
                if int(ann["frame_id"]) % 20 == 0:
                    print(f"{ann['frame_id']:>8}  {handedness:<8}  {finger:<8}  "
                          f"{depth_mm:>10.1f}  {dist_str:>20}")

    if not all_depths_mm:
        print("\nNo non-null fingertip depths found at all -- every reading was "
              "None/0. That's a different problem (depth sampling failing "
              "entirely), not a hole-fill issue.")
        return

    arr = np.array(all_depths_mm)
    print(f"\n{'='*60}")
    print(f"Fingertip depth stats across {checked_count} samples:")
    print(f"  min:    {arr.min():.1f}mm")
    print(f"  median: {np.median(arr):.1f}mm")
    print(f"  max:    {arr.max():.1f}mm")
    print(f"  std:    {arr.std():.1f}mm")

    if avg_surface_depth_m is not None:
        pct_near_surface = 100.0 * near_surface_count / checked_count
        print(f"\n  Samples within 3mm of calibrated surface depth: "
              f"{near_surface_count}/{checked_count} ({pct_near_surface:.0f}%)")
        print(f"\nWhat to look for:")
        print(f"  - If fingertip depth std is very small (a few mm) and most")
        print(f"    samples sit near the surface depth EVEN WHILE the hand was")
        print(f"    clearly hovering (not touching) during recording, that's the")
        print(f"    hole-fill-to-background symptom -- consider trying")
        print(f"    HOLE_FILL_MODE = 2 (nearest_from_around) in camera_depth.py's")
        print(f"    DepthFilter instead of 1 (farthest_from_around).")
        print(f"  - If fingertip depth clearly varies over a wide range (tens of")
        print(f"    mm) as you'd expect from raising/lowering your hand, and only")
        print(f"    drops near the surface value during actual taps, the depth")
        print(f"    pipeline is working correctly and the red-ish colorized image")
        print(f"    was just a visualization contrast artifact, not a data problem.")
    print(f"{'='*60}")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["frame_id", "handedness", "finger", "depth_mm", "dist_to_surface_mm"]
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote {len(csv_rows)} rows (every sample, not just the console "
              f"preview) to {args.csv_out}")


if __name__ == "__main__":
    main()