"""
build_dataset_manifest.py
==========================
Aggregates a recorded session's per-fingertip contact/hover labels
(labels/*.json) and per-fingertip key lookups (annotations/*.json's
"fingertip_keys" field) into ONE FLAT ROW PER (frame, hand).

UPDATED: now generic over however many cameras are listed in a
session's metadata.json["cameras"] -- not hardcoded to a single
"depth_cam_name". For every camera in that list it adds:
    rgb_<camera_name>_path    -- always present right after recording
    depth_<camera_name>_path  -- for the depth camera, this is the real
                                  D405 depth/ folder; for RGB cameras,
                                  this is the ESTIMATED depth written by
                                  estimate_phone_depth.py -- empty string
                                  if that hasn't been run yet for this
                                  camera/session.

Because different sessions can have different camera rigs (e.g. old
2-camera sessions vs new 4-camera sessions), the final CSV's columns
are the UNION of every column seen across all processed sessions --
missing values are written as "".
"""

import argparse
import csv
import json
import os
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def process_session(session_dir):
    session_dir = Path(session_dir)
    meta_path = session_dir / "metadata.json"
    labels_dir = session_dir / "labels"
    annotations_dir = session_dir / "annotations"

    if not meta_path.exists():
        print(f"  SKIP {session_dir.name}: no metadata.json")
        return []
    if not labels_dir.exists():
        print(f"  SKIP {session_dir.name}: no labels/ -- run generate_contact_labels.py first")
        return []
    if not annotations_dir.exists():
        print(f"  SKIP {session_dir.name}: no annotations/")
        return []

    meta = load_json(meta_path)
    participant = meta.get("participant", "unknown")
    session_id = meta.get("session_id", session_dir.name)
    surface = meta.get("surface", "unknown")

    # All cameras present in this session (depth + every rgb one).
    cameras = meta.get("cameras", []) or []
    depth_cams = [c for c in cameras if c.get("type") == "depth"]
    rgb_cams = [c for c in cameras if c.get("type") != "depth"]

    primary_depth_cam = depth_cams[0] if depth_cams else None
    angle = primary_depth_cam.get("angle") if primary_depth_cam else None

    def camera_paths(cam_name, frame_id):
        """Returns (rgb_path, depth_path) for one camera + frame, empty
        string for either that doesn't exist on disk yet."""
        rgb_path = session_dir / f"rgb_{cam_name}" / f"{frame_id}.png"
        depth_path = session_dir / f"depth_{cam_name}" / f"{frame_id}.png"
        # The primary depth camera's REAL depth lives in plain depth/,
        # not depth_<name>/ -- matches recorder.py's _save_depth_frame.
        if primary_depth_cam and cam_name == primary_depth_cam.get("name"):
            depth_path = session_dir / "depth" / f"{frame_id}.png"
        return (
            str(rgb_path) if rgb_path.exists() else "",
            str(depth_path) if depth_path.exists() else "",
        )

    rows = []
    label_files = sorted(labels_dir.glob("*.json"))

    for label_file in label_files:
        frame_id = label_file.stem
        ann_file = annotations_dir / label_file.name
        if not ann_file.exists():
            print(f"    WARNING: {label_file.name} has no matching annotations/ file, skipping frame")
            continue

        label_data = load_json(label_file)
        ann_data = load_json(ann_file)

        label_hands = label_data.get("hands", [])
        ann_hands = ann_data.get("hands", [])
        if len(label_hands) != len(ann_hands):
            print(f"    WARNING: frame {frame_id} hand-count mismatch between "
                  f"labels/ ({len(label_hands)}) and annotations/ ({len(ann_hands)}) "
                  f"-- zipping only the overlapping hands, rest dropped for this frame")

        for label_hand, ann_hand in zip(label_hands, ann_hands):
            handedness = label_hand.get("handedness", "Unknown")
            fingertips = label_hand.get("fingertips", {})
            fingertip_keys = ann_hand.get("fingertip_keys", {})

            has_contact = any(f.get("state") == "contact" for f in fingertips.values())

            clicked_key = None
            for finger, finfo in fingertips.items():
                if finfo.get("state") == "contact":
                    key = fingertip_keys.get(finger)
                    if key:
                        clicked_key = key
                        break

            row = {
                "participant": participant,
                "session": session_id,
                "frame_id": frame_id,
                "angle": angle,
                "surface": surface,
                "has_contact": has_contact,
                "handedness": handedness,
                "clicked_key": clicked_key or "",
            }

            # One rgb_<name>_path / depth_<name>_path pair per camera
            # actually present in THIS session's metadata.
            for cam in depth_cams + rgb_cams:
                cam_name = cam.get("name")
                if not cam_name:
                    continue
                rgb_path, depth_path = camera_paths(cam_name, frame_id)
                row[f"rgb_{cam_name}_path"] = rgb_path
                row[f"depth_{cam_name}_path"] = depth_path

            rows.append(row)

    n_contact = sum(1 for r in rows if r["has_contact"])
    n_keyed = sum(1 for r in rows if r["clicked_key"])
    cam_names = [c.get("name") for c in depth_cams + rgb_cams]
    print(f"  {session_dir.name}: {len(rows)} row(s) ({n_contact} contact, {n_keyed} with a matched key) "
          f"-- cameras: {cam_names}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-dir", default=None, help="Single session directory.")
    parser.add_argument("--data-root", default=None,
                         help="Process every session found under this root "
                              "(looks for <root>/<participant>/<session>/metadata.json).")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()

    if not args.session_dir and not args.data_root:
        print("ERROR: provide --session-dir or --data-root")
        raise SystemExit(1)

    if args.session_dir:
        sessions = [args.session_dir]
    else:
        data_root = Path(args.data_root)
        sessions = sorted(str(p.parent) for p in data_root.glob("*/*/metadata.json"))
        print(f"Found {len(sessions)} session(s) under {args.data_root}")

    all_rows = []
    for session_dir in sessions:
        print(f"Processing: {session_dir}")
        all_rows.extend(process_session(session_dir))

    if not all_rows:
        print("\nNo rows produced -- nothing written.")
        return

    # Base columns always present, in a fixed order, followed by every
    # rgb_*/depth_* column seen ANYWHERE across all rows (sessions with
    # fewer cameras just get "" for the columns they don't have).
    base_fields = ["participant", "session", "frame_id", "angle", "surface",
                   "has_contact", "handedness", "clicked_key"]
    dynamic_fields = sorted({
        k for row in all_rows for k in row.keys() if k not in base_fields
    })
    fieldnames = base_fields + dynamic_fields

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(all_rows)

    total_contact = sum(1 for r in all_rows if r["has_contact"])
    total_keyed = sum(1 for r in all_rows if r["clicked_key"])
    print(f"\nWrote {len(all_rows)} row(s) to {args.out}")
    print(f"  columns: {fieldnames}")
    print(f"  has_contact=True: {total_contact} ({total_contact / len(all_rows) * 100:.1f}%)")
    print(f"  clicked_key set:  {total_keyed} ({total_keyed / len(all_rows) * 100:.1f}%)")


if __name__ == "__main__":
    main()