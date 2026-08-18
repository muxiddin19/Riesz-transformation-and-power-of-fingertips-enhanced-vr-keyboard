"""
build_dataset_manifest.py
==========================
Aggregates a recorded session's per-fingertip contact/hover labels
(labels/*.json, written by generate_contact_labels.py) and per-fingertip
key lookups (annotations/*.json's "fingertip_keys" field, written by the
patched recorder.py) into ONE FLAT ROW PER (frame, hand) -- matching the
exact column layout your published Hugging Face dataset already uses:

    participant, session, frame_id, angle, surface, has_contact, handedness

plus one new column for the keyboard-click feature added on top of
that:

    clicked_key -- which key (if any) was under a fingertip that was
                   actually in "contact" state this frame, or empty/None
                   if no fingertip was in contact, or the contacting
                   fingertip wasn't over any annotated key

This is the piece that's been missing this whole time: nothing before
this script ever turns generate_contact_labels.py's fine-grained
per-fingertip output into the flat has_contact row your dataset's
schema actually expects.

Prerequisite: generate_contact_labels.py must have already run for a
session (it needs to have produced that session's labels/ folder --
this script only COMBINES what's already on disk, it does no contact
detection of its own).

Usage:
  # one session:
  python build_dataset_manifest.py \\
      --session-dir data/P01/P01_white_desk_typing_231530 \\
      --out data/P01/P01_white_desk_typing_231530/manifest.csv

  # every session under a root, into one combined manifest:
  python build_dataset_manifest.py \\
      --data-root data --out data/full_manifest.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path


def find_primary_depth_camera(meta):
    for cam in meta.get("cameras", []) or []:
        if cam.get("type") == "depth":
            return cam
    return None


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

    depth_cam = find_primary_depth_camera(meta)
    angle = depth_cam.get("angle") if depth_cam else None
    depth_cam_name = depth_cam.get("name") if depth_cam else None

    rows = []
    label_files = sorted(labels_dir.glob("*.json"))

    for label_file in label_files:
        frame_id = label_file.stem  # e.g. "000123"
        ann_file = annotations_dir / label_file.name
        if not ann_file.exists():
            # A label exists but its matching raw annotation doesn't.
            # Shouldn't normally happen -- generate_contact_labels.py only
            # ever reads frames FROM annotations/ in the first place --
            # but don't silently drop this without saying so.
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

        # generate_contact_labels.py builds labels/*.json's "hands" list by
        # iterating annotations/*.json's "hands" list IN ORDER and appending
        # 1:1 -- so index i in each list is the same hand. See
        # generate_contact_labels.py's process_session() if you ever need
        # to double check this assumption after an upstream change.
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

            rows.append({
                "participant": participant,
                "session": session_id,
                "frame_id": frame_id,
                "angle": angle,
                "surface": surface,
                "has_contact": has_contact,
                "handedness": handedness,
                "clicked_key": clicked_key or "",
                "rgb_path": str(session_dir / f"rgb_{depth_cam_name}" / f"{frame_id}.png") if depth_cam_name else "",
                "depth_path": str(session_dir / "depth" / f"{frame_id}.png"),
            })

    n_contact = sum(1 for r in rows if r["has_contact"])
    n_keyed = sum(1 for r in rows if r["clicked_key"])
    print(f"  {session_dir.name}: {len(rows)} row(s) ({n_contact} contact, {n_keyed} with a matched key)")
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

    fieldnames = ["participant", "session", "frame_id", "angle", "surface",
                  "has_contact", "handedness", "clicked_key", "rgb_path", "depth_path"]
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    total_contact = sum(1 for r in all_rows if r["has_contact"])
    total_keyed = sum(1 for r in all_rows if r["clicked_key"])
    print(f"\nWrote {len(all_rows)} row(s) to {args.out}")
    print(f"  has_contact=True: {total_contact} ({total_contact / len(all_rows) * 100:.1f}%)")
    print(f"  clicked_key set:  {total_keyed} ({total_keyed / len(all_rows) * 100:.1f}%)")


if __name__ == "__main__":
    main()