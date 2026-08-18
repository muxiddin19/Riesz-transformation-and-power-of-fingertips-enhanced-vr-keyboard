

import argparse
import csv
import os

import cv2
import numpy as np


def load_manifest(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["has_contact"] = str(r.get("has_contact", "")).strip().lower() in ("true", "1", "yes")
    return rows


def split_by_participant(rows, train_ratio, val_ratio, seed):
    participants = sorted(set(r["participant"] for r in rows))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(participants))
    participants = [participants[i] for i in perm]

    n = len(participants)

    if n <= 2:
        # Too few participants to stratify meaningfully -- split FRAMES
        # directly instead (same fallback prepare_dataset.py uses).
        print(f"Only {n} participant(s) -- splitting frames directly instead of by "
              f"participant (some identity leakage across splits is possible with "
              f"this few participants -- record more before trusting eval numbers).")
        idx = list(range(len(rows)))
        rng2 = np.random.RandomState(seed)
        rng2.shuffle(idx)
        n_train_f = int(len(idx) * train_ratio)
        n_val_f = int(len(idx) * val_ratio)
        train_idx = set(idx[:n_train_f])
        val_idx = set(idx[n_train_f:n_train_f + n_val_f])
        return (
            [r for i, r in enumerate(rows) if i in train_idx],
            [r for i, r in enumerate(rows) if i in val_idx],
            [r for i, r in enumerate(rows) if i not in train_idx and i not in val_idx],
        )

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train_p = set(participants[:n_train])
    val_p = set(participants[n_train:n_train + n_val])
    test_p = set(participants[n_train + n_val:])

    print(f"Train participants: {sorted(train_p)}")
    print(f"Val participants:   {sorted(val_p)}")
    print(f"Test participants:  {sorted(test_p)}")

    train_rows = [r for r in rows if r["participant"] in train_p]
    val_rows = [r for r in rows if r["participant"] in val_p]
    test_rows = [r for r in rows if r["participant"] in test_p]
    return train_rows, val_rows, test_rows


def build_split(rows, out_dir, split_name):
    if not rows:
        print(f"  {split_name}: 0 rows, skipping")
        return

    rgb_list = []
    depth_list = []
    contact_list = []
    kept_rows = []

    for r in rows:
        rgb_path = r.get("rgb_path", "")
        depth_path = r.get("depth_path", "")
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR) if rgb_path else None
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) if depth_path else None
        if rgb is None or depth is None:
            print(f"    WARNING: could not load '{rgb_path}' / '{depth_path}', skipping row")
            continue
        rgb_list.append(rgb)
        depth_list.append(depth)
        contact_list.append(1 if r["has_contact"] else 0)
        kept_rows.append(r)

    if not rgb_list:
        print(f"  {split_name}: 0 loadable rows, skipping")
        return

    # NOTE: this assumes every row's RGB/depth share the same resolution
    # (true as long as they all come from the same physical camera --
    # build_dataset_manifest.py only ever points rgb_path/depth_path at
    # the primary depth camera's own frames). Mixing cameras of
    # different resolutions here would make np.stack fail loudly, which
    # is the right failure mode -- silently resizing would hide a real
    # inconsistency instead.
    rgb_arr = np.stack(rgb_list, axis=0).astype(np.uint8)
    depth_arr = np.stack(depth_list, axis=0).astype(np.uint16)
    contact_arr = np.array(contact_list, dtype=np.uint8)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{split_name}_rgb.npy"), rgb_arr)
    np.save(os.path.join(out_dir, f"{split_name}_depth.npy"), depth_arr)
    np.save(os.path.join(out_dir, f"{split_name}_has_contact.npy"), contact_arr)

    fieldnames = list(kept_rows[0].keys())
    with open(os.path.join(out_dir, f"{split_name}_manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"  {split_name}: {len(kept_rows)} sample(s) -> "
          f"rgb{rgb_arr.shape} depth{depth_arr.shape} "
          f"contact_rate={contact_arr.mean() * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True,
                         help="CSV from build_dataset_manifest.py (single session or --data-root combined).")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    print(f"Loaded {len(rows)} row(s) from {args.manifest}")

    train_rows, val_rows, test_rows = split_by_participant(
        rows, args.train_ratio, args.val_ratio, args.seed
    )

    print("\nPacking .npy arrays...")
    build_split(train_rows, args.out_dir, "train")
    build_split(val_rows, args.out_dir, "val")
    build_split(test_rows, args.out_dir, "test")

    print(f"\nDone. Files written to {args.out_dir}/")


if __name__ == "__main__":
    main()