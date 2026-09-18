"""
run_pipeline.py
================
Single entry point for the whole recording -> dataset pipeline, so you
don't have to remember which script to run, in what order, with which
flags.

    python run_pipeline.py

then pick a numbered option. This doesn't change what any individual
script does -- it just calls them for you and asks for the handful of
values each one needs (paths, camera names, etc).

IMPORTANT: steps 1-3 (calibration and recording) still need YOU
physically there moving a chessboard or your hands in front of a
camera -- no script can automate that part. Steps 4-6 (manifest
building, depth estimation, exporting training arrays) are fully
automatic once you point them at a data folder.
"""

import os
import subprocess
import sys

import config


def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[PIPELINE] '{' '.join(cmd)}' exited with code {result.returncode} -- stopping.")
        sys.exit(result.returncode)


def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or default


def step_calibrate_intrinsics():
    names = [c["name"] for c in config.CAMERAS]
    print(f"\nCameras in config.py: {names}")
    print("Calibrate them ONE AT A TIME. Do the depth camera first.")
    for name in names:
        again = ask(f"Calibrate intrinsics for '{name}' now? (y/n)", "y")
        if again.lower().startswith("y"):
            run([sys.executable, "calibrate_cameras.py", "--intrinsics", "--camera", name])


def step_calibrate_extrinsics():
    run([sys.executable, "calibrate_cameras.py", "--extrinsics"])


def step_record():
    run([sys.executable, "main.py"])


def step_build_manifest():
    data_root = ask("Path to recorded data root", config.OUTPUT_ROOT)
    out_csv = ask("Output manifest CSV path", os.path.join(data_root, "manifest.csv"))
    run([sys.executable, "build_dataset_manifest.py", "--data-root", data_root, "--out", out_csv])
    return out_csv


def step_estimate_phone_depth():
    if not os.path.exists("estimate_phone_depth.py"):
        print("\n[PIPELINE] estimate_phone_depth.py not found in this folder -- "
              "skipping. Only needed if you want estimated depth for the RGB "
              "cameras (build_dataset_manifest.py's docstring references it; "
              "not required if you only train on the real depth camera's data).")
        return
    print("\nRequires the official Depth-Anything-V2 repo importable "
          "(git clone https://github.com/DepthAnything/Depth-Anything-V2.git, "
          "run from inside it or add it to PYTHONPATH) and your fine-tuned "
          "checkpoint from depth_finetuning/train.py.")
    checkpoint = ask("Path to fine-tuned checkpoint",
                      "checkpoints/depth_anything_v2_vits_d405_finetuned.pth")
    data_root = ask("Path to recorded data root (or leave blank to pick one session)",
                     config.OUTPUT_ROOT)
    session_dir = None
    if not data_root:
        session_dir = ask("Path to a single session directory")
    for cam_cfg in config.CAMERAS:
        if cam_cfg["type"] == "depth":
            continue
        do_it = ask(f"Estimate depth for '{cam_cfg['name']}'? (y/n)", "y")
        if not do_it.lower().startswith("y"):
            continue
        cmd = [sys.executable, "estimate_phone_depth.py",
               "--camera-name", cam_cfg["name"], "--checkpoint", checkpoint]
        cmd += ["--data-root", data_root] if data_root else ["--session-dir", session_dir]
        run(cmd)


def step_export_npy(manifest_csv=None):
    manifest_csv = manifest_csv or ask("Path to manifest CSV (from build_dataset_manifest.py)")
    out_dir = ask("Output directory for .npy arrays", "./training_data")
    run([sys.executable, "export_npy_dataset.py", "--manifest", manifest_csv, "--out-dir", out_dir])


MENU = """
=== Recording Pipeline ===
  1. Calibrate camera intrinsics (one-time per camera -- do this before anything else)
  2. Calibrate joint extrinsics (one-time, after step 1 for ALL cameras)
  3. Record a session (runs main.py)
  4. Build dataset manifest from recorded sessions
  5. Estimate RGB-camera depth (if estimate_phone_depth.py exists)
  6. Export training .npy arrays
  7. Run 4 -> 5 -> 6 in sequence (full post-processing)
  q. Quit
"""


def main():
    while True:
        print(MENU)
        choice = input("Choose an option: ").strip().lower()
        if choice == "1":
            step_calibrate_intrinsics()
        elif choice == "2":
            step_calibrate_extrinsics()
        elif choice == "3":
            step_record()
        elif choice == "4":
            step_build_manifest()
        elif choice == "5":
            step_estimate_phone_depth()
        elif choice == "6":
            step_export_npy()
        elif choice == "7":
            manifest_csv = step_build_manifest()
            step_estimate_phone_depth()
            step_export_npy(manifest_csv)
        elif choice == "q":
            break
        else:
            print("Not a valid option.")


if __name__ == "__main__":
    main()