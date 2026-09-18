"""
calibrate_cameras.py
=====================
Chessboard calibration for the full multi-camera rig.

Run this ONCE per physical rig setup, before recording any participant,
with an empty desk. Produces the calibration files that camera_depth.py
and camera_rgb.py already know how to load (they warn "no calibration
found ... run calibrate_cameras.py before recording" -- this is that
script):

    data/calibrations/intrinsics_<camera_name>.json   (per camera)
    data/calibrations/extrinsics.json                 (RGB cams -> depth cam)

USAGE
-----
1. Per-camera intrinsics -- run once for EACH camera in config.CAMERAS
   (including the depth camera's color stream):

       python calibrate_cameras.py --intrinsics --camera depth_cam
       python calibrate_cameras.py --intrinsics --camera cam1
       python calibrate_cameras.py --intrinsics --camera cam2
       python calibrate_cameras.py --intrinsics --camera cam3

   Hold the printed chessboard (config.CHECKERBOARD internal corners,
   config.SQUARE_SIZE_MM per square) in front of that camera at many
   different positions/angles/distances/tilts -- variety matters more
   than count. SPACE captures a frame once corners are found (drawn in
   green), ESC finishes and computes. Needs >= 15 good captures.

2. Joint (extrinsic) calibration -- run ONCE, after step 1 has produced
   intrinsics for the depth camera AND every RGB camera you want
   related to it:

       python calibrate_cameras.py --extrinsics

   For each RGB camera in turn, hold the SAME chessboard where it is
   simultaneously visible to BOTH the depth camera and that RGB camera.
   SPACE captures only when both views see the board at once, ESC moves
   to the next camera. This is what lets you take a point (e.g. a
   fingertip) in the depth camera's 3D frame and reproject it into any
   RGB camera's pixel coordinates later, or vice versa.

Do this ONCE per physical rig, before participant 1, with the rig fully
mounted where it will stay. Do not move any camera afterward -- see
STUDENT_RECORDING_GUIDE.md. If anything gets bumped, redo both steps
for the affected camera(s) before recording again.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

import config
from camera_depth import DepthCamera
from camera_rgb import RGBCamera


def _object_points():
    """3D chessboard corner positions in the board's own coordinate
    frame (mm), z=0 since the board is planar."""
    cols, rows = config.CHECKERBOARD
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= config.SQUARE_SIZE_MM
    return objp


def _find_corners(gray):
    cols, rows = config.CHECKERBOARD
    found, corners = cv2.findChessboardCorners(
        gray, (cols, rows),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def _build_camera(name):
    for cam_cfg in config.CAMERAS:
        if cam_cfg["name"] == name:
            if cam_cfg["type"] == "depth":
                cam = DepthCamera(name, config.DEPTH_CAM_WIDTH, config.DEPTH_CAM_HEIGHT, config.TARGET_FPS)
            else:
                cam = RGBCamera(name, cam_cfg["source"], config.RGB_WIDTH, config.RGB_HEIGHT,
                                 config.TARGET_FPS, cam_cfg.get("backend", "auto"))
            cam.start()
            return cam, cam_cfg["type"]
    raise ValueError(f"No camera named '{name}' in config.CAMERAS")


def _get_color_frame(cam):
    """Works for both DepthCamera (4-tuple) and RGBCamera (2-tuple) --
    color/frame is always element 0."""
    data = cam.get_latest()
    return None if data is None else data[0]


def run_intrinsics(camera_name, min_captures=15):
    print(f"\n=== Intrinsic calibration: {camera_name} ===")
    print("SPACE = capture (only works when corners are found, shown in green)")
    print(f"ESC   = finish and compute (need at least {min_captures} good captures)\n")

    cam, _ = _build_camera(camera_name)
    time.sleep(1.0)  # let auto-exposure settle

    objp = _object_points()
    objpoints, imgpoints = [], []
    img_shape = None

    try:
        while True:
            frame = _get_color_frame(cam)
            if frame is None:
                continue
            img_shape = frame.shape[1::-1]  # (w, h)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = _find_corners(gray)

            display = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(display, config.CHECKERBOARD, corners, True)
            cv2.putText(display, f"captures: {len(objpoints)}/{min_captures}+",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(f"calibrate: {camera_name}", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == 32 and corners is not None:  # SPACE
                objpoints.append(objp)
                imgpoints.append(corners)
                print(f"  captured {len(objpoints)}")
    finally:
        cam.stop()
        cv2.destroyAllWindows()

    if len(objpoints) < min_captures:
        print(f"ERROR: only {len(objpoints)} captures, need at least {min_captures}. Not saving.")
        return

    print("\nComputing intrinsics...")
    reproj_err, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    quality = "good" if reproj_err < 0.5 else "high -- consider recapturing with more varied angles"
    print(f"Reprojection error: {reproj_err:.3f}px ({quality})")

    os.makedirs(config.CALIBRATION_DIR, exist_ok=True)
    out_path = os.path.join(config.CALIBRATION_DIR, f"intrinsics_{camera_name}.json")
    with open(out_path, "w") as f:
        json.dump({
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "reprojection_error_px": float(reproj_err),
            "image_size": img_shape,
        }, f, indent=2)
    print(f"Saved: {out_path}")


def run_extrinsics(min_captures=15):
    """Extrinsics of every RGB camera relative to the depth camera, via
    cv2.stereoCalibrate on simultaneous chessboard views. Requires
    intrinsics already computed for every camera involved."""
    depth_cfg = next((c for c in config.CAMERAS if c["type"] == "depth"), None)
    rgb_cfgs = [c for c in config.CAMERAS if c["type"] != "depth"]
    if depth_cfg is None:
        print("ERROR: no depth camera in config.CAMERAS")
        return

    def load_intrinsics(name):
        path = os.path.join(config.CALIBRATION_DIR, f"intrinsics_{name}.json")
        if not os.path.exists(path):
            print(f"ERROR: no intrinsics for '{name}' -- run --intrinsics --camera {name} first")
            return None
        with open(path) as f:
            data = json.load(f)
        return np.array(data["camera_matrix"]), np.array(data["dist_coeffs"])

    depth_intr = load_intrinsics(depth_cfg["name"])
    if depth_intr is None:
        return
    depth_camera_matrix, depth_dist = depth_intr

    print(f"\n=== Joint calibration: {depth_cfg['name']} <-> {[c['name'] for c in rgb_cfgs]} ===")
    print("Hold the chessboard where BOTH the depth camera and the RGB camera")
    print("currently being paired can see it. SPACE = capture, ESC = next camera.\n")

    depth_cam, _ = _build_camera(depth_cfg["name"])
    time.sleep(1.0)
    objp = _object_points()
    extrinsics = {}

    try:
        for rgb_cfg in rgb_cfgs:
            name = rgb_cfg["name"]
            rgb_intr = load_intrinsics(name)
            if rgb_intr is None:
                continue
            rgb_camera_matrix, rgb_dist = rgb_intr

            rgb_cam, _ = _build_camera(name)
            time.sleep(1.0)

            objpoints, img_pts_depth, img_pts_rgb = [], [], []
            img_shape = None

            print(f"\n--- Pairing {depth_cfg['name']} <-> {name} ---")
            try:
                while True:
                    d_frame = _get_color_frame(depth_cam)
                    r_frame = _get_color_frame(rgb_cam)
                    if d_frame is None or r_frame is None:
                        continue
                    img_shape = d_frame.shape[1::-1]
                    corners_d = _find_corners(cv2.cvtColor(d_frame, cv2.COLOR_BGR2GRAY))
                    corners_r = _find_corners(cv2.cvtColor(r_frame, cv2.COLOR_BGR2GRAY))

                    disp_d, disp_r = d_frame.copy(), r_frame.copy()
                    if corners_d is not None:
                        cv2.drawChessboardCorners(disp_d, config.CHECKERBOARD, corners_d, True)
                    if corners_r is not None:
                        cv2.drawChessboardCorners(disp_r, config.CHECKERBOARD, corners_r, True)
                    cv2.putText(disp_d, f"captures: {len(objpoints)}/{min_captures}+",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow(depth_cfg["name"], disp_d)
                    cv2.imshow(name, disp_r)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break
                    if key == 32 and corners_d is not None and corners_r is not None:
                        objpoints.append(objp)
                        img_pts_depth.append(corners_d)
                        img_pts_rgb.append(corners_r)
                        print(f"  captured {len(objpoints)}")
            finally:
                cv2.destroyAllWindows()
                rgb_cam.stop()

            if len(objpoints) < min_captures:
                print(f"  SKIP {name}: only {len(objpoints)} captures, need {min_captures}+")
                continue

            print(f"  Computing extrinsics for {name}...")
            reproj_err, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
                objpoints, img_pts_depth, img_pts_rgb,
                depth_camera_matrix, depth_dist,
                rgb_camera_matrix, rgb_dist,
                img_shape, flags=cv2.CALIB_FIX_INTRINSIC,
            )
            print(f"  Stereo reprojection error: {reproj_err:.3f}px")
            extrinsics[name] = {
                "reference_camera": depth_cfg["name"],
                "R": R.tolist(),
                "T": T.tolist(),
                "reprojection_error_px": float(reproj_err),
            }
    finally:
        depth_cam.stop()
        cv2.destroyAllWindows()

    if not extrinsics:
        print("\nNo extrinsics computed -- nothing saved.")
        return

    os.makedirs(config.CALIBRATION_DIR, exist_ok=True)
    out_path = os.path.join(config.CALIBRATION_DIR, "extrinsics.json")
    with open(out_path, "w") as f:
        json.dump(extrinsics, f, indent=2)
    print(f"\nSaved joint calibration: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--intrinsics", action="store_true",
                         help="Calibrate one camera's intrinsics/distortion.")
    parser.add_argument("--extrinsics", action="store_true",
                         help="Joint-calibrate every RGB camera against the depth camera.")
    parser.add_argument("--camera", default=None,
                         help="Camera name from config.CAMERAS (required with --intrinsics).")
    parser.add_argument("--min-captures", type=int, default=15)
    args = parser.parse_args()

    if args.intrinsics:
        if not args.camera:
            print("ERROR: --intrinsics requires --camera <name>")
            raise SystemExit(1)
        run_intrinsics(args.camera, args.min_captures)
    elif args.extrinsics:
        run_extrinsics(args.min_captures)
    else:
        print("Specify --intrinsics --camera <name>  or  --extrinsics")


if __name__ == "__main__":
    main()