

import os
import sys
import time
import subprocess
from collections import deque

import cv2
import numpy as np

import config
from session_manager import SessionManager
from sync_manager import MultiCameraSync
from depth_visualize import visualize_session_depth
from hand_depth_tracker import HandDepthTracker, SurfaceCalibrator
from keyboard_layout import load_keyboard_layout, find_key_at_point

TRACK_HANDS_ON_RGB_ONLY_CAMERAS = True

CALIBRATION_FRAMES = 30


LIVE_CONTACT_THRESHOLD_M = 0.0045
LIVE_CONTACT_WINDOW_FRAMES = 150 
_DEFAULT_LABELER_RELPATH = os.path.join("..", "custom_data_recording", "generate_contact_labels.py")

_DEFAULT_KEYBOARD_LAYOUT_RELPATH = os.path.join("..", "assets", "keyboard_annotations.json")


class Recorder:
    def __init__(self):
        self.sync = MultiCameraSync(
            camera_configs=config.CAMERAS,
            rgb_width=config.RGB_WIDTH,
            rgb_height=config.RGB_HEIGHT,
            depth_width=config.DEPTH_CAM_WIDTH,
            depth_height=config.DEPTH_CAM_HEIGHT,
            fps=config.TARGET_FPS,
            sync_tolerance_sec=config.SYNC_TOLERANCE_SEC,
        )
        self.session = SessionManager()

        self.is_recording = False
        self.frame_counter = 0

        self._live_contact_window = deque(maxlen=LIVE_CONTACT_WINDOW_FRAMES)
        self._session_contact_total = 0
        self._session_sample_total = 0

        
        self.hand_tracker = HandDepthTracker()
        self.primary_depth_cam_name = self._find_primary_depth_camera_name()
        self.calibrations_dir = os.path.join(config.OUTPUT_ROOT, "calibrations")
        self.current_calibration = None  # dict from SurfaceCalibrator.fit_plane(), or None

        
        labeler_relpath = getattr(config, "CONTACT_LABELER_SCRIPT_RELPATH", _DEFAULT_LABELER_RELPATH)
        self.contact_labeler_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), labeler_relpath
        )
        if not os.path.exists(self.contact_labeler_script):
            print(f"[LABELER] WARNING: expected generate_contact_labels.py at "
                  f"{self.contact_labeler_script} but it does not exist. "
                  f"Auto-labeling after each session will be skipped -- set "
                  f"CONTACT_LABELER_SCRIPT_RELPATH in config.py if your layout "
                  f"differs.")

        
        keyboard_layout_relpath = getattr(
            config, "KEYBOARD_ANNOTATIONS_RELPATH", _DEFAULT_KEYBOARD_LAYOUT_RELPATH
        )
        self.keyboard_layout_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), keyboard_layout_relpath
        )
        self.keyboard_layout = load_keyboard_layout(self.keyboard_layout_path)
        if self.keyboard_layout:
            print(f"[KEYBOARD] loaded {len(self.keyboard_layout)} key(s) from "
                  f"{self.keyboard_layout_path} -- fingertip annotations for "
                  f"'{self.primary_depth_cam_name}' will include which key each "
                  f"fingertip is over.")
        else:
            print(f"[KEYBOARD] no keyboard layout found at {self.keyboard_layout_path} "
                  f"-- annotations will NOT include a 'key' field. Run your "
                  f"keyboard-annotation tool once against the D405 (it never "
                  f"moves, so you only need to do this once for the whole "
                  f"project) if you want per-key click labels. Not needed for "
                  f"plain surface has_contact labeling.")

    
    def _find_primary_depth_camera_name(self):
        for cam_cfg in config.CAMERAS:
            if cam_cfg["type"] == "depth":
                return cam_cfg["name"]
        print("[RECORDER] WARNING: no camera with type=='depth' in config.CAMERAS — "
              "contact labeling will not be possible (no metric depth source).")
        return None

    def _calibration_path(self, surface):
        angle = self.sync.camera_angle(self.primary_depth_cam_name)
        fname = f"calib_{self.primary_depth_cam_name}_angle{angle}_{surface}.json"
        return os.path.join(self.calibrations_dir, fname)

    def _load_calibration_for_current_session(self):
        """Auto-load an existing calibration for the primary depth
        camera's current angle + this session's surface, if one exists,
        so metadata.json carries it even if you don't recalibrate today."""
        if self.primary_depth_cam_name is None or self.session.surface is None:
            return

        path = self._calibration_path(self.session.surface)
        if not os.path.exists(path):
            print(f"[CALIBRATION] none found at {path} — press 'c' to calibrate "
                  f"before recording (needed later for has_contact labeling).")
            self.current_calibration = None
            return

        import json
        with open(path) as f:
            self.current_calibration = json.load(f)
        self.session.update_surface_calibration(self.current_calibration)
        print(f"[CALIBRATION] loaded {path} "
              f"(avg surface depth = {self.current_calibration['avg_surface_depth_m'] * 1000:.1f}mm)")

    def _set_camera_angle(self):
        """PATCH: live angle change for a non-depth (RGB) camera -- e.g.
        the phone -- without restarting the whole program. Only touches
        config.CAMERAS/self.sync.camera_configs in-memory; does NOT
        restart any camera hardware, so the D405 never gets
        hardware_reset() just because you moved the phone. Prompts on
        the console since there's no text-input widget in the OpenCV
        preview window."""
        rgb_cam_names = [c["name"] for c in config.CAMERAS if c["type"] != "depth"]
        if not rgb_cam_names:
            print("[ANGLE] no non-depth cameras configured -- nothing to change.")
            return

        was_recording = self.is_recording
        self.is_recording = False
        if was_recording:
            print("[RECORDING] paused to change camera angle")

        if len(rgb_cam_names) == 1:
            target_name = rgb_cam_names[0]
        else:
            print(f"[ANGLE] which camera? {rgb_cam_names}")
            target_name = input("  camera name: ").strip()
            if target_name not in rgb_cam_names:
                print(f"[ANGLE] '{target_name}' is not a configured RGB camera -- cancelled.")
                return

        current = self.sync.camera_angle(target_name)
        raw = input(f"[ANGLE] new angle for '{target_name}' (currently {current}): ").strip()
        try:
            new_angle = int(raw)
        except ValueError:
            print(f"[ANGLE] '{raw}' is not a valid integer -- cancelled, angle unchanged.")
            return

        for cam_cfg in self.sync.camera_configs:
            if cam_cfg["name"] == target_name:
                cam_cfg["angle"] = new_angle
                break

        
        print(f"[ANGLE] '{target_name}' angle set to {new_angle}. "
              f"Any calibration for the OLD angle no longer applies to this "
              f"camera if it's ever used as a depth source -- for a plain "
              f"RGB camera like the phone this just changes the label saved "
              f"into future sessions' metadata.json and filenames.")

        if was_recording:
            print("[RECORDING] press 'r' to resume")

    def _run_calibration(self):
        """Blocking: collect CALIBRATION_FRAMES frames of the primary
        depth camera with an empty desk, fit a plane, save it."""
        if self.primary_depth_cam_name is None:
            print("[CALIBRATION] no depth camera configured — cannot calibrate.")
            return
        if self.session.surface is None:
            print("[CALIBRATION] no active session — start a session first (it picks the surface).")
            return

        print(f"\n[CALIBRATION] Remove all objects from '{self.session.surface}'. "
              f"Collecting {CALIBRATION_FRAMES} frames...")
        calibrator = SurfaceCalibrator(num_frames=CALIBRATION_FRAMES)

        collected = 0
        while collected < CALIBRATION_FRAMES:
            synced = self.sync.get_synced_frames()
            depth_data = synced.get(self.primary_depth_cam_name)
            if depth_data is None:
                cv2.waitKey(1)
                continue
            _color, depth_m, _ts = depth_data
            done = calibrator.collect_frame(depth_m)
            collected = len(calibrator.depth_frames)
            cv2.waitKey(1)

        try:
            calib_result = calibrator.fit_plane()
        except ValueError as e:
            print(f"[CALIBRATION] failed: {e}")
            return

        os.makedirs(self.calibrations_dir, exist_ok=True)
        path = self._calibration_path(self.session.surface)
        import json
        with open(path, "w") as f:
            json.dump(calib_result, f, indent=2)

        self.current_calibration = calib_result
        self.session.update_surface_calibration(calib_result)
        print(f"[CALIBRATION] saved to {path}")
        print(f"[CALIBRATION] also written into this session's metadata.json\n")

    def _track_hands(self, synced):
        """Run hand tracking on every camera present this frame.
        Returns {camera_name: hands_data}. Only the primary depth camera
        gets real metric depth; other cameras get landmarks/pixels only
        (fingertip depths = None) unless TRACK_HANDS_ON_RGB_ONLY_CAMERAS
        is False, in which case they're skipped entirely."""
        all_hands = {}
        for name, data in synced.items():
            cam_type = self.sync.camera_type(name)
            if cam_type == "depth":
                color, depth_m, _ts = data
                all_hands[name] = self.hand_tracker.process(color, depth_m=depth_m)
            elif TRACK_HANDS_ON_RGB_ONLY_CAMERAS:
                frame, _ts = data
                all_hands[name] = self.hand_tracker.process(frame, depth_m=None)
        return all_hands

    @staticmethod
    def _plane_distance_to_surface_m(plane_coefficients, avg_surface_depth_m, px, py, depth_m):
        """Same plane-equation math as generate_contact_labels.py's
        compute_surface_depth_at_pixel() + distance calc, minus the
        hysteresis/cooldown state machine -- deliberately simplified for
        a cheap per-frame live estimate. See LIVE_CONTACT_THRESHOLD_M's
        comment for why this is a live indicator, not a source of truth."""
        a, b, c, d = plane_coefficients
        if abs(c) < 1e-10:
            surface_depth = avg_surface_depth_m
        else:
            surface_depth = -(a * px + b * py + d) / c
            if surface_depth <= 0 or surface_depth > 1.0:
                surface_depth = avg_surface_depth_m
        return max(surface_depth - depth_m, 0.0)

    def _update_live_contact_hud(self, all_hands):
        """PATCH: called once per RECORDED frame. Counts, among the
        primary depth camera's fingertips this frame, how many are
        within LIVE_CONTACT_THRESHOLD_M of the calibrated surface, and
        folds that into both a rolling recent-window ratio and a
        whole-session ratio, so _show_preview() can display both."""
        if self.current_calibration is None:
            return
        hands_data = all_hands.get(self.primary_depth_cam_name)
        if not hands_data:
            return

        plane = self.current_calibration.get("plane_coefficients")
        avg_surface = self.current_calibration.get("avg_surface_depth_m")
        if plane is None or avg_surface is None:
            return

        frame_contact = 0
        frame_total = 0
        for hand in hands_data:
            for finger, depth_m in hand["fingertip_depths"].items():
                if depth_m is None or depth_m <= 0:
                    continue
                px, py = hand["fingertip_pixels"][finger]
                dist_m = self._plane_distance_to_surface_m(plane, avg_surface, px, py, depth_m)
                frame_total += 1
                if dist_m <= LIVE_CONTACT_THRESHOLD_M:
                    frame_contact += 1

        if frame_total == 0:
            return

        self._live_contact_window.append((frame_contact, frame_total))
        self._session_contact_total += frame_contact
        self._session_sample_total += frame_total

    def _live_contact_ratios(self):
        """Returns (recent_pct, session_pct), either None if no samples
        yet (e.g. calibration missing, or no hand visible so far)."""
        recent_pct = None
        if self._live_contact_window:
            c = sum(x[0] for x in self._live_contact_window)
            t = sum(x[1] for x in self._live_contact_window)
            if t > 0:
                recent_pct = 100.0 * c / t

        session_pct = None
        if self._session_sample_total > 0:
            session_pct = 100.0 * self._session_contact_total / self._session_sample_total

        return recent_pct, session_pct

    def _save_hand_annotations(self, frame_id, all_hands):
        """Write one annotation JSON per camera that has hand data this
        frame. Primary depth camera -> annotations/ (matches
        generate_contact_labels.py's expected location exactly). Every
        other camera -> annotations_{name}/ (additive, doesn't collide
        with existing tooling)."""
        import json

        for cam_name, hands_data in all_hands.items():
            if not hands_data:
                continue  

            if cam_name == self.primary_depth_cam_name:
                ann_dir = os.path.join(self.session.session_dir, "annotations")
            else:
                ann_dir = os.path.join(self.session.session_dir, f"annotations_{cam_name}")
            os.makedirs(ann_dir, exist_ok=True)

            frame_annotation = {
                "frame_id": frame_id,
                "num_hands": len(hands_data),
                "hands": [],
            }
            
            is_primary_depth = (cam_name == self.primary_depth_cam_name)

            for hand in hands_data:
                fingertip_keys = {}
                if is_primary_depth and self.keyboard_layout:
                    for finger, (px, py) in hand["fingertip_pixels"].items():
                        fingertip_keys[finger] = find_key_at_point(self.keyboard_layout, px, py)

                frame_annotation["hands"].append({
                    "handedness": hand["handedness"],
                    "fingertip_depths_m": hand["fingertip_depths"],  # may contain None values
                    "fingertip_pixels": {
                        k: list(v) for k, v in hand["fingertip_pixels"].items()
                    },
                    "fingertip_keys": fingertip_keys,  # {} if no keyboard layout loaded
                    "landmarks_px": hand["landmarks_px"],
                })

            with open(os.path.join(ann_dir, f"{frame_id:06d}.json"), "w") as f:
                json.dump(frame_annotation, f)


    def run(self):
        print("[RECORDER] starting cameras...")
        self.sync.start()

        self.session.start_new_session(config.CAMERAS)
        self._load_calibration_for_current_session()

        print("[CONTROLS] R = start/stop recording | N = new session | "
              "C = calibrate surface | A = change an RGB camera's angle | "
              "Q = quit\n")

        try:
            while True:
                synced = self.sync.get_synced_frames()

                
                all_hands = self._track_hands(synced)

                self._show_preview(synced, all_hands)

                if self.is_recording and self._all_cameras_present(synced):
                    self._save_frame_set(synced)
                    self._save_hand_annotations(self.frame_counter, all_hands)
                    self._update_live_contact_hud(all_hands)
                    self.frame_counter += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._toggle_recording()
                elif key == ord('n'):
                    self._start_new_session()
                elif key == ord('c'):
                    was_recording = self.is_recording
                    self.is_recording = False
                    if was_recording:
                        print("[RECORDING] paused for calibration")
                    self._run_calibration()
                elif key == ord('a'):
                    self._set_camera_angle()

        finally:
            was_recording = self.is_recording
            self.is_recording = False
            self.sync.stop()
            self.hand_tracker.close()
            cv2.destroyAllWindows()
            self._label_current_session(reason="quit")
            self._visualize_current_session(reason="quit")
            print("[RECORDER] stopped")

    def _all_cameras_present(self, synced):
        return len(synced) == len(config.CAMERAS)

    def _toggle_recording(self):
        self.is_recording = not self.is_recording
        self.frame_counter = 0
        state = "STARTED" if self.is_recording else "STOPPED"
        print(f"[RECORDING] {state}")

        if self.is_recording:
            self._live_contact_window.clear()
            self._session_contact_total = 0
            self._session_sample_total = 0
        else:
            recent_pct, session_pct = self._live_contact_ratios()
            if session_pct is not None:
                print(f"[RECORDING] this take's contact ratio: {session_pct:.0f}% "
                      f"of fingertip samples were in contact -- if this is close to "
                      f"0% or 100%, you probably didn't get enough of a mix of "
                      f"hovering and touching (see the recording guide).")

        
        if self.is_recording and self.current_calibration is None:
            print("\n" + "!" * 70)
            print("[RECORDING] WARNING: no surface calibration loaded for this "
                  "session's surface + camera angle.")
            print("[RECORDING] generate_contact_labels.py will SKIP this "
                  "session entirely -- no has_contact labels will be produced.")
            print("[RECORDING] Press 'r' to stop, then 'c' to calibrate "
                  "(empty desk), then 'r' again to resume recording.")
            print("!" * 70 + "\n")

    def _start_new_session(self):
        was_recording = self.is_recording
        self.is_recording = False
        if was_recording:
            print("[RECORDING] stopped (new session requested)")


        self._label_current_session(reason="new session")
        self._visualize_current_session(reason="new session")

        self.session.start_new_session(config.CAMERAS)
        self._load_calibration_for_current_session()

    def _label_current_session(self, reason=""):

        if self.session.session_dir is None:
            return
        if not os.path.exists(self.contact_labeler_script):
            print(f"[LABELER] skipping auto-label ({reason}) -- script not found at "
                  f"{self.contact_labeler_script}")
            return

        ann_dir = os.path.join(self.session.session_dir, "annotations")
        if not os.path.isdir(ann_dir) or not os.listdir(ann_dir):
            print(f"[LABELER] skipping auto-label ({reason}) -- no annotations "
                  f"were recorded in that session.")
            return

        print(f"\n[LABELER] running generate_contact_labels.py on "
              f"{self.session.session_dir} ({reason})...")
        result = subprocess.run(
            [sys.executable, self.contact_labeler_script,
             "--session-dir", self.session.session_dir],
            check=False,
        )
        if result.returncode != 0:
            print(f"[LABELER] generate_contact_labels.py exited with code "
                  f"{result.returncode} -- see its output above for details.")
        print("[LABELER] done\n")

    def _visualize_current_session(self, reason=""):
        """Auto-colorizes the depth frames of the session that's ending.
        Safe no-op if nothing was ever recorded in it."""
        if self.session.session_dir is None:
            return

        print(f"[VISUALIZE] converting depth frames for previous session "
              f"({reason})...")
        stats = visualize_session_depth(self.session.session_dir)

        if stats.get("skipped"):
            print("[VISUALIZE] no depth frames recorded in that session, skipping")
            return

        print(f"[VISUALIZE] {stats['converted']} frame(s) converted -> "
              f"{stats['out_dir']}")
        if stats["empty_frames"] > 0:
            print(f"[VISUALIZE] WARNING: {stats['empty_frames']} frame(s) "
                  f"were all-zero depth — camera may have lost tracking "
                  f"or been occluded at that moment. Worth a look.")

    def _save_frame_set(self, synced):
        frame_id = str(self.frame_counter).zfill(6)

        for name, data in synced.items():
            cam_type = self.sync.camera_type(name)
            if cam_type == "depth":
                color, depth_m, _ts = data
                self._save_depth_frame(name, frame_id, color, depth_m)
            else:
                frame, _ts = data
                self._save_rgb_frame(name, frame_id, frame)

    def _save_depth_frame(self, name, frame_id, color, depth_m):
        rgb_dir = os.path.join(self.session.session_dir, f"rgb_{name}")
        depth_dir = os.path.join(self.session.session_dir, "depth")

        cv2.imwrite(os.path.join(rgb_dir, f"{frame_id}.png"), color)

        depth_mm = (depth_m * 1000.0).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_dir, f"{frame_id}.png"), depth_mm)

    def _save_rgb_frame(self, name, frame_id, frame):
        rgb_dir = os.path.join(self.session.session_dir, f"rgb_{name}")
        cv2.imwrite(os.path.join(rgb_dir, f"{frame_id}.png"), frame)

    def _show_preview(self, synced, all_hands=None):
        all_hands = all_hands or {}
        for name, data in synced.items():
            cam_type = self.sync.camera_type(name)
            frame = data[0]  # color frame for both depth-cam and rgb-cam tuples

            display = frame.copy()
            status = "REC" if self.is_recording else "preview"
            color = (0, 0, 255) if self.is_recording else (0, 255, 0)
            cv2.putText(display, f"{name} ({cam_type}) - {status}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if self.is_recording:
                cv2.putText(display, f"frame {self.frame_counter}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if name == self.primary_depth_cam_name:
                    recent_pct, session_pct = self._live_contact_ratios()
                    if recent_pct is not None:
                        hud_color = (0, 255, 0) if 15 <= recent_pct <= 85 else (0, 255, 255)
                        cv2.putText(display, f"contact (last ~5s): {recent_pct:.0f}%  "
                                              f"(session: {session_pct:.0f}%)",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1)

                if self.current_calibration is None:
                    h, w = display.shape[:2]
                    cv2.rectangle(display, (0, h - 30), (w, h), (0, 0, 200), -1)
                    cv2.putText(display, "NO CALIBRATION -- has_contact labels will NOT be generated",
                                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)

            is_primary_depth = (name == self.primary_depth_cam_name)
            for hand in all_hands.get(name, []):
                has_depth_source = any(d is not None for d in hand["fingertip_depths"].values())
                dot_color = (0, 255, 0) if has_depth_source else (0, 255, 255)
                for finger, (px, py) in hand["fingertip_pixels"].items():
                    cv2.circle(display, (px, py), 5, dot_color, -1)
                    depth_val = hand["fingertip_depths"].get(finger)
                    label_y_offset = -5
                    if depth_val is not None and depth_val > 0:
                        cv2.putText(display, f"{depth_val * 1000:.0f}mm", (px + 8, py - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, dot_color, 1)
                        label_y_offset -= 14

                    if is_primary_depth and self.keyboard_layout:
                        key_here = find_key_at_point(self.keyboard_layout, px, py)
                        if key_here is not None:
                            cv2.putText(display, key_here, (px + 8, py + label_y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 2)

            cv2.imshow(name, display)