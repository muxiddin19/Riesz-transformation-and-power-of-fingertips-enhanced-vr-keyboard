import json
import os
import time

import config


class SessionManager:
    def __init__(self, output_root=config.OUTPUT_ROOT):
        self.output_root = output_root
        self.participant = None
        self.surface = None
        self.session_id = None
        self.session_dir = None
        self.camera_configs = None

    def start_new_session(self, camera_configs):
        """
        camera_configs: list of dicts, one per camera. Each dict should
        at minimum have 'name' (existing requirement). Optionally include
        calibration fields now, or fill them in later via
        update_calibration():

            {
                "name": "d405",
                "intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...},
                "depth_scale": 0.0001,          # meters per depth unit
                "depth_filter_params": {...},   # e.g. hole-filling, decimation settings
            }

        Surface plane coefficients (from RANSAC) are session-level, not
        per-camera, and go through update_surface_calibration().
        """
        self.participant = self._pick_participant()
        self.surface = self._pick_surface()
        self.session_id = self._make_session_id(self.participant, self.surface)
        self.session_dir = os.path.join(self.output_root, self.participant, self.session_id)
        self.camera_configs = camera_configs


        self._surface_calibration = None
        self._surface_plane = None

        self._make_folders(camera_configs)
        self._write_metadata()

        print(f"\n[SESSION] {self.participant} / {self.session_id}")
        print(f"[SESSION] saving to: {self.session_dir}\n")
        return self.session_dir

    def update_calibration(self, camera_name, intrinsics=None, depth_scale=None,
                            depth_filter_params=None, extrinsics_to=None):
    
        if self.camera_configs is None:
            raise RuntimeError("No active session — call start_new_session() first.")

        for cam_cfg in self.camera_configs:
            if cam_cfg["name"] == camera_name:
                if intrinsics is not None:
                    cam_cfg["intrinsics"] = intrinsics
                if depth_scale is not None:
                    cam_cfg["depth_scale"] = depth_scale
                if depth_filter_params is not None:
                    cam_cfg["depth_filter_params"] = depth_filter_params
                if extrinsics_to is not None:
                    cam_cfg.setdefault("extrinsics_to", {}).update(extrinsics_to)
                break
        else:
            raise ValueError(f"No camera named '{camera_name}' in this session's camera_configs.")

        self._write_metadata()

    def update_surface_plane(self, plane_coefficients, inlier_count=None, rmse=None):

        self._surface_plane = {
            "coefficients": list(plane_coefficients),
            "inlier_count": inlier_count,
            "rmse": rmse,
        }
        self._write_metadata()

    def update_surface_calibration(self, calib_result):

        if self.session_dir is None:
            raise RuntimeError("No active session — call start_new_session() first.")

        self._surface_calibration = calib_result
        self._write_metadata()

    def _pick_participant(self):
        print("\nParticipants:")
        for i, p in enumerate(config.PARTICIPANTS):
            print(f"  {i + 1}. {p}")
        print("  or type a new ID directly (e.g. P16)")

        choice = input("Select participant: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(config.PARTICIPANTS):
            return config.PARTICIPANTS[int(choice) - 1]
        if choice:
            return choice
        return "P_unknown"

    def _pick_surface(self):
        print("\nSurfaces:")
        for i, s in enumerate(config.SURFACES):
            print(f"  {i + 1}. {s}")
        print("  or type a new surface name directly")

        choice = input("Select surface: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(config.SURFACES):
            return config.SURFACES[int(choice) - 1]
        if choice:
            return choice
        return "unknown_surface"

    def _make_session_id(self, participant, surface):
        stamp = time.strftime("%H%M%S")
        return f"{participant}_{surface}_typing_{stamp}"

    def _make_folders(self, camera_configs):
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "annotations"), exist_ok=True)

        for cam_cfg in camera_configs:
            rgb_dir = os.path.join(self.session_dir, f"rgb_{cam_cfg['name']}")
            os.makedirs(rgb_dir, exist_ok=True)

    def _write_metadata(self):
        meta = {
            "participant": self.participant,
            "surface": self.surface,
            "session_id": self.session_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cameras": self.camera_configs,
            "surface_plane": getattr(self, "_surface_plane", None),
            # PATCH: this is the key generate_contact_labels.py actually reads.
            "surface_calibration": getattr(self, "_surface_calibration", None),
        }
        meta_path = os.path.join(self.session_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)