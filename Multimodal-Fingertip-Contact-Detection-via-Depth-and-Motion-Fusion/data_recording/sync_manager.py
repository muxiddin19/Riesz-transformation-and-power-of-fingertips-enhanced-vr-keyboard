
import time

from camera_depth import DepthCamera
from camera_rgb import RGBCamera


class MultiCameraSync:
    def __init__(self, camera_configs, rgb_width, rgb_height, depth_width, depth_height,
                 fps, sync_tolerance_sec):
        self.camera_configs = camera_configs
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.fps = fps
        self.sync_tolerance_sec = sync_tolerance_sec
        self.cameras = {}  # name -> DepthCamera or RGBCamera

    def start(self):
        
        ordered_configs = sorted(self.camera_configs, key=lambda c: c["type"] == "depth")

        for cam_cfg in ordered_configs:
            cam = self._build_camera(cam_cfg)
            cam.start()
            self.cameras[cam_cfg["name"]] = cam

        
        time.sleep(0.5)

    def _build_camera(self, cam_cfg):
        if cam_cfg["type"] == "depth":
            return DepthCamera(cam_cfg["name"], self.depth_width, self.depth_height, self.fps)
        backend = cam_cfg.get("backend", "auto")
        return RGBCamera(cam_cfg["name"], cam_cfg["source"], self.rgb_width, self.rgb_height,
                          self.fps, backend)

    def get_synced_frames(self):
        
        raw = {}
        for name, cam in self.cameras.items():
            latest = cam.get_latest()
            if latest is not None:
                raw[name] = latest

        if not raw:
            return {}

        timestamps = [data[-1] for data in raw.values()]
        newest_ts = max(timestamps)

        synced = {}
        for name, data in raw.items():
            ts = data[-1]
            if newest_ts - ts <= self.sync_tolerance_sec:
                synced[name] = data

        return synced

    def camera_type(self, name):
        for cam_cfg in self.camera_configs:
            if cam_cfg["name"] == name:
                return cam_cfg["type"]
        return None

    def camera_angle(self, name):
        for cam_cfg in self.camera_configs:
            if cam_cfg["name"] == name:
                return cam_cfg["angle"]
        return None

    def stop(self):
        for cam in self.cameras.values():
            cam.stop()