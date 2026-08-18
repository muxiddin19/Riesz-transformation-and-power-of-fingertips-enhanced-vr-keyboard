"""
camera_depth.py
================
Thin wrapper around an Intel RealSense D405. Runs its own grab thread
and always exposes the LATEST (color, depth, timestamp) triple so the
sync manager can just peek at it whenever it wants.

PATCH: added the same RealSense post-processing filter chain
(decimation -> spatial -> temporal -> hole-filling -> threshold) that
custom_data_recording/record.py already uses and is proven to work well
for close-range hand capture. Previously this file returned RAW aligned
depth with no filtering at all -- fine for far-field/static scenes, but
at 15-40cm (hand-to-camera range) raw D405 depth is genuinely noisy and
full of holes, which is what was showing up as a mostly-blank/speckled
depth_visualized/ image with almost no recognizable hand shape. This
filter chain, plus clamping to a sane working range (DEPTH_MIN_M /
DEPTH_MAX_M, now in config.py), should make the hand actually visible
in depth and make per-fingertip depth sampling much more reliable for
both the live preview and generate_contact_labels.py.
"""

import threading
import time

import numpy as np
import pyrealsense2 as rs

import config


class DepthFilter:
    """Multi-stage depth filtering for clean close-range hand capture.

    Ported directly from custom_data_recording/record.py's DepthFilter --
    same filter chain, same tuning, since it's already proven on this
    exact camera (D405) at this exact range (hands 15-40cm away).
    """

    # Tuned for close-range hand capture -- see record.py for the same values.
    SPATIAL_MAGNITUDE = 2
    SPATIAL_SMOOTH_ALPHA = 0.5
    SPATIAL_SMOOTH_DELTA = 20
    TEMPORAL_SMOOTH_ALPHA = 0.4
    TEMPORAL_SMOOTH_DELTA = 20
    HOLE_FILL_MODE = 1  # farthest from around

    def __init__(self, min_depth_m, max_depth_m):
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 1)  # no decimation, keep resolution

        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, self.SPATIAL_MAGNITUDE)
        self.spatial.set_option(rs.option.filter_smooth_alpha, self.SPATIAL_SMOOTH_ALPHA)
        self.spatial.set_option(rs.option.filter_smooth_delta, self.SPATIAL_SMOOTH_DELTA)
        self.spatial.set_option(rs.option.holes_fill, self.HOLE_FILL_MODE)

        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, self.TEMPORAL_SMOOTH_ALPHA)
        self.temporal.set_option(rs.option.filter_smooth_delta, self.TEMPORAL_SMOOTH_DELTA)

        self.hole_fill = rs.hole_filling_filter()
        self.hole_fill.set_option(rs.option.holes_fill, self.HOLE_FILL_MODE)

        self.threshold = rs.threshold_filter()
        self.threshold.set_option(rs.option.min_distance, min_depth_m)
        self.threshold.set_option(rs.option.max_distance, max_depth_m)

    def apply(self, depth_frame):
        """Apply full filter chain: threshold -> spatial -> temporal -> hole fill."""
        frame = depth_frame
        frame = self.threshold.process(frame)
        frame = self.spatial.process(frame)
        frame = self.temporal.process(frame)
        frame = self.hole_fill.process(frame)
        return frame


class DepthCamera:
    def __init__(self, name: str, width: int = 848, height: int = 480, fps: int = 30):
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.align = None
        self.depth_scale = None

        # PATCH: post-processing filter chain, built once depth_scale is
        # known isn't actually required (thresholds are in meters,
        # independent of depth_scale), so it's safe to build in __init__.
        self.depth_filter = DepthFilter(
            min_depth_m=getattr(config, "DEPTH_MIN_M", 0.07),
            max_depth_m=getattr(config, "DEPTH_MAX_M", 0.50),
        )

        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._latest = None  # (color, depth_meters, timestamp)

    def start(self):
        self._reset_device_if_present()
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # PATCH: D405 short-range visual preset -- also proven in
        # record.py, tunes the sensor's own stereo-matching for
        # close-range accuracy rather than relying on filtering alone.
        try:
            depth_sensor.set_option(rs.option.visual_preset, 4)
            print(f"[{self.name}] set short-range visual preset")
        except Exception as e:
            print(f"[{self.name}] could not set visual preset: {e}")

        # let auto-exposure settle before we start trusting frames
        for _ in range(30):
            self.pipeline.wait_for_frames()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[{self.name}] started (depth_scale={self.depth_scale:.6f})")

    def _reset_device_if_present(self):
        """Hardware-reset the D405 before opening it. Clears any stuck
        firmware state left over from a crash, an unclean exit, or
        another app (RealSense Viewer) having held the camera open."""
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            return
        print(f"[{self.name}] resetting device before start...")
        devices[0].hardware_reset()
        time.sleep(5)

    def _capture_loop(self):
        consecutive_failures = 0
        warned_at_failure_count = 0

        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                consecutive_failures += 1
                if consecutive_failures in (5, 20, 100) and consecutive_failures != warned_at_failure_count:
                    warned_at_failure_count = consecutive_failures
                    print(f"[{self.name}] {consecutive_failures} consecutive frame timeouts — "
                          f"camera may have disconnected or stalled.")
                time.sleep(min(1.0, 0.1 * consecutive_failures))
                continue

            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # PATCH: apply the filter chain to the ALIGNED depth frame,
            # same order as record.py (align first, then filter).
            depth_frame = self.depth_filter.apply(depth_frame)

            consecutive_failures = 0
            warned_at_failure_count = 0

            color = np.asanyarray(color_frame.get_data())
            depth_m = np.asanyarray(depth_frame.get_data()) * self.depth_scale
            ts = time.time()

            with self._lock:
                self._latest = (color, depth_m, ts)

    def get_latest(self):
        """Returns (color, depth_meters, timestamp) or None if nothing yet."""
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.pipeline.stop()
        print(f"[{self.name}] stopped")