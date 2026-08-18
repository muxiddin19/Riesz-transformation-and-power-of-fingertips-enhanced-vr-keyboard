import numpy as np
import cv2
import mediapipe as mp


class HandDepthTracker:
    """MediaPipe hand tracking with fringe-safe depth sampling.

    See module docstring: ported from custom_data_recording/record.py,
    only change is depth_m is now optional.
    """

    FINGERTIP_IDS = {
        "thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20
    }

    def __init__(self, max_hands=2, min_detection_confidence=0.7,
                 min_depth_m=0.07, max_depth_m=0.50):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

    def process(self, rgb, depth_m=None):

        rgb_input = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_input)

        if not result.multi_hand_landmarks:
            return []

        h, w = rgb.shape[:2]

        if depth_m is not None and depth_m.shape[:2] != (h, w):
            raise ValueError(
                f"depth_m shape {depth_m.shape[:2]} does not match rgb "
                f"shape {(h, w)} -- depth must already be aligned to this "
                f"camera's color frame before calling process()."
            )

        hands_data = []

        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            handedness = "Unknown"
            if result.multi_handedness and i < len(result.multi_handedness):
                handedness = result.multi_handedness[i].classification[0].label

            landmarks_px = []
            for lm in hand_landmarks.landmark:
                px = int(np.clip(lm.x * w, 0, w - 1))
                py = int(np.clip(lm.y * h, 0, h - 1))
                landmarks_px.append((px, py))

            mask = None
            if depth_m is not None:
                mask = self._make_eroded_mask(landmarks_px, h, w)

            tips = {}
            tips_px = {}
            for name, idx in self.FINGERTIP_IDS.items():
                px, py = landmarks_px[idx]
                tips_px[name] = (px, py)
                if depth_m is not None:
                    tips[name] = self._sample_depth(depth_m, mask, px, py)
                else:
                    tips[name] = None  # no depth source for this camera yet

            hands_data.append({
                "landmarks_px": landmarks_px,
                "fingertip_depths": tips,
                "fingertip_pixels": tips_px,
                "handedness": handedness,
            })

        return hands_data

    def _make_eroded_mask(self, landmarks_px, h, w):
        """Convex hull mask eroded by ~3px to avoid the depth-edge fringe."""
        points = np.array(landmarks_px)
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.erode(mask, kernel, iterations=1)

    def _sample_depth(self, depth_m, eroded_mask, px, py):
        """3x3 median if inside the eroded (safe) mask, 5x5 median if the
        landmark landed on the fringe -- avoids depth-edge bleed."""
        h, w = depth_m.shape
        px = int(np.clip(px, 3, w - 4))
        py = int(np.clip(py, 3, h - 4))

        if eroded_mask[py, px] == 0:
            patch = depth_m[py - 2:py + 3, px - 2:px + 3]
        else:
            patch = depth_m[py - 1:py + 2, px - 1:px + 2]

        valid = patch[(patch > self.min_depth_m) & (patch < self.max_depth_m)]
        return float(np.median(valid)) if len(valid) > 0 else 0.0

    def close(self):
        self.hands.close()


class SurfaceCalibrator:
    
    def __init__(self, num_frames=30):
        self.num_frames = num_frames
        self.depth_frames = []

    def collect_frame(self, depth_m):
        self.depth_frames.append(depth_m.copy())
        return len(self.depth_frames) >= self.num_frames

    def fit_plane(self):
        if len(self.depth_frames) < self.num_frames:
            raise ValueError(f"Need {self.num_frames} frames, got {len(self.depth_frames)}")

        avg_depth = np.mean(self.depth_frames, axis=0)

        ys, xs = np.where(avg_depth > 0)
        if len(ys) < 100:
            raise ValueError("Not enough valid depth points for plane fitting")

        idx = np.random.choice(len(ys), min(10000, len(ys)), replace=False)
        points = np.column_stack([xs[idx], ys[idx], avg_depth[ys[idx], xs[idx]]])

        best_plane = None
        best_inliers = 0
        # NOTE: kept identical to the original (0.008 = 8mm). The original
        # inline comment said "2mm", which was just a stale/incorrect
        # comment -- the value itself is unchanged here since it's proven
        # in production; fixing the comment only.
        inlier_threshold = 0.008  # 8mm

        for _ in range(1000):
            sample_idx = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[sample_idx]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            d = -np.dot(normal, p1)

            distances = np.abs(points @ normal + d)
            n_inliers = np.sum(distances < inlier_threshold)

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_plane = np.append(normal, d)

        if best_plane is None:
            raise ValueError("RANSAC failed to fit plane")

        inlier_ratio = best_inliers / len(points)
        print(f"Plane fit: {best_inliers}/{len(points)} inliers ({inlier_ratio:.1%})")
        print(f"Plane equation: {best_plane[0]:.4f}x + {best_plane[1]:.4f}y + "
              f"{best_plane[2]:.4f}z + {best_plane[3]:.4f} = 0")

        return {
            "plane_coefficients": best_plane.tolist(),
            "inlier_ratio": float(inlier_ratio),
            "avg_surface_depth_m": float(np.median(avg_depth[avg_depth > 0])),
            "num_frames_used": len(self.depth_frames),
        }