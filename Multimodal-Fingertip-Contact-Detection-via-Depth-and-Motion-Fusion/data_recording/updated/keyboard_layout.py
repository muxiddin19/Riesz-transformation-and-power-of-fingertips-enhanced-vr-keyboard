

import json
import os

import numpy as np
import cv2


def load_keyboard_layout(path):

    if not path or not os.path.exists(path):
        return []

    with open(path) as f:
        data = json.load(f)

    keys = []
    for item in data:
        if "key" not in item or "points" not in item or len(item["points"]) != 4:
            continue
        corners = np.array([[p["x"], p["y"]] for p in item["points"]], dtype=np.float32)
        keys.append({"key": item["key"], "corners": corners})
    return keys


def draw_keyboard_overlay(frame, keys):
    """Draws each key's quadrilateral outline + label on frame in place,
    same green-outline/label style as src/keyboard_annotation.py's
    annotation tool. Returns frame for convenient chaining."""
    for key in keys:
        pts = key["corners"].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        x, y = key["corners"][0]
        cv2.putText(frame, key["key"], (int(x) + 5, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


def find_key_at_point(keys, x, y, margin_px=0.0):

    if not keys:
        return None

    point = (float(x), float(y))
    for key in keys:
        corners = key["corners"]
        if margin_px:
            center = corners.mean(axis=0)
            direction = corners - center
            norms = np.linalg.norm(direction, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            corners = corners + (direction / norms) * margin_px
        if cv2.pointPolygonTest(corners, point, False) >= 0:
            return key["key"]
    return None