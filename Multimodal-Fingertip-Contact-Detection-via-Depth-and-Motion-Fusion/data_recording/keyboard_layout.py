

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