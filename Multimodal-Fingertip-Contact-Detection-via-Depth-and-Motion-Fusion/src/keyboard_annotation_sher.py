"""
Keyboard Annotation Tool - DroidCam Version
============================================
Adapted for phone camera via DroidCam (no RealSense / CameraManager needed).
All original logic preserved: zoom, pan, 4-point annotation, JSON export.

Usage:
    1. Make sure DroidCam is connected and showing feed (index 1, CAP_MSMF)
    2. Run this script
    3. Press 'c' to capture a frame
    4. Click 4 corners around each key
    5. Type the key name and press ENTER
    6. Press 's' to save to assets/keyboard_annotations.json
    7. Press 'q' to quit
"""

import numpy as np
import cv2
import json
import os

# --- Configuration ---
CAMERA_ID     = 1              # DroidCam virtual camera index
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30
CAMERA_BACKEND = cv2.CAP_MSMF  # Required for DroidCam on Windows

# --- Global variables for annotation ---
annotations       = []
current_raw_frame = None
window_name       = 'Keyboard Annotation Tool'
output_filename = 'assets/keyboard_annotations.json'


# Variables to manage the 4-point annotation process
temp_key_points = []
POINTS_PER_KEY  = 4

# --- Zoom and Pan Variables ---
zoom_factor     = 1.0
pan_x           = 0
pan_y           = 0
pan_speed       = 20
max_zoom_factor = 5.0
min_zoom_factor = 1.0


# --- Load existing annotations if file exists ---
def load_annotations(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
                validated = []
                for item in data:
                    if 'key' in item and 'points' in item and len(item['points']) == POINTS_PER_KEY:
                        validated.append(item)
                    else:
                        print(f"Warning: Skipping malformed entry: {item}")
                print(f"Loaded {len(validated)} existing key(s) from {filename}")
                return validated
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {filename}. Starting fresh.")
                return []
    return []


# Start fresh (swap the line below to load existing annotations instead)
annotations = []
# annotations = load_annotations(output_filename)


# --- Mouse callback function for annotations ---
def mouse_callback(event, x, y, flags, param):
    global annotations, current_raw_frame, temp_key_points, zoom_factor, pan_x, pan_y

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_raw_frame is None:
            
            print("No frame captured yet. Press 'c' first.")
            return

        # Transform clicked coordinates from zoomed view back to original frame

        rect = cv2.getWindowImageRect(window_name)
        win_w, win_h = rect[2], rect[3]

        norm_x = x * (CAMERA_WIDTH / win_w)
        norm_y = y * (CAMERA_HEIGHT / win_h)
        original_x = int(pan_x + norm_x / zoom_factor)
        original_y = int(pan_y + norm_y / zoom_factor)

        # Clamp to frame boundaries
        original_x = max(0, min(original_x, CAMERA_WIDTH - 1))
        original_y = max(0, min(original_y, CAMERA_HEIGHT - 1))

        temp_key_points.append({'x': original_x, 'y': original_y})
        print(f"  Point {len(temp_key_points)}/{POINTS_PER_KEY} -> original coords: ({original_x}, {original_y})")

        draw_current_frame_with_annotations()

        if len(temp_key_points) == POINTS_PER_KEY:
            key_value = show_input_box("Enter key value (e.g. A, space, enter, B.Spa):")

            if key_value:
                annotations.append({'key': key_value, 'points': temp_key_points.copy()})
                print(f"  [SAVED] Key='{key_value}' with {POINTS_PER_KEY} points.")
            else:
                print("  Annotation cancelled. Points reset.")

            temp_key_points = []
            draw_current_frame_with_annotations()


# --- Custom OpenCV input box ---
def show_input_box(prompt):
    """Simple text input box rendered with OpenCV."""
    input_text  = ""
    box_w, box_h = 450, 160
    box_img = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    box_img[:] = (50, 50, 50)

    cv2.putText(box_img, "Input Required",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(box_img, prompt,
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    while True:
        display = box_img.copy()
        cv2.putText(display, f"Key: {input_text}_",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)
        cv2.putText(display, "ENTER = confirm    ESC = cancel",
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow("Input Box", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:       # Enter
            cv2.destroyWindow("Input Box")
            return input_text if input_text else None
        elif key == 27:     # ESC
            cv2.destroyWindow("Input Box")
            return None
        elif key == 8:      # Backspace
            input_text = input_text[:-1]
        elif 32 <= key <= 126:
            input_text += chr(key)


# --- Apply zoom and pan to a frame ---
def apply_zoom_and_pan(frame):
    if frame is None:
        return None

    view_w = int(CAMERA_WIDTH  / zoom_factor)
    view_h = int(CAMERA_HEIGHT / zoom_factor)

    # Clamp pan so we never go out of bounds
    safe_pan_x = max(0, min(pan_x, CAMERA_WIDTH  - view_w))
    safe_pan_y = max(0, min(pan_y, CAMERA_HEIGHT - view_h))

    cropped = frame[safe_pan_y : safe_pan_y + view_h,
                    safe_pan_x : safe_pan_x + view_w]
    zoomed  = cv2.resize(cropped, (CAMERA_WIDTH, CAMERA_HEIGHT),
                         interpolation=cv2.INTER_LINEAR)
    return zoomed


# --- Draw annotations + UI text on the display frame ---
def draw_current_frame_with_annotations():
    global current_raw_frame, temp_key_points, annotations, zoom_factor, pan_x, pan_y

    if current_raw_frame is None:
        return

    display = current_raw_frame.copy()

    # Draw all saved annotations in green
    for annotation in annotations:
        transformed = []
        for p in annotation['points']:
            tx = int((p['x'] - pan_x) * zoom_factor)
            ty = int((p['y'] - pan_y) * zoom_factor)
            transformed.append({'x': tx, 'y': ty})

        for p in transformed:
            cv2.circle(display, (p['x'], p['y']), 5, (0, 255, 0), -1)

        if len(transformed) == POINTS_PER_KEY:
            pts = np.array([[p['x'], p['y']] for p in transformed], np.int32)
            cv2.polylines(display, [pts.reshape((-1, 1, 2))], True, (0, 255, 0), 2)

        if transformed:
            cv2.putText(display, annotation['key'],
                        (transformed[0]['x'] + 5, transformed[0]['y'] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw temporary (in-progress) points in orange
    for i, p in enumerate(temp_key_points):
        tx = int((p['x'] - pan_x) * zoom_factor)
        ty = int((p['y'] - pan_y) * zoom_factor)
        cv2.circle(display, (tx, ty), 6, (0, 165, 255), -1)
        cv2.putText(display, str(i + 1),
                    (tx + 7, ty - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        # Draw connecting lines between temp points
        if i > 0:
            px = int((temp_key_points[i-1]['x'] - pan_x) * zoom_factor)
            py = int((temp_key_points[i-1]['y'] - pan_y) * zoom_factor)
            cv2.line(display, (px, py), (tx, ty), (0, 165, 255), 1)

    # Apply zoom/pan to the annotated frame
    final = apply_zoom_and_pan(display)
    if final is None:
        return

    # Overlay HUD text
    key_count = len(annotations)
    point_count = len(temp_key_points)

    cv2.putText(final, f"Zoom: {zoom_factor:.1f}x  |  Keys annotated: {key_count}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

    if point_count > 0:
        cv2.putText(final, f"Clicking point {point_count + 1} / {POINTS_PER_KEY}...",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA)

    cv2.putText(final, "c=Capture  s=Save  r=Reset  +/-=Zoom  Arrows=Pan  q=Quit",
                (10, CAMERA_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(window_name, final)


# ============================================================
# Main
# ============================================================
def main():
    global current_raw_frame, temp_key_points, zoom_factor, pan_x, pan_y, annotations

    # Make sure assets folder exists
    os.makedirs('assets', exist_ok=True)

    # --- Open DroidCam ---
    print(f"[CAMERA] Opening camera index {CAMERA_ID} with CAP_MSMF...")
    cap = cv2.VideoCapture(CAMERA_ID, CAMERA_BACKEND)

    if not cap.isOpened():
        print("[ERROR] Could not open camera. Check DroidCam is connected and running.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    print(f"[OK] Camera opened at {CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS}fps")
    print()
    print("  Instructions:")
    print(f"    c        → Capture a frame to annotate")
    print(f"    click x4 → Mark 4 corners of a key, then type its name")
    print(f"    s        → Save annotations to '{output_filename}'")
    print(f"    +/-      → Zoom in / out")
    print(f"    Arrows   → Pan the zoomed view")
    print(f"    r        → Reset zoom and pan")
    print(f"    q        → Quit")
    print()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    is_live_view = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to grab frame, retrying...")
                continue

            current_raw_frame = frame

            if is_live_view:
                draw_current_frame_with_annotations()

            key = cv2.waitKey(1) & 0xFF

            # --- Quit ---
            if key == ord('q') or key == 27:
                print("Exiting.")
                break

            # --- Capture frame ---
            elif key == ord('c'):
                is_live_view = False
                temp_key_points = []
                draw_current_frame_with_annotations()
                print(f"[CAPTURED] Click {POINTS_PER_KEY} points to annotate a key.")

            # --- Resume live view ---
            elif key == ord('v'):
                is_live_view = True
                temp_key_points = []
                print("[LIVE] Resumed live view.")

            # --- Save ---
            elif key == ord('s'):
                if annotations:
                    with open(output_filename, 'w') as f:
                        json.dump(annotations, f, indent=4)
                    print(f"[SAVED] {len(annotations)} keys → '{output_filename}'")
                else:
                    print("[WARN] No annotations to save yet.")

            # --- Reset zoom/pan ---
            elif key == ord('r'):
                zoom_factor = 1.0
                pan_x, pan_y = 0, 0
                temp_key_points = []
                print("[RESET] Zoom and pan reset.")
                draw_current_frame_with_annotations()

            # --- Undo last annotation ---
            elif key == ord('u'):
                if annotations:
                    removed = annotations.pop()
                    print(f"[UNDO] Removed key: '{removed['key']}'")
                    draw_current_frame_with_annotations()
                else:
                    print("[UNDO] Nothing to undo.")

            # --- Zoom in ---
            elif key == ord('+') or key == ord('='):
                old_vw = CAMERA_WIDTH  / zoom_factor
                old_vh = CAMERA_HEIGHT / zoom_factor
                zoom_factor = min(max_zoom_factor, zoom_factor + 0.2)
                new_vw = CAMERA_WIDTH  / zoom_factor
                new_vh = CAMERA_HEIGHT / zoom_factor
                pan_x += int((old_vw - new_vw) / 2)
                pan_y += int((old_vh - new_vh) / 2)
                print(f"[ZOOM] {zoom_factor:.1f}x")
                draw_current_frame_with_annotations()

            # --- Zoom out ---
            elif key == ord('-'):
                old_vw = CAMERA_WIDTH  / zoom_factor
                old_vh = CAMERA_HEIGHT / zoom_factor
                zoom_factor = max(min_zoom_factor, zoom_factor - 0.2)
                new_vw = CAMERA_WIDTH  / zoom_factor
                new_vh = CAMERA_HEIGHT / zoom_factor
                pan_x -= int((new_vw - old_vw) / 2)
                pan_y -= int((new_vh - old_vh) / 2)
                print(f"[ZOOM] {zoom_factor:.1f}x")
                draw_current_frame_with_annotations()

            # --- Pan with arrow keys ---
            elif key == 82:  # Up
                pan_y = max(0, pan_y - pan_speed)
                draw_current_frame_with_annotations()
            elif key == 84:  # Down
                max_py = CAMERA_HEIGHT - int(CAMERA_HEIGHT / zoom_factor)
                pan_y = min(max_py, pan_y + pan_speed)
                draw_current_frame_with_annotations()
            elif key == 81:  # Left
                pan_x = max(0, pan_x - pan_speed)
                draw_current_frame_with_annotations()
            elif key == 83:  # Right
                max_px = CAMERA_WIDTH - int(CAMERA_WIDTH / zoom_factor)
                pan_x = min(max_px, pan_x + pan_speed)
                draw_current_frame_with_annotations()

    finally:
        print("[CLEANUP] Releasing camera and closing windows.")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()