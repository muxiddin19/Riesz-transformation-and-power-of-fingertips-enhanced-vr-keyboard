"""
config.py
=========
All the knobs live here. Edit this file when you add/remove a camera,
add a new student, or change the desk setup — nothing else should need
to change.
"""

# ------------------------------------------------------------------
# Participants (students). You can also just type a new ID at runtime,
# this list is only used to show a quick-pick menu.
# ------------------------------------------------------------------
PARTICIPANTS = [f"P{str(i).zfill(2)}" for i in range(1, 16)]  # P01 .. P15

# ------------------------------------------------------------------
# Surfaces (desk types). Same idea — quick-pick menu, free text allowed.
# ------------------------------------------------------------------
SURFACES = [
    "white_desk",
    "wood_grain",
    "semi_reflective",
]

# ------------------------------------------------------------------
# Camera rig. One camera MUST be "depth" type (the D405). All others
# are "rgb" type — they can be a plain webcam index (int) or a DroidCam
# / IP-camera URL (str), e.g. "http://192.168.0.23:4747/video".
#
# "angle" is just metadata saved into metadata.json / used in filenames,
# it does not rotate anything automatically — position the camera
# physically at that angle yourself.
# ------------------------------------------------------------------
CAMERAS = [
    {
        "name": "depth_cam",
        "type": "depth",
        "source": 0,          # RealSense picks the first D405 it finds
        "angle": 45,
    },
    {
        "name": "phone_droidcam",
        "type": "rgb",
        # a device NAME substring, not an index — DirectShow indices
        # shift once the D405 has been opened, so we resolve this to
        # whatever index it currently is at start() time instead.
        "source": "DroidCam",
        "backend": "dshow",
        "angle": 30,
    },
    # To add a second phone or a real webcam later, copy the block above.
    # If it's a plain USB webcam (not DroidCam), "backend": "dshow" is usually best.
]

# ------------------------------------------------------------------
# Capture settings
# ------------------------------------------------------------------
TARGET_FPS = 30

# Resolution for plain RGB cameras (webcam / DroidCam). Most webcams
# are fine at 640x480.
RGB_WIDTH = 640
RGB_HEIGHT = 480

# Resolution for the D405 specifically. Some D405 firmware/driver
# combos hang (not error, just hang) on 640x480 — 848x480 is the
# resolution RealSense Viewer defaults to and is known-solid. Confirm
# in RealSense Viewer what resolution streams cleanly for you before
# changing this.
DEPTH_CAM_WIDTH = 848
DEPTH_CAM_HEIGHT = 480

# PATCH: D405 valid working range for hand capture, in meters. Used by
# camera_depth.py's post-processing filter chain (threshold filter) to
# clamp/reject depth outside this range, and by anything else that needs
# to know "what counts as a plausible hand-to-camera distance." Matches
# the values already proven in custom_data_recording/record.py.
DEPTH_MIN_M = 0.07
DEPTH_MAX_M = 0.50

# how far apart (seconds) two frames from different cameras can be and
# still count as "the same synced frame"
SYNC_TOLERANCE_SEC = 0.08

# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------
OUTPUT_ROOT = "./data"

# PATCH: Stage 1 automation -- after each session ends (new session or
# quit), recorder.py shells out to generate_contact_labels.py so
# has_contact labels are produced automatically. This path is relative
# to data_recording/ (where recorder.py lives). Only set this if your
# folder layout differs from the default:
#   <project_root>/data_recording/        <- recorder.py, this file
#   <project_root>/custom_data_recording/ <- generate_contact_labels.py
# CONTACT_LABELER_SCRIPT_RELPATH = "../custom_data_recording/generate_contact_labels.py"

# PATCH: which-key-was-touched labeling for the PRIMARY depth camera
# (the D405) only -- see recorder.py/keyboard_layout.py for why this is
# deliberately scoped to just that one, permanently-fixed camera. Path
# is relative to data_recording/. Produced by your existing click-4-
# corners annotation tool; run it once against the D405 at its
# permanent mount and this stays valid for the rest of the project.
# KEYBOARD_ANNOTATIONS_RELPATH = "../assets/keyboard_annotations.json"