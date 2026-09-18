import os


PARTICIPANTS = [f"P{str(i).zfill(2)}" for i in range(1, 16)]  

SURFACES = [
    "white_desk",
    "wood_grain",
    "semi_reflective",
]

# NOTE on "source": RGB cameras below are resolved by DEVICE NAME
# substring, not raw index. This project got bitten twice by hardcoded
# int indices -- DirectShow enumeration order shifts (and devices drop
# out of it entirely) whenever something is plugged/unplugged/power-
# cycled, so a fixed int silently ends up pointing at whatever NOW
# occupies that slot (once, an unrelated already-open device; another
# time, DroidCam) instead of erroring out. Name-substring resolution
# (RGBCamera._resolve_name_to_index in camera_rgb.py, run fresh at each
# start() rather than baked in here) finds the right physical device
# every time regardless of current index, and raises a clear error if
# it's genuinely not connected instead of silently opening the wrong
# camera. Run list_camera_names.py any time you want to see the current
# raw enumeration for reference; last observed:
#   WCAM100                              <- cam1
#   Intel(R) RealSense(TM) Depth Camera 405  Depth   <- owned by depth_cam
#                                               via pyrealsense2 exclusively;
#                                               NEVER give an rgb camera a
#                                               source matching this name
#                                               (DirectShow/librealsense will
#                                               fight over the same USB
#                                               device and BOTH fail to open)
#   USB2.0 UVC PC Camera                 <- cam2
#   DroidCam Video                       <- unused (needs the DroidCam
#                                               Client app running/connected
#                                               on the phone first, and even
#                                               then just shows its "Start
#                                               DroidCam" placeholder image
#                                               instead of a live feed if the
#                                               phone-side app isn't actually
#                                               streaming -- switched cam3 to
#                                               a real webcam instead)
#   USB2.0 PC CAMERA                     <- cam3 (note: "USB2.0 PC CAMERA" is
#                                               NOT a substring of "USB2.0 UVC
#                                               PC Camera", so this stays
#                                               unambiguous even when both
#                                               USB webcams are connected)
CAMERAS = [
    {
        "name": "depth_cam",
        "type": "depth",
        "source": 0,  # unused -- DepthCamera always opens the D405 via pyrealsense2.pipeline(), not this
        "angle": 45,
    },
    {
        "name": "cam1",
        "type": "rgb",
        "source": "WCAM100",
        "backend": "dshow",
        "angle": 30,
    },
    {
        "name": "cam2",
        "type": "rgb",
        # PATCH: the original "USB2.0 UVC PC Camera" was swapped out for a
        # second unit of the SAME model as cam3 ("USB2.0 PC CAMERA") --
        # both now enumerate under the identical name. Name-substring
        # resolution can still guarantee cam2 and cam3 get two DIFFERENT
        # physical devices (via RGBCamera._claimed_indices) when both are
        # opened together in one process/session, since cam2 is built
        # before cam3 and claims the first match -- but it CANNOT
        # guarantee which physical unit that is consistently across
        # reboots/replugs, because the devices are literally
        # indistinguishable by name. If which physical camera plays
        # "cam2" (30 deg) vs "cam3" (90 deg) matters for your rig, label
        # the cables/ports yourself and don't rely on this to track
        # identity for you. Calibrating cam2 and cam3 one at a time (as
        # in the pipeline menu) is unaffected either way -- whichever one
        # opens just gets calibrated as itself.
        "source": "USB2.0 PC CAMERA",
        # PATCH: this device's DirectShow path is flakey -- opening +
        # setting properties under CAP_DSHOW produced a real, usable
        # frame only about half the time in repeated testing (10/10
        # failures traced to DSHOW specifically, not to which properties
        # were set or how long we waited). CAP_MSMF was 100% reliable
        # (10/10) with the identical open/set/read sequence, so use it
        # as the primary backend for this camera; "msmf" still falls
        # back to DSHOW/ANY per RGBCamera._backend_candidates if MSMF
        # itself ever fails to open.
        "backend": "msmf",
        "angle": 60,
    },
    {
        "name": "cam3",
        "type": "rgb",
        # PATCH: was DroidCam, but the phone-side DroidCam app wasn't
        # actually streaming -- the "camera" just returned a static
        # "Start DroidCam" placeholder image forever, which is what showed
        # up as a permanently "stalled" stream. Switched to the spare
        # physical USB webcam instead.
        "source": "USB2.0 PC CAMERA",
        "backend": "dshow",
        "angle": 90,
    },
]
TARGET_FPS = 30

RGB_WIDTH = 640
RGB_HEIGHT = 480


DEPTH_CAM_WIDTH = 848
DEPTH_CAM_HEIGHT = 480


DEPTH_MIN_M = 0.07
DEPTH_MAX_M = 0.50 

# PATCH: tightened from 0.08 -- 0.08s allows up to ~2.4 frames of skew at
# 30fps, which is larger than the fingertip motion between frames you're
# trying to label. See sync_manager.py's module docstring.
SYNC_TOLERANCE_SEC = 0.015

OUTPUT_ROOT = "./data"
CALIBRATION_DIR = os.path.join(OUTPUT_ROOT, "calibrations")
CHECKERBOARD = (9, 6)     # internal corners (width, height) — match your printed board
SQUARE_SIZE_MM = 25.0     # measure your actual printed squares, don't guess