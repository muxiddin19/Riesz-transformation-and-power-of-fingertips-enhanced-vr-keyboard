
PARTICIPANTS = [f"P{str(i).zfill(2)}" for i in range(1, 16)]  

SURFACES = [
    "white_desk",
    "wood_grain",
    "semi_reflective",
]

CAMERAS = [
    {
        "name": "depth_cam",
        "type": "depth",
        "source": 0,
        "angle": 45,
    },
    {
        "name": "cam1",
        "type": "rgb",
        "source": "DroidCam",       # was "phone_droidcam" -- renamed for consistency, see note below
        "backend": "dshow",
        "angle": 30,
    },
    {
        "name": "cam2",
        "type": "rgb",
        "source": 1,                # plain webcam device index -- change to your actual index
        "backend": "dshow",
        "angle": 60,
    },
    {
        "name": "cam3",
        "type": "rgb",
        "source": "http://192.168.0.XX:4747/video",  # 2nd DroidCam / IP cam -- replace with real URL
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

SYNC_TOLERANCE_SEC = 0.08

OUTPUT_ROOT = "./data"
