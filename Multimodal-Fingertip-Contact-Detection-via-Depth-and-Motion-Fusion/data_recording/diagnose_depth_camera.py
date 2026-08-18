
import time

import pyrealsense2 as rs


def try_stream(label, enable_fn, timeout_ms=8000):
    
    print(f"\n  Testing: {label}")
    pipeline = rs.pipeline()
    cfg = rs.config()
    enable_fn(cfg)

    try:
        pipeline.start(cfg)
    except Exception as e:
        print(f"    -> FAILED at start(): {e}")
        return False

    try:
        pipeline.wait_for_frames(timeout_ms=timeout_ms)
        print(f"    -> OK: frame received")
        return True
    except RuntimeError as e:
        print(f"    -> FAILED waiting for frame: {e}")
        return False
    finally:
        pipeline.stop()


print("[1/5] Listing connected RealSense devices...")
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("  -> NO RealSense devices found. Check USB connection.")
    raise SystemExit(1)

dev = devices[0]
name = dev.get_info(rs.camera_info.name)
serial = dev.get_info(rs.camera_info.serial_number)
usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
print(f"  -> found: {name} (serial={serial}, usb={usb_type})")

print("\n[2/5] Hardware-resetting the device to clear any stuck state...")
dev.hardware_reset()
print("  -> reset sent, waiting 5s for the device to re-enumerate...")
time.sleep(5)


ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("  -> device did not come back after reset! Try a physical replug.")
    raise SystemExit(1)
print(f"  -> device back online: {devices[0].get_info(rs.camera_info.name)}")

print("\n[3/5] Testing DEPTH stream alone...")
depth_ok = try_stream(
    "depth only, 848x480 @ 30fps",
    lambda cfg: cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30),
)

print("\n[4/5] Testing COLOR stream alone...")
color_ok = try_stream(
    "color only, 848x480 @ 30fps",
    lambda cfg: cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30),
)

if depth_ok and color_ok:
    print("\n[5/5] Both worked alone — testing DEPTH + COLOR together...")
    both_ok = try_stream(
        "depth + color, 848x480 @ 30fps",
        lambda cfg: (
            cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30),
            cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30),
        ),
    )
    if both_ok:
        print("\nRESULT: everything works. Re-run main.py.")
    else:
        print("\nRESULT: each stream works ALONE but not together —")
        print("        likely a USB bandwidth/negotiation issue when both")
        print("        are requested at once. Try dropping fps to 15, or")
        print("        a lower resolution for one of the two streams.")
else:
    print("\n[5/5] Skipped combined test since a single stream already failed.")
    if not depth_ok:
        print("RESULT: DEPTH sensor itself is not producing frames.")
        print("        Likely stereo-module firmware/hardware issue on this unit.")
    if not color_ok:
        print("RESULT: COLOR (RGB) sensor itself is not producing frames.")
        print("        Likely RGB sensor firmware/hardware issue on this unit.")