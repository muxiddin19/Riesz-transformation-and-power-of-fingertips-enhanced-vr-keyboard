# Recording Guide — VR Keyboard Hand-Depth Dataset

This walks you through recording a session, start to finish. Follow it in order the
first time — most problems people hit come from skipping the calibration step or
recording too short/too uniform a clip.

## 1. One-time setup on your machine

Before you can run anything:

1. Get the project folder (`data_recording/`, `custom_data_recording/`, `assets/`) from
   whoever is coordinating this.
2. Install Python 3.10+ and create a virtual environment.
3. Install dependencies:
   ```
   pip install opencv-python mediapipe numpy pyrealsense2 pygrabber
   ```
4. Plug in the Intel RealSense D405. If you're also using a phone camera via DroidCam,
   install the DroidCam Client app on your phone AND the DroidCam Windows client, and
   connect them over USB before you start recording (not after).

## 2. One-time setup for YOUR physical rig

**Do this once per physical camera setup you use** — if you and another student each
have your own D405 and desk, each of you does this once, for your own rig. If you're
all sharing one rig, only whoever sets it up needs to do this once, ever.

1. Mount the D405 at a fixed position and angle above the desk. Once it's calibrated,
   **do not move it again** — moving it invalidates the calibration and (if you're using
   the keyboard-key feature) the keyboard annotation too, and you'd have to redo both.
2. Open `config.py` and confirm the D405's `"angle"` value matches where you actually
   mounted it.
3. Run `python main.py`.
4. With the desk completely empty (no hands, no objects, no keyboard sheet in the way
   of the surface itself), press **C** to calibrate. Wait for
   `[CALIBRATION] saved to ...` in the console, and check the red "NO CALIBRATION"
   banner is gone from the preview window before continuing.
5. **If you want per-key click labels** (which key was touched, not just "was
   something touched"): run the separate keyboard annotation tool once against this
   same D405 position, with the printed keyboard sheet laid out exactly where it'll be
   during real recording. Ask your coordinator for that tool if you don't have it.
   Skip this step if you only need general contact/no-contact labels — everything else
   still works without it.

If you're recording with a phone/webcam as a second camera, its position can change
freely between sessions (see step 4 in the next section) — none of the calibration
steps above apply to it.

## 3. Recording a session

1. Run `python main.py`.
2. Pick your participant ID and surface type when prompted.
3. Check the console: you should see `[CALIBRATION] loaded ...` (not a "no calibration
   found" warning). If you see the warning, something's wrong with step 2 above —
   don't record yet, sort that out first.
4. **If you're using a second (phone/webcam) camera and it's not at the angle you
   want**: press **A**, follow the prompts to enter its new angle, then continue. No
   need to restart the program for this.
5. Press **R** to start recording.
6. Record — see the section below for exactly how to move your hand during this part.
   It matters more than it sounds like it would.
7. Press **R** again to stop, or **N** to end this session and immediately start a new
   one (new participant/surface).
8. Watch the console after stopping: you should see a `[LABELER]` line run
   automatically. If instead you see `SKIP: No calibration`, this session did not get
   calibrated and its data can't be used — you'll need to redo it.
9. Press **Q** to quit when you're done for the day.

## 4. How to move your hand — read this before recording

This is the part that's easy to get wrong without knowing it. A previous test batch
recorded clips where almost every single frame showed "contact" — meaning the hand was
resting on or touching the keyboard nearly the whole time. That's nearly useless for
training: a model can get 99% accuracy on data like that by always guessing "contact"
and never actually learning anything.

For each session, deliberately alternate between two clearly different things, several
times over:

- **Hover**: lift your whole hand a few centimeters above the keys, clearly not
  touching anything, and hold it there for a second or two.
- **Tap / type**: bring your hand down and actually press keys, like normal typing.

Don't just type continuously the whole time. A good ~20-30 second session might look
like: hover for 2 seconds → type a short word → lift off and hover for 2 seconds →
type again → pause with hand fully off the keyboard for a few seconds → type again.
The more genuine variety between "clearly off" and "clearly touching," the better the
resulting labels will be for training.

**Watch the number on screen while you record.** The D405 preview window shows a live
`contact (last ~5s): NN%  (session: NN%)` readout while you're recording. It turns
green when you're in a healthy range (roughly 15-85%) and yellow when it's stuck near
0% or 100% — yellow means you're not getting enough real variety and should keep
mixing it up before you stop. You don't need to hit an exact number, just don't finish
a take while it's still yellow. When you stop recording, the console also prints that
take's final contact ratio as a reminder.

## 5. What happens automatically (you don't need to do these by hand)

- Every frame is saved while recording, whether or not a tap happened — you don't need
  to do anything special to capture "no contact" frames, they're saved the same as any
  other frame.
- Contact/hover labeling runs automatically right after you stop recording or start a
  new session.
- Depth frames get converted to a viewable, colorized preview automatically too
  (`depth_visualized/` inside your session folder) — useful for a quick visual check
  that a hand is actually visible in the depth data.

## 6. Common problems

**Red "NO CALIBRATION" banner won't go away.** You haven't pressed C yet for this
angle/surface combination, or calibration failed. Stop recording, press C on an empty
desk, confirm the save message, then resume.

**Console says `SKIP: No calibration` after you stop recording.** Same root cause as
above, caught after the fact — this session's data is not usable and needs to be
redone with calibration done first.

**Phone camera won't open / DirectShow errors.** Make sure the DroidCam Client app is
open and connected on your phone before starting `main.py`, not after. If you have
multiple phones, check the console for which physical device index each got assigned.

**You're not sure if a recorded session is any good.** Ask your coordinator to run
`inspect_fingertip_depths.py` and `build_dataset_manifest.py` against your session —
they'll be able to tell from the printed stats whether the contact/hover balance and
depth values look reasonable, without needing to watch the recording back.

## 7. Before you finish for the day

- Confirm each session folder has a `labels/` subfolder (means labeling succeeded).
- Send/share your session folders (or their location) to your coordinator so they can
  be combined into the full dataset.


## 8. Practical steps


  python main.py

python inspect_fingertip_depths.py data\P01\P01_white_desk_typing_<timestamp> --csv-out depth_report.csv

python inspect_phone_depth.py data\P01\P01_white_desk_typing_<timestamp> --camera-name cam1 --csv-out cam1_depth_report.csv
python inspect_phone_depth.py data\P01\P01_white_desk_typing_<timestamp> --camera-name cam2 --csv-out cam2_depth_report.csv

cd ..\custom_data_recording
python generate_contact_labels.py --data-root ..\data_recording\data

cd ..\depth_finetuning
$env:PYTHONPATH = "D:\DL\Research\Depth-Anything-V2\metric_depth"
python estimate_phone_depth.py --data-root ..\data_recording\data --camera-name cam1 --checkpoint ..\checkpoints\depth_anything_v2_vits_d405_finetuned.pth
python estimate_phone_depth.py --data-root ..\data_recording\data --camera-name cam2 --checkpoint ..\checkpoints\depth_anything_v2_vits_d405_finetuned.pth

cd ..\data_recording
python build_dataset_manifest.py --data-root data --out manifest.csv

python export_npy_dataset.py --manifest manifest.csv --out-dir dataset\npy_v1 --train-ratio 0.8 --val-ratio 0.1
