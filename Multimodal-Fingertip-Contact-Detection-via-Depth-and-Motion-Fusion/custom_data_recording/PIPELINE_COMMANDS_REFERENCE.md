# Full Pipeline — Command Reference

This covers everything that happens to a session's data, start to finish:
recording itself, what runs automatically, and the extra commands used to
check data quality and build the final training files. Students only need
Section A (recording) — Sections B and C are normally run by whoever is
coordinating/combining everyone's data, not by each student individually.
That's why `STUDENT_RECORDING_GUIDE.md` doesn't mention most of these.

Folder layout this assumes (adjust paths below if yours differs):

```
Multimodal-Fingertip-Contact-Detection-via-Depth-and-Motion-Fusion\
  checkpoints\                     <- trained depth model .pth files
  custom_data_recording\           <- generate_contact_labels.py lives here
  data_recording\                  <- main.py, recorder.py, data\, all diagnostic scripts
  depth_finetuning\                <- train.py, estimate_phone_depth.py
```

Every command below shows which of these folders to `cd` into first —
running a script from the wrong folder is the single most common error
(wrong relative paths to checkpoints/session data).

---

## A. Recording a session — fully automatic, one command

From `data_recording\`:

```
python main.py
```

Follow the on-screen prompts (participant ID, surface, calibrate with
**C**, record with **R**, change phone/webcam angle with **A**, new
session with **N**, quit with **Q**). See `STUDENT_RECORDING_GUIDE.md`
for the full walkthrough — this is the only command students need.

The following happen automatically, no extra command needed:
- Calibration warnings (console + red on-screen banner) if you forgot to
  calibrate.
- Contact/hover labeling (`generate_contact_labels.py`) runs right after
  you stop recording or start a new session — writes `labels\` inside the
  session folder.
- D405 depth gets a colorized, human-viewable copy written to
  `depth_visualized\` inside the session folder.
- Live contact-ratio HUD and per-key overlay shown during recording (no
  files written for this — it's just on-screen feedback).

---

## B. Per-session diagnostics (optional, one session at a time)

Use these to sanity-check one specific session before trusting it. Both
only take a single session, not a whole folder of sessions.

**Check the D405's real depth quality** — from `data_recording\`:

```
python inspect_fingertip_depths.py data\P07\P07_semi_reflective_typing_015707 --csv-out depth_report.csv
```

**Check the phone/webcam's estimated depth quality** (after step C2 below
has been run for this session) — from `data_recording\`:

```
python inspect_phone_depth.py data\P07\P07_semi_reflective_typing_015707 --camera-name phone_droidcam --csv-out phone_depth_report.csv
```

Both just print stats and write a CSV next to wherever you run them —
they don't change any of the actual data files.

---

## C. Building the full dataset (run once, over ALL collected sessions)

Do these after everyone's session folders have been combined into one
`data_recording\data\` folder (each student's participant ID should be
unique, e.g. P01, P02, ... so folders don't collide).

### C1. Re-run contact labeling in bulk (only if needed)

Normally not needed — step A already labels each session automatically
as it's recorded. Use this if you changed the contact/hover thresholds,
or need to re-label sessions from before the automation existed.

From `custom_data_recording\`:

```
python generate_contact_labels.py --data-root ..\data_recording\data
```

(Or `--session-dir ..\data_recording\data\P07\P07_semi_reflective_typing_015707`
for just one session.)

### C2. Estimate depth for every phone/webcam session

The phone/webcam has no real depth sensor, so this uses the fine-tuned
model (`checkpoints\depth_anything_v2_vits_d405_finetuned.pth`) to predict
it from the RGB frames. From `depth_finetuning\`:

```
$env:PYTHONPATH = "D:\DL\Research\Depth-Anything-V2\metric_depth"
python estimate_phone_depth.py --data-root ..\data_recording\data --camera-name phone_droidcam --checkpoint ..\checkpoints\depth_anything_v2_vits_d405_finetuned.pth
```

The `$env:PYTHONPATH` line only needs to be set once per terminal window
(it resets if you close and reopen PowerShell). This writes, per session:
- `depth_phone_droidcam\` — real depth data (uint16 millimeters), used by
  everything downstream.
- `depth_phone_droidcam_visualized\` — colorized copy for a human to look
  at. Opening the raw `depth_phone_droidcam\` PNGs directly in an image
  viewer looks solid black — that's expected, not an error (see the
  script's own printed note when it finishes).

### C3. Build the flat has_contact / clicked_key manifest

Combines every session's per-fingertip labels into one flat CSV matching
the target dataset schema. From `data_recording\`:

```
python build_dataset_manifest.py --data-root data --out manifest.csv
```

### C4. Pack training arrays (.npy) — best-guess format, confirm with mentor first

From `data_recording\`:

```
python export_npy_dataset.py --manifest manifest.csv --out-dir dataset\npy_v1 --train-ratio 0.8 --val-ratio 0.1
```

This format hasn't been confirmed by the mentor yet (see the script's own
docstring for the reasoning) — hold off on treating its output as final
until that's confirmed. Everything else above (labels, manifest, depth
images) is already the settled/expected output.

---

## Quick order-of-operations summary

1. Students record (Section A) — everything in Section A happens per
   student, automatically, no extra steps.
2. Coordinator collects all session folders into one `data_recording\data\`.
3. Coordinator runs C2 (phone depth) → C3 (manifest) → C4 (npy, once
   confirmed) once over the combined data.
4. Section B diagnostics are spot-checks, run on any session anyone wants
   a second opinion on — not a required step in the main pipeline.