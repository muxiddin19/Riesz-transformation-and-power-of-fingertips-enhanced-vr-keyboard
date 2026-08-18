"""
estimate_phone_depth.py
========================
Runs YOUR already fine-tuned Depth Anything V2 checkpoint
(checkpoints/depth_anything_v2_vits_d405_finetuned.pth, produced by
depth_finetuning/train.py) on a recorded session's phone/webcam RGB
frames, and writes an estimated depth PNG for each one -- same
uint16-millimeters convention your D405 frames already use
(depth/*.png), so the rest of your pipeline (generate_contact_labels.py,
inspect_fingertip_depths.py, etc.) can use these exactly like real
depth frames without any special-casing.

THIS REPLACES the earlier estimate_phone_depth.py (and retires
train_depth_model.py entirely) -- both of those assumed a Hugging Face
`transformers`-format checkpoint. Yours is NOT that: it's a raw state
dict trained with the OFFICIAL Depth-Anything-V2 repo's own
DepthAnythingV2 class (depth_anything_v2/dpt.py), saved exactly the way
your own depth_finetuning/train.py saves checkpoints -- a dict with a
'model' key, DistributedDataParallel-wrapped (keys prefixed 'module.').
You do NOT need to train anything here -- the checkpoint already
exists and, per your FINETUNING.md, already hits ~3.2mm MAE on held-out
D405 data. This script only runs inference with it, on the phone's
frames instead of the D405's.

Prerequisite (per your own FINETUNING.md): the official repo must be
importable --
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
Run this script from inside that repo (or anywhere with its
`depth_anything_v2` package on PYTHONPATH) -- exactly the same
requirement depth_finetuning/train.py already has, so if training ran,
this will import fine too.

CRITICAL, straight from your own FINETUNING.md's "Common Issues"
section: max_depth must be EXACTLY what training used (0.5m) -- the DPT
head computes depth = sigmoid(features) * max_depth internally, so
getting this wrong silently produces depth values 20-50x too large,
with no error or warning. Defaults here (--encoder vits, --img-size 518,
--max-depth 0.5) match configs/train_d405.sh -- override them only if
you know your actual final run used different values.

Writes TWO things per session, both under the session folder:
  depth_<camera_name>/*.png             real depth, uint16 millimeters --
                                         this is what the rest of the
                                         pipeline (labels, manifest,
                                         inspect_phone_depth.py) reads.
                                         Opening these directly in a
                                         normal image viewer will look
                                         almost solid black -- that's
                                         expected, NOT a bug: real values
                                         are a few hundred (mm) out of a
                                         possible 65535, so an unscaled
                                         viewer can't show the contrast.
  depth_<camera_name>_visualized/*.png  colorized, human-viewable-only
                                         preview of the same data (uint8
                                         BGR, JET colormap, close=warm).
                                         Purely for eyeballing -- nothing
                                         downstream reads this folder.

Usage:
  python estimate_phone_depth.py \\
      --session-dir data/P01/P01_angle30_white_desk_typing_100129 \\
      --camera-name phone_droidcam \\
      --checkpoint checkpoints/depth_anything_v2_vits_d405_finetuned.pth

  # Process every session under a participant's folder in one go:
  python estimate_phone_depth.py \\
      --data-root data/P01 --camera-name phone_droidcam \\
      --checkpoint checkpoints/depth_anything_v2_vits_d405_finetuned.pth
"""

import argparse
import os
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    raise ImportError(
        "Cannot import depth_anything_v2.dpt.DepthAnythingV2 -- clone the official "
        "repo (git clone https://github.com/DepthAnything/Depth-Anything-V2.git) and "
        "run this script from inside it (or add it to PYTHONPATH). This is the same "
        "requirement depth_finetuning/train.py already has -- if training ran "
        "successfully, this import should too. See this script's module docstring."
    )

# Same ImageNet normalization as custom_depth_dataset.py -- MUST match
# training exactly, or the model sees out-of-distribution input and
# silently produces worse depth without erroring.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Same table as depth_finetuning/train.py -- keep in sync with it if you
# ever add/change an encoder size there.
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}


def load_model(checkpoint_path, encoder, max_depth, device):
    config = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**{**config, 'max_depth': max_depth})

    raw = torch.load(checkpoint_path, map_location='cpu')
    # train.py saves {'model': state_dict, 'optimizer': ..., 'epoch': ...},
    # DDP-wrapped (keys prefixed 'module.'). Handle both that AND a plain
    # state-dict-only .pth defensively, in case this specific file was
    # exported/renamed differently than latest.pth/best.pth.
    state_dict = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def preprocess(rgb_bgr, img_size):
    """Mirrors CustomDepthDataset.__getitem__'s RGB pipeline exactly:
    BGR->RGB, resize to (img_size, img_size) -- a direct squash, NOT
    aspect-preserving, because that's what training did -- /255, then
    ImageNet normalize, then to a CHW tensor. Any mismatch here feeds
    the model out-of-distribution input with no error raised."""
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
    return tensor.unsqueeze(0)  # add batch dim


def colorize_depth_m(depth_m, vis_min_m, vis_max_m):
    """Same idea as the D405 pipeline's depth_visualized/ frames --
    the raw uint16-mm PNGs are correct but essentially unviewable
    directly (real values top out a few hundred mm, versus a 65535
    uint16 ceiling, so any plain image viewer renders them near-black).
    This clips/normalizes to a FIXED range (not per-frame min/max, so
    brightness is comparable across frames and sessions) and applies a
    colormap purely for human eyeballing -- never fed back into the
    pipeline, only written alongside the real depth_<camera_name>/ PNGs.
    """
    clipped = np.clip(depth_m, vis_min_m, vis_max_m)
    norm = (clipped - vis_min_m) / max(vis_max_m - vis_min_m, 1e-6)
    gray_u8 = (norm * 255.0).astype(np.uint8)
    # INVERTED so "close" (small depth_m) reads as warm/bright -- matches
    # the usual "closer = hotter" convention people expect from a depth map.
    colored = cv2.applyColorMap(255 - gray_u8, cv2.COLORMAP_JET)
    return colored


def process_session(session_dir, camera_name, model, device, img_size,
                     vis_min_m=0.15, vis_max_m=0.55):
    rgb_dir = os.path.join(session_dir, f"rgb_{camera_name}")
    if not os.path.isdir(rgb_dir):
        print(f"  SKIP: {rgb_dir} not found")
        return 0

    out_dir = os.path.join(session_dir, f"depth_{camera_name}")
    os.makedirs(out_dir, exist_ok=True)

    vis_dir = os.path.join(session_dir, f"depth_{camera_name}_visualized")
    os.makedirs(vis_dir, exist_ok=True)

    rgb_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    if not rgb_files:
        print(f"  SKIP: no PNG frames in {rgb_dir}")
        return 0

    with torch.no_grad():
        for fname in tqdm(rgb_files, desc=f"  {Path(session_dir).name}"):
            rgb_path = os.path.join(rgb_dir, fname)
            rgb_bgr = cv2.imread(rgb_path)
            if rgb_bgr is None:
                print(f"    WARNING: could not read {rgb_path}, skipping")
                continue
            orig_h, orig_w = rgb_bgr.shape[:2]

            input_tensor = preprocess(rgb_bgr, img_size).to(device)
            pred = model(input_tensor)  # meters -- max_depth already applied internally

            # Model output resolution may not exactly equal img_size (DPT
            # head internals) -- resize to the ORIGINAL phone frame size
            # using the SAME align_corners=True convention train.py's own
            # validation loop uses when comparing predictions to ground
            # truth, so this stays consistent with how the checkpoint was
            # actually evaluated.
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)  # (B, H', W') -> (B, 1, H', W')
            pred = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
            depth_m = pred[0, 0].cpu().numpy()

            depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(out_dir, fname), depth_mm)

            vis = colorize_depth_m(depth_m, vis_min_m, vis_max_m)
            cv2.imwrite(os.path.join(vis_dir, fname), vis)

    print(f"  -> wrote {len(rgb_files)} frame(s) to {out_dir}")
    print(f"  -> wrote {len(rgb_files)} colorized preview frame(s) to {vis_dir} "
          f"(viewable directly -- open these instead of depth_{camera_name}/ "
          f"if you just want to look at it)")
    return len(rgb_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-dir", default=None,
                         help="Single session directory to process.")
    parser.add_argument("--data-root", default=None,
                         help="Process every session found under this root "
                              "(e.g. data/P01, or data/ for everything).")
    parser.add_argument("--camera-name", required=True,
                         help="e.g. phone_droidcam -- reads rgb_<camera-name>/*.png.")
    parser.add_argument("--checkpoint", required=True,
                         help="e.g. checkpoints/depth_anything_v2_vits_d405_finetuned.pth")
    parser.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS.keys()),
                         help="MUST match what you trained with (your FINETUNING.md uses vits).")
    parser.add_argument("--img-size", type=int, default=518,
                         help="MUST match training (your FINETUNING.md uses 518).")
    parser.add_argument("--max-depth", type=float, default=0.5,
                         help="MUST match training EXACTLY (your FINETUNING.md uses 0.5m) "
                              "-- see this script's docstring for why a mismatch here "
                              "silently produces wildly wrong depth with no error.")
    parser.add_argument("--vis-min-m", type=float, default=0.15,
                         help="Colorized-preview clip range (meters), NOT used for the "
                              "real depth values -- just controls contrast in the "
                              "human-viewable depth_<camera-name>_visualized/ PNGs. "
                              "Tighten this range if the preview looks too flat/washed out.")
    parser.add_argument("--vis-max-m", type=float, default=0.55,
                         help="See --vis-min-m.")
    args = parser.parse_args()

    if not args.session_dir and not args.data_root:
        print("ERROR: provide --session-dir or --data-root")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFER] device: {device}")
    print(f"[INFER] loading checkpoint: {args.checkpoint} "
          f"(encoder={args.encoder}, img_size={args.img_size}, max_depth={args.max_depth}m)")
    model = load_model(args.checkpoint, args.encoder, args.max_depth, device)

    if args.session_dir:
        sessions = [args.session_dir]
    else:
        data_root = Path(args.data_root)
        sessions = sorted(str(p.parent) for p in data_root.glob(f"**/rgb_{args.camera_name}"))
        print(f"[INFER] found {len(sessions)} session(s) with rgb_{args.camera_name}/ "
              f"under {args.data_root}")

    total_frames = 0
    for session_dir in sessions:
        print(f"[INFER] session: {session_dir}")
        total_frames += process_session(session_dir, args.camera_name, model, device, args.img_size,
                                         vis_min_m=args.vis_min_m, vis_max_m=args.vis_max_m)

    print(f"[INFER] done -- {total_frames} frame(s) across {len(sessions)} session(s).")


if __name__ == "__main__":
    main()