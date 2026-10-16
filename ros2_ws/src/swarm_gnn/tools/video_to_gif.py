#!/usr/bin/env python3
"""Convert a screen recording into a README-sized animated GIF.

    python3 tools/video_to_gif.py drones_GNN.mp4 --out docs/media/walkthrough.gif

A README animation has to be a few megabytes. The source here is 123 s of
1280x676 at 25 fps, which is far beyond that as a GIF, so three things are
traded off together:

  width     downscale; detail beyond ~760 px is lost to GIF quantisation anyway
  stride    keep every Nth frame, which time-lapses the clip rather than
            truncating it, so the whole mission is still shown
  colours   a single global adaptive palette across all frames

Frames are also deduplicated: a screen recording of a slow mission holds long
near-identical stretches, and dropping them costs nothing visually while
removing a large fraction of the file.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from PIL import Image


def frames_from(path: str, stride: int, width: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out, index = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            h, w = frame.shape[:2]
            scale = width / w
            small = cv2.resize(
                frame, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA
            )
            out.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        index += 1
    cap.release()
    return out, src_fps, total


def drop_near_duplicates(frames: list[np.ndarray], threshold: float):
    """Remove frames that barely differ from the one before.

    A mission recording spends long stretches with three slow robots and a
    static map. Those frames cost full size in the GIF and show nothing.
    """
    if not frames:
        return frames
    kept = [frames[0]]
    for frame in frames[1:]:
        diff = np.mean(np.abs(frame.astype(np.int16) - kept[-1].astype(np.int16)))
        if diff >= threshold:
            kept.append(frame)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--stride", type=int, default=6,
                    help="keep every Nth source frame")
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--ms", type=int, default=70, help="frame duration in the GIF")
    ap.add_argument("--dedupe", type=float, default=1.2,
                    help="mean abs pixel difference below which a frame is dropped")
    args = ap.parse_args()

    frames, src_fps, total = frames_from(args.video, args.stride, args.width)
    print(f"source      {total} frames at {src_fps:.1f} fps "
          f"({total / src_fps:.0f} s)")
    print(f"sampled     {len(frames)} frames (every {args.stride})")

    frames = drop_near_duplicates(frames, args.dedupe)
    print(f"deduplicated{len(frames):>5} frames kept")

    images = [Image.fromarray(f) for f in frames]

    # Build the shared palette from frames spread across the whole clip, not
    # from the first one. The opening frames of a mission recording are mostly
    # empty map, so a palette derived from them alone has nothing allocated to
    # the colours that appear later === threat markers, track rings, the HUD.
    sample_count = min(24, len(images))
    picks = [images[i * (len(images) - 1) // max(1, sample_count - 1)]
             for i in range(sample_count)]
    w, h = images[0].size
    montage = Image.new("RGB", (w, h * len(picks)))
    for i, im in enumerate(picks):
        montage.paste(im, (0, i * h))
    palette = montage.quantize(colors=args.colors, method=Image.MEDIANCUT)
    images = [im.quantize(palette=palette, dither=Image.NONE) for im in images]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    images[0].save(
        args.out, save_all=True, append_images=images[1:],
        duration=args.ms, loop=0, optimize=True,
    )
    mb = os.path.getsize(args.out) / 1e6
    playback = len(images) * args.ms / 1000.0
    speedup = (total / src_fps) / playback if playback else 0
    print(f"wrote       {args.out}")
    print(f"            {len(images)} frames, {images[0].size[0]}x{images[0].size[1]}, "
          f"{mb:.1f} MB")
    print(f"            {playback:.0f} s playback = {speedup:.1f}x speed")


if __name__ == "__main__":
    main()
