#!/usr/bin/env python3
"""Render a recorded mission into the figures and animation used in the README.

    python3 tools/record_mission.py --out /tmp/mission.jsonl   # while running
    python3 tools/render_media.py /tmp/mission.jsonl --out ../../../docs/media

Produces:
    mission.gif          animated replay of the whole mission
    sector.png           the layout, emitters and insertion/rally points
    pipeline.png         the coordination pipeline, drawn to scale
    localization.png     fix error against DF baseline, per emitter
    phases.png           phase timeline with coverage and threat progress

Drawn with Pillow only. matplotlib is unusable in this environment (the
installed build is ABI-incompatible with numpy 2.x), and a dependency-free
renderer also means the figures regenerate anywhere the project builds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")
from swarm_gnn.model import sector  # noqa: E402

# ============================== presentation ================================
BG = (22, 24, 28)
PANEL = (30, 33, 38)
GRID = (44, 48, 54)
BUILDING = (62, 67, 75)
BUILDING_EDGE = (82, 88, 97)
RUBBLE = (104, 76, 48)
ROUTE = (40, 43, 49)
TEXT = (226, 230, 236)
MUTED = (138, 146, 158)

PLATFORM_COLOUR = {0: (70, 180, 255), 1: (245, 190, 60), 2: (110, 225, 120)}
PLATFORM_NAME = {0: "RECON (drone)", 1: "MULE (UGV)", 2: "HOUND (quadruped)"}

# Track colour carries the *decision*, not just the pipeline stage: once a
# track is confirmed, a no-strike civilian must not look like a hostile. An
# earlier version coloured every confirmed track red, which made the one
# emitter the team is forbidden to touch the most alarming thing on screen.
THREAT_STATE_COLOUR = {
    0: (225, 225, 90),    # candidate  === something is transmitting
    1: (255, 150, 40),    # localized  === fix tight enough to act on
    3: (120, 128, 140),   # neutralised
}
CONFIRMED_HOSTILE = (255, 70, 70)
CONFIRMED_CIVILIAN = (90, 160, 255)


def threat_colour(threat: dict) -> tuple[int, int, int]:
    if threat["state"] == 3:
        return THREAT_STATE_COLOUR[3]
    if threat["state"] == 2:
        return CONFIRMED_CIVILIAN if threat["cls"] == 3 else CONFIRMED_HOSTILE
    return THREAT_STATE_COLOUR.get(threat["state"], (200, 200, 200))
CLASS_NAME = {0: "unknown", 1: "JAMMER", 2: "IED TRIGGER", 3: "civilian"}
EMITTER_COLOUR = {
    "jammer": (255, 80, 80),
    "ied_trigger": (255, 165, 50),
    "civilian": (90, 160, 255),
}

MARGIN = 54
HUD_H = 132


def _font(size: int, bold: bool = False):
    names = (
        ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"] if bold else ["DejaVuSans.ttf"]
    )
    for name in names:
        for root in (
            "/usr/share/fonts/truetype/dejavu/",
            "/usr/share/fonts/TTF/",
            "/usr/share/fonts/",
        ):
            path = os.path.join(root, name)
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    pass
    return ImageFont.load_default()


class Canvas:
    """World-to-pixel mapping for the sector, plus the drawing primitives."""

    def __init__(self, size: int = 720, hud: int = 0) -> None:
        self.size = size
        self.hud = hud
        self.img = Image.new("RGB", (size, size + hud), BG)
        self.d = ImageDraw.Draw(self.img, "RGBA")
        self.half = sector.SECTOR_HALF
        self.scale = (size - 2 * MARGIN) / (2 * self.half)

    def px(self, x: float, y: float) -> tuple[float, float]:
        # y is flipped: world north is up.
        return (
            MARGIN + (x + self.half) * self.scale,
            MARGIN + (self.half - y) * self.scale,
        )

    def box(self, cx, cy, hx, hy, fill, outline=None, width=1):
        x0, y0 = self.px(cx - hx, cy + hy)
        x1, y1 = self.px(cx + hx, cy - hy)
        self.d.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=width)

    def disc(self, x, y, r_world, fill, outline=None, width=1):
        cx, cy = self.px(x, y)
        r = max(1.5, r_world * self.scale)
        self.d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill,
                       outline=outline, width=width)

    def line(self, a, b, fill, width=1):
        self.d.line([self.px(*a), self.px(*b)], fill=fill, width=width)

    def text(self, x, y, s, font, fill=TEXT, anchor="mm"):
        self.d.text(self.px(x, y), s, font=font, fill=fill, anchor=anchor)


def draw_sector(c: Canvas, *, label_emitters: bool = False,
                emitter_truth: bool = False) -> None:
    """Buildings, routes, rally points === the static scene."""
    f_small = _font(11)

    # Sector bound and a 10 m grid, for scale.
    x0, y0 = c.px(-c.half, c.half)
    x1, y1 = c.px(c.half, -c.half)
    c.d.rectangle([x0, y0, x1, y1], fill=PANEL)
    for v in range(-50, 60, 10):
        c.line((v, -c.half), (v, c.half), GRID)
        c.line((-c.half, v), (c.half, v), GRID)

    # The two open routes.
    rw = sector.ROUTE_HALF_WIDTH
    c.box(0, 0, sector.SEARCH_HALF, rw, ROUTE)
    c.box(0, 0, rw, sector.SEARCH_HALF, ROUTE)

    for b in sector.BUILDINGS:
        # Taller blocks read lighter, so the skyline is legible in plan view.
        shade = 52 + int(38 * min(1.0, b.height / sector.MAX_STRUCTURE_HEIGHT))
        c.box(b.x, b.y, b.half_x, b.half_y, (shade, shade + 5, shade + 13),
              outline=BUILDING_EDGE)
    for b in sector.RUBBLE:
        c.box(b.x, b.y, b.half_x, b.half_y, RUBBLE, outline=(140, 100, 60))

    for label, (px, py), col in (
        ("INSERTION", sector.INSERTION, (70, 200, 95)),
        ("RALLY", sector.RALLY, (90, 150, 255)),
    ):
        c.disc(px, py, 3.0, col + (90,), outline=col, width=2)
        c.text(px, py - 6.5, label, f_small, col)

    if emitter_truth:
        for e in sector.EMITTERS:
            col = EMITTER_COLOUR[e.kind]
            c.disc(e.x, e.y, 2.2, col, outline=(255, 255, 255), width=1)
            if label_emitters:
                c.text(e.x, e.y + 6.0, f"{e.name}\n{e.kind}", f_small, col)


# ================================ animation =================================
def draw_frame(frame: dict, size: int = 640) -> Image.Image:
    c = Canvas(size, HUD_H)
    draw_sector(c)
    f_small, f_mid, f_big = _font(11), _font(13), _font(17, True)

    # Open tasks.
    for t in frame["tasks"]:
        if t["id"].startswith("cell_") or t["id"].startswith("rally_"):
            c.disc(t["x"], t["y"], 1.1, (150, 160, 175, 150))
        else:
            c.disc(t["x"], t["y"], 1.8, (255, 215, 90, 210),
                   outline=(255, 235, 150), width=1)

    # Threat tracks: ring = uncertainty, core = state.
    for th in frame["threats"]:
        col = threat_colour(th)
        # Ring is the 2-sigma uncertainty, so the fix visibly tightens.
        c.disc(th["x"], th["y"], max(2.0, min(th["sigma"] * 2.0, 22.0)),
               col + (34,), outline=col + (140,), width=1)
        c.disc(th["x"], th["y"], 1.9, col)
        if th["state"] == 3:
            # Struck through once a marker has been emplaced.
            cx, cy = c.px(th["x"], th["y"])
            for dx, dy in ((-1, -1, ), (-1, 1)):
                c.d.line([cx + dx * 5, cy + dy * 5, cx - dx * 5, cy - dy * 5],
                         fill=(235, 240, 245), width=2)
        if th["state"] >= 2:
            label = CLASS_NAME.get(th["cls"], "?")
            if th["state"] == 3:
                label += "  marked"
            c.text(th["x"], th["y"] - 6.0, label, f_small, col)

    # Assignment lines, then the platforms on top.
    by_id = {a["id"]: a for a in frame["agents"]}
    task_pos = {t["id"]: (t["x"], t["y"]) for t in frame["tasks"]}
    for agent_id, task_id in frame["assignments"].items():
        a, t = by_id.get(agent_id), task_pos.get(task_id)
        if a and t:
            col = PLATFORM_COLOUR.get(a["platform"], (200, 200, 200))
            c.line((a["x"], a["y"]), t, col + (110,), 1)

    for a in frame["agents"]:
        col = PLATFORM_COLOUR.get(a["platform"], (200, 200, 200))
        # Heading spike.
        hx = a["x"] + 4.6 * math.cos(a["yaw"])
        hy = a["y"] + 4.6 * math.sin(a["yaw"])
        c.line((a["x"], a["y"]), (hx, hy), col, 2)
        if a["platform"] == 0:
            # The aircraft gets a halo scaled by altitude, so climb and
            # descent are visible in a plan view.
            c.disc(a["x"], a["y"], 2.0 + a["z"] * 0.09, col + (22,))
        c.disc(a["x"], a["y"], 2.1, col, outline=(255, 255, 255), width=1)

    # ------------------------------ HUD ------------------------------------
    s = frame["status"]
    top = c.size + 8
    c.d.rectangle([0, c.size, c.size, c.size + HUD_H], fill=(16, 18, 21))
    c.d.text((MARGIN, top), "OPERATION SENTINEL GRID", font=f_big, fill=TEXT)
    c.d.text((MARGIN + 282, top + 3),
             f"t+{s['t']:.0f}s", font=f_mid, fill=MUTED)

    phase_x = MARGIN
    for name in ("INGRESS", "SEARCH", "LOCALIZE", "CONFIRM",
                 "NEUTRALIZE", "BDA", "EGRESS", "COMPLETE"):
        active = name == s["phase_name"]
        col = (90, 220, 140) if active else (74, 80, 90)
        c.d.text((phase_x, top + 28), name, font=f_small,
                 fill=col if active else MUTED)
        w = c.d.textlength(name, font=f_small)
        c.d.rectangle([phase_x, top + 43, phase_x + w, top + 45],
                      fill=col if active else (48, 52, 58))
        phase_x += w + 14

    # Coverage bar.
    bar_w = c.size - 2 * MARGIN
    c.d.rectangle([MARGIN, top + 58, MARGIN + bar_w, top + 68], fill=(44, 48, 54))
    c.d.rectangle([MARGIN, top + 58, MARGIN + bar_w * s["coverage"], top + 68],
                  fill=(70, 170, 130))
    c.d.text((MARGIN, top + 74),
             f"sector swept {s['coverage'] * 100:.0f}%", font=f_small, fill=MUTED)
    c.d.text((MARGIN + 160, top + 74),
             f"emitters  {s['detected']} detected   "
             f"{s['confirmed']} confirmed   {s['neutralized']} neutralised",
             font=f_small, fill=MUTED)
    c.d.text((MARGIN, top + 92), s["narrative"][:96], font=f_small, fill=(170, 180, 195))

    # Legend.
    lx = MARGIN
    for pid in (0, 1, 2):
        col = PLATFORM_COLOUR[pid]
        c.d.ellipse([lx, top + 110, lx + 9, top + 119], fill=col)
        c.d.text((lx + 14, top + 109), PLATFORM_NAME[pid], font=f_small, fill=MUTED)
        lx += int(c.d.textlength(PLATFORM_NAME[pid], font=f_small)) + 34
    return c.img


def render_gif(frames: list[dict], out: str, stride: int, ms: int) -> None:
    """Animate the recording.

    Coverage is shown by the HUD bar, taken straight from MissionStatus, rather
    than by shading individual cells. Per-cell state is not in the recording,
    and reconstructing it from which cell tasks are currently on offer gets it
    wrong at the start and end of the sweep === the bar is the figure the
    executive actually publishes.
    """
    images = [
        draw_frame(frame)
        for k, frame in enumerate(frames)
        if k % stride == 0
    ]
    if not images:
        raise SystemExit("recording produced no frames")
    # Quantise to a shared adaptive palette. A README animation has to be a
    # few megabytes, not twenty: full-colour frames produced a 17 MB file.
    palette = images[0].quantize(colors=96, method=Image.MEDIANCUT)
    images = [im.quantize(palette=palette, dither=Image.NONE) for im in images]
    images[0].save(
        out, save_all=True, append_images=images[1:],
        duration=ms, loop=0, optimize=True,
    )
    print(f"  mission.gif        {len(images)} frames, "
          f"{os.path.getsize(out) / 1e6:.1f} MB")


# ================================ figures ===================================
def render_pipeline(out: str) -> None:
    """The coordination pipeline, as a box-and-arrow diagram.

    Drawn rather than described because the division of responsibility is the
    design: the executive decides *what the mission needs*, the network decides
    *who should do it*, and the solver enforces the combinatorics. Those three
    are separable and the diagram should make that obvious.
    """
    W, H = 980, 560
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    f_title, f_box, f_small, f_tag = (
        _font(19, True), _font(13, True), _font(11), _font(10))

    d.text((34, 24), "Coordination pipeline \u2014 one tick, 2 Hz",
           font=f_title, fill=TEXT)
    d.text((34, 50),
           "Simulation feeds perception; perception feeds decision; decision "
           "feeds motion; motion feeds back into simulation.",
           font=f_small, fill=MUTED)

    boxes = [
        ("threat_field", "RF propagation + CBRN plume\nground truth -> noisy detections",
         (34, 96), (150, 90, 70)),
        ("rf_localizer", "Bayesian grid per channel\n-> multilateration -> tracks",
         (34, 196), (70, 130, 170)),
        ("mission_executive", "phase machine\nWHAT the mission needs -> tasks",
         (34, 296), (80, 150, 110)),
        ("gnn_allocator", "heterogeneous GAT + Hungarian\nWHO does WHAT -> assignments",
         (34, 396), (160, 120, 180)),
    ]
    bw, bh = 300, 74
    for name, sub, (x, y), col in boxes:
        d.rounded_rectangle([x, y, x + bw, y + bh], 7, fill=col + (46,),
                            outline=col, width=2)
        d.text((x + 14, y + 11), name, font=f_box, fill=TEXT)
        for i, line in enumerate(sub.split("\n")):
            d.text((x + 14, y + 32 + i * 15), line, font=f_small, fill=MUTED)
        if y < 396:
            d.line([x + bw / 2, y + bh, x + bw / 2, y + 100], fill=(96, 104, 116), width=2)
            d.polygon([(x + bw / 2 - 5, y + 95), (x + bw / 2 + 5, y + 95),
                       (x + bw / 2, y + 102)], fill=(96, 104, 116))

    # Per-agent column.
    ax = 420
    d.text((ax, 76), "per platform  (x3)", font=f_box, fill=TEXT)
    agent_boxes = [
        ("agent_controller", "A* route + potential field\nsensor actions, evidence",
         (150, 150, 170)),
        ("kinematic_platform", "integrates cmd_vel\ncollision, scan, joint animation",
         (120, 140, 150)),
    ]
    for i, (name, sub, col) in enumerate(agent_boxes):
        y = 110 + i * 100
        d.rounded_rectangle([ax, y, ax + bw, y + bh], 7, fill=col + (40,),
                            outline=col, width=2)
        d.text((ax + 14, y + 11), name, font=f_box, fill=TEXT)
        for j, line in enumerate(sub.split("\n")):
            d.text((ax + 14, y + 32 + j * 15), line, font=f_small, fill=MUTED)
        if i == 0:
            d.line([ax + bw / 2, y + bh, ax + bw / 2, y + 100],
                   fill=(96, 104, 116), width=2)

    # Loop arrows.
    d.line([334, 433, 392, 433, 392, 150, 420, 150], fill=(150, 130, 180), width=2)
    d.polygon([(414, 145), (414, 155), (422, 150)], fill=(150, 130, 180))
    d.text((352, 414), "assignments", font=f_tag, fill=(170, 150, 200))

    d.line([570, 284, 570, 514, 18, 514, 18, 133, 28, 133],
           fill=(110, 150, 120), width=2)
    d.polygon([(26, 128), (26, 138), (34, 133)], fill=(110, 150, 120))
    d.text((292, 522), "platform poses \u2192 new detections  (closes the loop)",
           font=f_tag, fill=(140, 180, 150))

    # Right-hand notes: the two constraints that are structural, not learned.
    nx = 756
    d.text((nx, 96), "enforced structurally", font=f_box, fill=TEXT)
    notes = [
        ("capability", "an agent-task edge exists\nonly if the agent can do it,\n"
                       "so an infeasible pairing\nis unscoreable"),
        ("priority", "mission-critical tasks are\nmatched before routine ones,\n"
                     "in a separate tier"),
        ("commitment", "an agent keeps its task\nuntil it is done, gone, or\n"
                       "preempted by urgent work"),
    ]
    y = 124
    for title, body in notes:
        d.text((nx, y), title, font=f_small, fill=(200, 175, 110))
        for i, line in enumerate(body.split("\n")):
            d.text((nx, y + 16 + i * 14), line, font=f_tag, fill=MUTED)
        y += 16 + 14 * len(body.split("\n")) + 14

    img.save(out)
    print("  pipeline.png")


def render_timeline(frames: list[dict], out: str) -> None:
    """Coverage and threat progress against mission time."""
    W, H = 980, 440
    PL, PR, PT, PB = 70, 250, 104, 56
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    f_title, f_small, f_tag = _font(19, True), _font(11), _font(10)

    d.text((34, 24), "Mission progress", font=f_title, fill=TEXT)
    d.text((34, 50), "one recorded run, sampled at 1 Hz. Phases overlap by "
           "design \u2014 the team does not idle waiting for a boundary.",
           font=f_small, fill=MUTED)

    times = [f["status"]["t"] for f in frames]
    t_max = max(times) or 1.0
    pw, ph = W - PL - PR, H - PT - PB

    def X(t):
        return PL + pw * (t / t_max)

    def Y(v):
        return PT + ph * (1.0 - v)

    d.rectangle([PL, PT, PL + pw, PT + ph], fill=PANEL)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = Y(frac)
        d.line([PL, y, PL + pw, y], fill=GRID)
        d.text((PL - 10, y), f"{frac * 100:.0f}%", font=f_tag, fill=MUTED, anchor="rm")
    for t in range(0, int(t_max) + 1, 60):
        d.line([X(t), PT, X(t), PT + ph], fill=GRID)
        d.text((X(t), PT + ph + 8), f"{t}s", font=f_tag, fill=MUTED, anchor="mt")

    # Phase bands along the top.
    phase_col = {
        "INGRESS": (90, 110, 150), "SEARCH": (80, 140, 120),
        "LOCALIZE": (170, 150, 70), "CONFIRM": (190, 130, 60),
        "NEUTRALIZE": (190, 80, 80), "BDA": (120, 170, 180),
        "EGRESS": (110, 130, 190), "COMPLETE": (90, 170, 110),
    }
    run_start, run_name = 0.0, frames[0]["status"]["phase_name"]
    for f in frames + [None]:
        name = f["status"]["phase_name"] if f else None
        if name != run_name:
            col = phase_col.get(run_name, (100, 100, 100))
            t_end = f["status"]["t"] if f else t_max
            d.rectangle([X(run_start), PT - 30, X(t_end), PT - 12], fill=col + (210,))
            width_px = X(t_end) - X(run_start)
            label_px = d.textlength(run_name, font=f_tag)
            if width_px > label_px + 8:
                d.text(((X(run_start) + X(t_end)) / 2, PT - 21), run_name,
                       font=f_tag, fill=(18, 20, 24), anchor="mm")
            elif width_px > 6:
                # Too narrow to name in place; tick it instead so the band is
                # still visibly a distinct phase rather than a colour smear.
                d.line([X(run_start), PT - 34, X(run_start), PT - 30],
                       fill=col, width=2)
            run_start, run_name = (f["status"]["t"] if f else t_max), name

    def series(key, scale, colour, width=2):
        pts = []
        for f in frames:
            pts.append((X(f["status"]["t"]), Y(min(1.0, f["status"][key] / scale))))
        d.line(pts, fill=colour, width=width)

    series("coverage", 1.0, (70, 200, 160), 3)
    series("confirmed", 4.0, (255, 165, 60), 3)
    series("neutralized", 3.0, (255, 80, 80), 3)

    lx, ly = PL + pw + 24, PT
    for label, colour, note in (
        ("sector swept", (70, 200, 160), "fraction of 49 cells"),
        ("emitters confirmed", (255, 165, 60), "of 4, two modalities each"),
        ("hostiles neutralised", (255, 80, 80), "of 3; the civilian is spared"),
    ):
        d.line([lx, ly + 6, lx + 22, ly + 6], fill=colour, width=3)
        d.text((lx + 30, ly), label, font=f_small, fill=TEXT)
        d.text((lx + 30, ly + 15), note, font=f_tag, fill=MUTED)
        ly += 42

    final = frames[-1]["status"]
    d.text((lx, ly + 14), "outcome", font=f_small, fill=TEXT)
    for i, line in enumerate([
        f"COMPLETE at t+{final['t']:.0f}s",
        f"{final['coverage'] * 100:.0f}% swept",
        f"{final['neutralized']}/3 hostiles neutralised",
        "0 fratricide",
    ]):
        d.text((lx, ly + 32 + i * 15), line, font=f_tag, fill=(150, 200, 170))

    img.save(out)
    print("  phases.png")


def render_poster(frames: list[dict], out: str) -> None:
    """A clickable poster for the walkthrough video.

    GitHub's markdown sanitiser strips <video>, so a repo-hosted clip cannot
    play inline in a README. The pattern that does work is a still image
    linked to the file: the image renders, and clicking it opens GitHub's own
    player. This draws that still from the recording so the poster is a real
    frame of the system rather than a stock graphic.
    """
    # A frame late enough to show results, early enough to still show motion.
    frame = frames[int(len(frames) * 0.62)]
    img = draw_frame(frame, size=760).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Dim the plan view so the play affordance reads clearly.
    d.rectangle([0, 0, w, h - HUD_H], fill=(10, 12, 15, 110))

    cx, cy, r = w // 2, (h - HUD_H) // 2, 58
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 24, 30, 215),
              outline=(236, 240, 246), width=3)
    d.polygon([(cx - 17, cy - 27), (cx - 17, cy + 27), (cx + 28, cy)],
              fill=(236, 240, 246))

    f_title, f_sub = _font(27, True), _font(14)
    d.text((cx, cy + r + 30), "Watch the walkthrough", font=f_title,
           fill=(240, 244, 250), anchor="mm")
    d.text((cx, cy + r + 58),
           "full mission in RViz \u2014 search, localise, confirm, neutralise",
           font=f_sub, fill=(176, 186, 200), anchor="mm")
    img.save(out)
    print("  video_poster.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--out", default="docs/media")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--ms", type=int, default=110)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames = [json.loads(l) for l in open(args.recording) if l.strip()]
    print(f"loaded {len(frames)} frames")

    render_gif(frames, os.path.join(args.out, "mission.gif"), args.stride, args.ms)

    c = Canvas(760)
    draw_sector(c, label_emitters=True, emitter_truth=True)
    f = _font(15, True)
    c.d.text((MARGIN, 16), "The contested sector \u2014 64 buildings, two open routes, "
             "four emitters concealed in block courtyards", font=_font(12), fill=MUTED)
    c.img.save(os.path.join(args.out, "sector.png"))
    print("  sector.png")

    render_poster(frames, os.path.join(args.out, "video_poster.png"))
    render_pipeline(os.path.join(args.out, "pipeline.png"))
    render_timeline(frames, os.path.join(args.out, "phases.png"))


if __name__ == "__main__":
    main()
