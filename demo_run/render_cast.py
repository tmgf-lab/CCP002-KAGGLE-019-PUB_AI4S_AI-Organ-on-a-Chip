"""Render an asciinema recording (.cast) of a real terminal session to a 1920x1080 MP4.

The terminal is replayed by a terminal emulator (pyte) exactly as recorded; only the playback speed changes, and the header
says by how much. Usage:
    ~/.venvs/rec/bin/python render_cast.py step.cast out.mp4 --title "Q2 guided search" --speed 4 [--hold 2]
"""
import argparse
import json
import subprocess

import pyte
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 15
BG, HEAD, INK, MUTED, ACCENT = (24, 24, 27), (250, 250, 248), (11, 11, 11), (90, 90, 86), (235, 104, 52)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COLORS = {"default": (226, 226, 222), "black": (60, 60, 60), "red": (240, 98, 98), "green": (110, 210, 140),
          "brown": (230, 190, 90), "yellow": (240, 210, 100), "blue": (110, 160, 240), "magenta": (210, 140, 230),
          "cyan": (110, 210, 220), "white": (240, 240, 240)}


def colour(c, default):
    if c == "default":
        return default
    if c in COLORS:
        return COLORS[c]
    if len(c) == 6:
        try:
            return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cast")
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold the last frame")
    a = ap.parse_args()
    lines = open(a.cast).read().splitlines()
    head = json.loads(lines[0])
    events = [json.loads(x) for x in lines[1:]]
    events = [(t, d) for t, kind, d in events if kind == "o"]
    cols, rows = head["width"], head["height"]
    screen = pyte.Screen(cols, rows)
    stream = pyte.ByteStream(screen)
    top = 84
    cw = (W - 80) / cols
    size = int(cw / 0.602)
    font, font_b = ImageFont.truetype(FONT, size), ImageFont.truetype(FONT_B, size)
    lh = min(int(size * 1.22), (H - top - 30) // rows)
    title_f, badge_f = ImageFont.truetype(SANS_B, 30), ImageFont.truetype(SANS_B, 24)
    badge = "REAL RUN" + (f"  ·  ×{a.speed:g} speed" if a.speed != 1 else "  ·  real time")

    def frame():
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, top - 14], fill=HEAD)
        d.rectangle([0, top - 18, W, top - 14], fill=ACCENT)
        d.text((40, 20), a.title, font=title_f, fill=INK)
        bw = d.textlength(badge, font=badge_f) + 36
        d.rounded_rectangle([W - 40 - bw, 16, W - 40, 56], radius=8, fill=ACCENT)
        d.text((W - 40 - bw + 18, 22), badge, font=badge_f, fill=(255, 255, 255))
        for y in range(rows):
            row = screen.buffer[y]
            x = 0
            while x < cols:
                ch = row[x]
                style = (ch.fg, ch.bold)
                run = ch.data
                x2 = x + 1
                while x2 < cols and (row[x2].fg, row[x2].bold) == style:
                    run += row[x2].data
                    x2 += 1
                if run.strip():
                    d.text((40 + x * cw, top + 8 + y * lh), run, font=font_b if ch.bold else font,
                           fill=colour(ch.fg, COLORS["default"]))
                x = x2
        cx, cy = screen.cursor.x, screen.cursor.y
        d.rectangle([40 + cx * cw, top + 8 + cy * lh, 40 + (cx + 1) * cw - 2, top + 8 + (cy + 1) * lh - 4], fill=(200, 200, 196))
        return im.tobytes()

    dur = events[-1][0] / a.speed + a.hold
    n = int(dur * FPS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", a.out],
                          stdin=subprocess.PIPE)
    i, last, dirty = 0, None, True
    for k in range(n):
        t_in = k / FPS * a.speed
        while i < len(events) and events[i][0] <= t_in:
            stream.feed(events[i][1].encode())
            i += 1
            dirty = True
        if dirty:
            last, dirty = frame(), False
        ff.stdin.write(last)
    ff.stdin.close()
    ff.wait()
    print(f"{a.out}: {dur:.1f} s (recorded {events[-1][0]:.1f} s at x{a.speed:g})")


if __name__ == "__main__":
    main()
