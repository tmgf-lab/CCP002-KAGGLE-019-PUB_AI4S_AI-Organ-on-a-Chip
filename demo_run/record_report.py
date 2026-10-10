"""Screen-record the HTML report written by vcp_system.py (scrolling through it in a headless browser).

    python demo_run/record_report.py /tmp/demo/system/report.html      # -> video/report_recording.mp4
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "video"


def main(report):
    from playwright.sync_api import sync_playwright
    shutil.rmtree(OUT / "report_raw", ignore_errors=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1600, "height": 900}, record_video_dir=str(OUT / "report_raw"),
                            record_video_size={"width": 1600, "height": 900})
        page = ctx.new_page()
        page.goto(Path(report).resolve().as_uri())
        page.wait_for_timeout(3000)
        for _ in range(9):
            page.mouse.wheel(0, 300)
            page.wait_for_timeout(1300)
        page.wait_for_timeout(1500)
        ctx.close()
        b.close()
    raw = max((OUT / "report_raw").glob("*.webm"), key=lambda f: f.stat().st_mtime)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(OUT / "report_recording.mp4")], check=True)
    shutil.rmtree(OUT / "report_raw", ignore_errors=True)
    print("recorded", OUT / "report_recording.mp4")


if __name__ == "__main__":
    main(sys.argv[1])
