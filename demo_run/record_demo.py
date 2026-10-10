"""Screen-record a real session of the web demo (apps/web_demo/app.py) with a headless browser (Playwright).

Starts the app locally on CPU, opens it, selects a held-out compound, clicks "Stain virtually", waits for the real result
and scrolls through it. Output: video/demo_recording.mp4 (used by build_video.py).
    ~/.venvs/gradio/bin/python demo_run/record_demo.py
"""
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "web_demo"
OUT = ROOT / "video"
PORT = 7871


def wait_up(url, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except Exception:  # noqa: BLE001
            time.sleep(2)
    raise TimeoutError(url)


def main():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "GRADIO_SERVER_PORT": str(PORT), "GRADIO_ANALYTICS_ENABLED": "False"}
    app = subprocess.Popen([str(Path.home() / ".venvs/vcp/bin/python"), "app.py"], cwd=APP, env=env,
                           stdout=open(OUT / "app.log", "w"), stderr=subprocess.STDOUT)
    try:
        wait_up(f"http://127.0.0.1:{PORT}/")
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            ctx = b.new_context(viewport={"width": 1600, "height": 900}, record_video_dir=str(OUT / "raw"),
                                record_video_size={"width": 1600, "height": 900})
            page = ctx.new_page()
            page.goto(f"http://127.0.0.1:{PORT}/")
            page.wait_for_timeout(2500)
            import json
            import time
            t0 = time.time()
            page.get_by_role("button", name="Stain virtually").first.click()      # tab 1: Q1 + Q2 + Q3
            page.wait_for_selector("text=Q3 single cells", timeout=300000)
            t_result = time.time() - t0
            page.wait_for_timeout(3000)
            for _ in range(4):
                page.mouse.wheel(0, 300)
                page.wait_for_timeout(1800)
            page.wait_for_timeout(1500)
            page.mouse.wheel(0, -3000)
            page.get_by_role("tab", name="3 · Chip quality (Q4)").click()              # tab 3: Q4 on a held-out chip
            page.wait_for_timeout(1500)
            page.locator("button.gallery-item").nth(2).click()  # 230320_169 (bad): one of the 6 random held-out examples
            page.wait_for_timeout(1500)
            page.get_by_role("button", name="Check chip quality").click()
            page.wait_for_selector("text=P(good chip)", timeout=120000)
            page.wait_for_timeout(5000)
            (OUT / "demo_events.json").write_text(json.dumps({"tab1_result": round(t_result, 2)}))
            ctx.close()
            b.close()
        raw = sorted((OUT / "raw").glob("*.webm"))[-1]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(OUT / "demo_recording.mp4")], check=True)
        shutil.rmtree(OUT / "raw", ignore_errors=True)
        print("recorded", OUT / "demo_recording.mp4")
    finally:
        app.terminate()


if __name__ == "__main__":
    main()
