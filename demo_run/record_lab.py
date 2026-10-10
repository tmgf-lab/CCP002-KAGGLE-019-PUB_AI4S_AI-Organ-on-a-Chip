"""Screen-record a real JupyterLab session on the repository: a short tour (file tree, README, a search tree, results), then
the walkthrough notebook executed cell by cell. Writes video/lab_raw.mp4 and video/lab_events.json (when each part and each
cell started and ended, used by build_video.py to fast-forward long-running cells, with the speed shown on screen).

    ~/.venvs/gradio/bin/python demo_run/record_lab.py
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("VCP_REPO", ROOT))  # record from a local-disk copy: NAS timestamps upset JupyterLab autosave
OUT = ROOT / "video"
PORT = 8899
VCP_PY = str(Path.home() / ".venvs/vcp/bin/python")


def wait_up(url, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise TimeoutError(url)


def main():
    shutil.rmtree("/tmp/vcp_walkthrough", ignore_errors=True)
    shutil.rmtree(OUT / "lab_raw", ignore_errors=True)  # never pick up a recording from an earlier attempt
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONWARNINGS": "ignore"}
    lab = subprocess.Popen([VCP_PY, "-m", "jupyterlab", "--no-browser", f"--port={PORT}", "--ServerApp.token=",
                            "--ServerApp.password=", "--LabApp.expose_app_in_browser=True", f"--ServerApp.root_dir={REPO}",
                            "--LabApp.app_dir=" + str(Path.home() / "anaconda3/share/jupyter/lab")],
                           env=env, stdout=open(OUT / "lab.log", "w"), stderr=subprocess.STDOUT)
    events = []
    try:
        wait_up(f"http://127.0.0.1:{PORT}/lab")
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            ctx = b.new_context(viewport={"width": 1600, "height": 900}, record_video_dir=str(OUT / "lab_raw"),
                                record_video_size={"width": 1600, "height": 900})
            page = ctx.new_page()
            t0 = time.time()

            def mark(kind, name):
                events.append({"t": round(time.time() - t0, 2), "kind": kind, "name": name})

            def cmd(c, args=None):
                page.evaluate("([c, a]) => (window.jupyterapp || window.jupyterlab).commands.execute(c, a || {})", [c, args])

            page.goto(f"http://127.0.0.1:{PORT}/lab?reset")
            page.wait_for_selector(".jp-DirListing-content", timeout=120000)
            page.wait_for_timeout(2500)
            try:  # "Would you like to receive official Jupyter news?" -> No
                page.locator(".jp-toast-buttonBar button", has_text="No").first.click(timeout=4000)
            except Exception:  # noqa: BLE001
                pass
            cmd("application:close-all")
            page.wait_for_timeout(1000)
            mark("start", "tour")
            # 1. README rendered
            cmd("docmanager:open", {"path": "README.md", "factory": "Markdown Preview"})
            page.wait_for_timeout(3000)
            for _ in range(10):
                page.mouse.move(1000, 500)
                page.mouse.wheel(0, 260)
                page.wait_for_timeout(1300)
            # 2. a search tree (Q2 config) and the results folder
            cmd("docmanager:open", {"path": "sweep/configs/q2_toxicity.yaml"})
            page.wait_for_timeout(4500)
            cmd("filebrowser:go-to-path", {"path": "results/a549/q2"})
            page.wait_for_timeout(2500)
            cmd("docmanager:open", {"path": "results/a549/q2/q2_best_paths.csv"})
            page.wait_for_timeout(4500)
            cmd("filebrowser:go-to-path", {"path": "notebooks"})
            mark("end", "tour")
            # 3. the walkthrough, cell by cell
            cmd("application:close-all")
            page.wait_for_timeout(1000)
            cmd("docmanager:open", {"path": "notebooks/walkthrough.ipynb"})
            page.wait_for_selector(".jp-Notebook", timeout=60000)
            page.wait_for_timeout(6000)  # kernel start
            n = page.evaluate("(window.jupyterapp || window.jupyterlab).shell.currentWidget.content.widgets.length")
            mark("start", "notebook")
            for i in range(n):
                kind = page.evaluate(f"(window.jupyterapp || window.jupyterlab).shell.currentWidget.content.widgets[{i}].model.type")
                mark("cell_start", f"{i}:{kind}")
                cmd("notebook:run-cell-and-select-next")
                if kind == "code":
                    page.wait_for_timeout(800)
                    while page.evaluate("(window.jupyterapp || window.jupyterlab).shell.currentWidget.sessionContext.session.kernel.status") != "idle":
                        page.wait_for_timeout(500)
                    page.wait_for_timeout(1800)  # let the output be seen
                else:
                    page.wait_for_timeout(2500)  # time to read the rendered text
                mark("cell_end", f"{i}:{kind}")
            mark("end", "notebook")
            page.wait_for_timeout(2000)
            ctx.close()
            b.close()
        raw = max((OUT / "lab_raw").glob("*.webm"), key=lambda f: f.stat().st_mtime)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", str(OUT / "lab_raw.mp4")], check=True)  # seekable copy for the video build
        shutil.rmtree(OUT / "lab_raw", ignore_errors=True)
        (OUT / "lab_events.json").write_text(json.dumps(events, indent=1))
        print("recorded", OUT / "lab_raw.mp4", f"{events[-1]['t']:.0f} s,", len(events), "events")
    finally:
        lab.terminate()


if __name__ == "__main__":
    main()
