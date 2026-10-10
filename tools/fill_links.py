"""Put the public links into every file that mentions them, then rebuild what depends on them.

1. Fill VALUES below (leave "" to skip a link for now).
2. python tools/fill_links.py
3. Re-run any time: placeholders already filled are no longer found.
"""
import subprocess
import sys
from pathlib import Path

VALUES = {
    "[GITHUB URL]": "https://github.com/tmgf-lab/CCP002-KAGGLE-019-PUB_AI4S_AI-Organ-on-a-Chip",
    "[KAGGLE NOTEBOOK URL]": "https://www.kaggle.com/code/tiagomgfernandes/organ-on-a-chip-virtual-staining-walkthrough",
    "[YOUTUBE URL]": "https://youtu.be/ddly-1rWOGY",
}

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", "docs/technical_report.md", "weights.py", "notebooks/build_walkthrough.py",
         "apps/web_demo/app.py", "apps/web_demo/README.md",
         "../kaggle_writeup/WRITEUP_FIELDS.md"]   # the Writeup lives outside the repository

changed = []
for rel in FILES:
    p = ROOT / rel
    if not p.exists():
        continue
    s = p.read_text()
    new = s
    for k, v in VALUES.items():
        if v:
            new = new.replace(k, v)
    if VALUES["[YOUTUBE URL]"]:  # on-screen text shown in the video before the link existed
        u = VALUES["[YOUTUBE URL]"]
        new = new.replace("see the Kaggle Writeup<!--YOUTUBE-->", f"[{u}]({u})")
    if new != s:
        p.write_text(new)
        changed.append(rel)
print("updated:", ", ".join(changed) or "nothing")

subprocess.run([sys.executable, "notebooks/build_walkthrough.py"], cwd=ROOT, check=True)
subprocess.run(["bash", "docs/build_pdf.sh"], cwd=ROOT, check=True)

left = subprocess.run(["grep", "-rnoE", r"\[(GITHUB URL|KAGGLE NOTEBOOK URL|YOUTUBE URL)\]",
                       *[f for f in FILES if (ROOT / f).exists()], "notebooks/walkthrough.ipynb"],
                      cwd=ROOT, capture_output=True, text=True).stdout
print("still to fill:\n" + left if left else "all links filled")
