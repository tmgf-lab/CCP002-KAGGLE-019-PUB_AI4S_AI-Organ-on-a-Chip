"""Build Kaggle kernels (notebook + kernel-metadata.json) that embed the current code.

Each kernel folder is written to kaggle_jobs/kernels/<name>/ and pushed with
`kaggle kernels push -p kaggle_jobs/kernels/<name>`.

    python kaggle_jobs/build.py prep_a549 prep_u2os prep_chip
"""
import json
import re
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
USER = "tiagomgfernandes"
COMPETITION = "ai-4-s-open-innovation-artificial-intelligence-for-life-scien"


def embed(files):
    """%%writefile cells, so the kernel runs exactly the local code."""
    return [nbf.v4.new_code_cell(f"%%writefile {f}\n" + (CODE / f).read_text()) for f in files]


def kernel(name, title, cells, gpu=False, internet=True, kernel_sources=(), dataset_sources=()):
    out = HERE / "kernels" / name
    out.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                   "language": "python"}})
    import ast
    for i, c in enumerate(cells):  # catch broken generated code before it reaches Kaggle
        if c.cell_type == "code" and not c.source.lstrip().startswith(("%%", "!")):
            ast.parse(c.source, filename=f"{name} cell {i}")
    nbf.write(nb, out / f"{name}.ipynb")
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")  # Kaggle derives the URL from the title
    meta = {"id": f"{USER}/{slug}", "title": title, "code_file": f"{name}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": gpu, "enable_tpu": False, "enable_internet": internet,
            "dataset_sources": list(dataset_sources), "competition_sources": [COMPETITION],
            "kernel_sources": list(kernel_sources), "model_sources": []}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("built", out)


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
PREP_HEAD = [code("!pip -q install tifffile imagecodecs 2>/dev/null\n"
                  "import io, urllib.request, tifffile  # fail fast: decode one real LZW-compressed CPJUMP1 image\n"
                  "u = 'https://cellpainting-gallery.s3.amazonaws.com/cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/"
                  "images/BR00116991__2020-11-05T19_51_35-Measurement1/Images/r01c01f01p01-ch1sk1fk1fl1.tiff'\n"
                  "print('decode check:', tifffile.imread(io.BytesIO(urllib.request.urlopen(u).read())).shape)")]


def run(cmd):
    """Shell step that FAILS the notebook on a non-zero exit (a bare `!cmd` does not)."""
    return code(f"import subprocess\nsubprocess.run({cmd!r}, shell=True, check=True)")


def check_packed(folder, plates):
    return code(f"import numpy as np, os\n"
                f"for p in {plates!r}:\n"
                f"    a = np.load(f'{folder}/{{p}}.npy', mmap_mode='r')\n"
                f"    nz = sum(int(a[i].any()) for i in range(0, len(a), max(1, len(a) // 50)))\n"
                f"    print(p, a.shape, 'non-empty sampled fields:', nz)\n"
                f"    assert nz > 0.9 * len(range(0, len(a), max(1, len(a) // 50))), f'{{p}}: empty fields - download failed'")
SIZE = code("!du -sh /kaggle/working/* && ls -la /kaggle/working/*/ | head -40")

Q1_FILES = ["config.py", "data.py", "model.py", "sweep/__init__.py", "sweep/stats.py",
            "sweep/q1_translation.py", "sweep/configs/q1_translation.yaml"]
Q1_SETUP = [
    code("!pip -q install segmentation-models-pytorch optuna 2>&1 | tail -1\n"
         "import os; os.makedirs('sweep/configs', exist_ok=True)\n!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    *[nbf.v4.new_code_cell(f"%%writefile {f}\n" + (CODE / f).read_text()) for f in Q1_FILES],
    code("import glob, os, numpy as np\n"
         "def find(name):\n"
         "    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)\n"
         "    assert hits, f'{name} not found in /kaggle/input'\n"
         "    return os.path.dirname(hits[0])\n"
         "a549, u2os = find('BR00117015.npy'), find('BR00117024.npy')\n"
         "extra = glob.glob('/kaggle/input/**/BR00117016.npy', recursive=True)\n"
         "dirs = [u2os] + [os.path.dirname(extra[0])] * bool(extra)\n"
         "os.environ.update(VCP_DATA=a549, VCP_DATA_EXTRA=os.pathsep.join(dirs), VCP_RUNS='/kaggle/working/runs', VCP_WORKERS='3')\n"
         "for p, d in [('BR00117015', a549), ('BR00117024', u2os)]:\n"
         "    print(p, np.load(f'{d}/{p}.npy', mmap_mode='r').shape)"),
]


def run_parallel(*cmds, limit_min=600):
    """One process per GPU; progress printed every 2 min (visible with `kaggle kernels logs`);
    processes are killed after limit_min; the cell FAILS if any process fails."""
    return code("import subprocess, sys, time, psutil\n"
                f"cmds, limit = {list(cmds)!r}, {limit_min} * 60\n"
                "procs = [subprocess.Popen(c, shell=True, stdout=open(f'/kaggle/working/proc{i}.log', 'w'), stderr=subprocess.STDOUT)\n"
                "         for i, c in enumerate(cmds)]\n"
                "t0, killed = time.time(), False\n"
                "while any(p.poll() is None for p in procs):\n"
                "    time.sleep(120)\n"
                "    mem = psutil.virtual_memory()\n"
                "    print(f'[{(time.time()-t0)/60:.0f} min] RAM {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB', flush=True)\n"
                "    for i in range(len(procs)):\n"
                "        lines = [l for l in open(f'/kaggle/working/proc{i}.log').read().splitlines() if 'arn' not in l]\n"
                "        print(f'  proc{i}:', ' | '.join(lines[-2:])[-300:], flush=True)\n"
                "    if time.time() - t0 > limit:\n"
                "        [p.kill() for p in procs if p.poll() is None]; killed = True; print('TIME LIMIT - killed', flush=True)\n"
                "for i, p in enumerate(procs):\n"
                "    print(f'--- proc{i} exit {p.returncode}'); print(open(f'/kaggle/working/proc{i}.log').read()[-3000:])\n"
                "assert killed or all(p.returncode == 0 for p in procs), 'a process failed'  # time-limit kills are expected")


Q1 = "python -m sweep.q1_translation"
PREP_SOURCES = ["tiagomgfernandes/vcp-prep-a549-plates", "tiagomgfernandes/vcp-prep-u2os-plates"]

JOBS = {
    "q1_smoke": lambda: kernel(
        "q1_smoke", "VCP Q1 smoke and timing probe",
        [md("Q1 code check on Kaggle: GPU 0 runs a tiny search (4 paths, ~10 steps per phase); GPU 1 times 200 steps "
            "of the cheapest and most expensive branches."),
         *Q1_SETUP, code(f"!{Q1} init"),
         run_parallel(f"{Q1} search --gpu 0 --smoke --trials 4", f"{Q1} search --gpu 1 --smoke --trials 4", limit_min=40),
         code(f"!{Q1} report")],
        gpu=True, kernel_sources=PREP_SOURCES),

    "prep_a549": lambda: kernel(
        "prep_a549", "VCP prep A549 plates",
        [md("Downloads CPJUMP1 A549 48 h plates (train BR00117015, test BR00117017/19) from the public "
            "Cell Painting Gallery (CC0) and packs them per plate. CPU only."),
         *PREP_HEAD, *embed(["config.py", "download.py"]),
         run("VCP_DATA=/kaggle/working/cpjump1 python download.py --workers 32"),
         check_packed("/kaggle/working/cpjump1", ["BR00117015", "BR00117017", "BR00117019"]),
         code("!rm -f /kaggle/working/cpjump1/*_done.npy"), SIZE]),
    "prep_u2os": lambda: kernel(
        "prep_u2os", "VCP prep U2OS plates",
        [md("Downloads CPJUMP1 U2OS plates: 48 h test plates BR00117011/12 (sites 1,5 of 9) and the 24 h calibration "
            "plate BR00117024 (sites 1,5,9). U2OS plates have 9 sites per well. CPU only."),
         *PREP_HEAD, *embed(["config.py", "download.py"]),
         run("VCP_DATA=/kaggle/working/cpjump1_u2os python download.py --workers 32 --plates BR00117011 BR00117012 --sites 1 5"),
         run("VCP_DATA=/kaggle/working/cpjump1_u2os python download.py --workers 32 --plates BR00117024 --sites 1 5 9"),
         check_packed("/kaggle/working/cpjump1_u2os", ["BR00117011", "BR00117012", "BR00117024"]),
         code("!rm -f /kaggle/working/cpjump1_u2os/*_done.npy"), SIZE]),
    "prep_u2os_train": lambda: kernel(
        "prep_u2os_train", "VCP prep U2OS training plate",
        [md("U2OS replication study: 48 h training plate BR00117011 with sites 1, 5, 9 of 9 (3 sites per well, as the "
            "A549 training plate). Test plate BR00117012 is in the U2OS prep kernel. CPU only."),
         *PREP_HEAD, *embed(["config.py", "download.py"]),
         run("VCP_DATA=/kaggle/working/cpjump1_u2os_train python download.py --workers 32 --plates BR00117011 --sites 1 5 9"),
         check_packed("/kaggle/working/cpjump1_u2os_train", ["BR00117011"]),
         code("!rm -f /kaggle/working/cpjump1_u2os_train/*_done.npy"), SIZE]),
    "prep_extra": lambda: kernel(
        "prep_extra", "VCP prep A549 extra plate",
        [md("Second A549 48 h replicate training plate BR00117016 (sites 1, 6, 11), CPU only."),
         *PREP_HEAD, *embed(["config.py", "download.py"]),
         run("VCP_DATA=/kaggle/working/cpjump1_extra python download.py --workers 32 --plates BR00117016 --sites 1 6 11"),
         check_packed("/kaggle/working/cpjump1_extra", ["BR00117016"]),
         code("!rm -f /kaggle/working/cpjump1_extra/*_done.npy"), SIZE]),
    "prep_chip": lambda: kernel(
        "prep_chip", "VCP prep organ-on-a-chip images",
        [md("Downloads the Organ-on-a-Chip (OOC) Image Dataset (Zenodo 10203721, CC BY 4.0). Kept zipped: "
            "notebook outputs are limited in file count; jobs unzip to local disk at start."),
         code("!mkdir -p /kaggle/working/ooc && cd /kaggle/working/ooc && "
              "curl -sL --retry 10 -o OOC_datasheet.xlsx 'https://zenodo.org/records/10203721/files/OOC_datasheet.xlsx?download=1' && "
              "curl -sL --retry 10 -C - -o OOC_image_dataset.zip 'https://zenodo.org/records/10203721/files/OOC_image_dataset.zip?download=1'"),
         code("!cd /kaggle/working/ooc && python -c \"import zipfile; z=zipfile.ZipFile('OOC_image_dataset.zip'); "
              "print(len(z.namelist()), 'entries; test', z.testzip())\""),
         SIZE]),
    "q1_search": lambda: kernel(
        "q1_search", "VCP Q1 tree search",
        [md("Q1 guided tree search (Optuna TPE + successive halving), 2 T4 GPUs in parallel. "
            "Paths: sweep/configs/q1_translation.yaml. Selection uses validation fields only."),
         *Q1_SETUP, code(f"!{Q1} init"),
         run_parallel(f"{Q1} search --gpu 0 --trials 20 --hours 6.5", f"{Q1} search --gpu 1 --trials 20 --hours 6.5",
                      limit_min=7 * 60),
         code(f"!{Q1} report")],
        gpu=True, kernel_sources=PREP_SOURCES),
    "q1_final": lambda: kernel(
        "q1_final", "VCP Q1 final seeds",
        [md("Q1 final stage: top 3 search paths x 3 seeds and the baseline x 3 seeds, full training. "
            "Continues from the search notebook's output (study database)."),
         *Q1_SETUP,
         code("import shutil, glob\n"
              "src = glob.glob('/kaggle/input/**/study.db', recursive=True)[0]\n"
              "dst = '/kaggle/working/runs/q1_translation'\n"
              "shutil.copytree(os.path.dirname(src), dst, dirs_exist_ok=True)\n"
              "print('study copied from', src)"),
         code("for grp in ('final', 'baseline'):  # runs finished by the previous (interrupted) final job\n"
              "    src = glob.glob(f'/kaggle/input/**/{grp}/*/train_log.json', recursive=True)\n"
              "    for f in src:\n"
              "        d = os.path.dirname(f)\n"
              "        shutil.copytree(d, f'/kaggle/working/runs/q1_translation/{grp}/' + os.path.basename(d), dirs_exist_ok=True)\n"
              "print(sorted(os.listdir('/kaggle/working/runs/q1_translation/final')), os.listdir('/kaggle/working/runs/q1_translation/baseline'))"),
         run_parallel(f"{Q1} final --gpu 0 --top 3 --seeds 1", f"{Q1} final --gpu 1 --top 3 --seeds 2", limit_min=240),
         run_parallel(f"{Q1} variants --gpu 0 --seeds 1", f"{Q1} variants --gpu 1 --seeds 2", limit_min=240),
         code(f"!{Q1} choose --gpu 0 && {Q1} report && cat /kaggle/working/runs/q1_translation/q1_winner.json")],
        gpu=True, dataset_sources=["tiagomgfernandes/vcp-q1-search-study", "tiagomgfernandes/vcp-q1-final-partial"],
        kernel_sources=PREP_SOURCES + ["tiagomgfernandes/vcp-prep-a549-extra-plate"]),
}

Q2_FILES = ["sweep/__init__.py", "sweep/engine.py", "sweep/stats.py", "sweep/q2_models.py", "sweep/q2_toxicity.py",
            "sweep/q2_search.py", "sweep/configs/q2_toxicity.yaml"]


def q2_setup():
    return [code("!pip -q install optuna xgboost 2>&1 | tail -1\nimport os, glob; os.makedirs('sweep/configs', exist_ok=True)"),
            *[nbf.v4.new_code_cell(f"%%writefile {f}\n" + (CODE / f).read_text()) for f in Q2_FILES],
            code("feat = glob.glob('/kaggle/input/**/base/site_features.csv', recursive=True)\n"
                 "assert feat, 'site_features.csv not found'\n"
                 "os.environ.update(Q2_FEATURES_PATH=feat[0], Q2_OUT='/kaggle/working/q2', OMP_NUM_THREADS='1')\n"
                 "print(feat[0])")]


def sh(cmd):
    """Shell step with noisy lines filtered; pipefail makes the step FAIL when the command fails (not grep's status)."""
    flt = "warn|Trial [0-9]+ (pruned|finished)|existing study|new study"
    return code("import subprocess\n"
                f"subprocess.run('set -o pipefail; ' + {cmd!r} + ' 2>&1 | grep -viE \"{flt}\"', "
                "shell=True, check=True, executable='/bin/bash')")


def q2_search_job(name, title, labels):
    return lambda: kernel(name, title, [md(f"Q2 guided searches for labels {labels} x 4 sources (CPU)."), *q2_setup(),
        sh(f"python -m sweep.q2_search all --workers 4 --budget 250 --hours 1.1 --labels {' '.join(labels)}")],
        gpu=False, dataset_sources=["tiagomgfernandes/vcp-artifacts"])


JOBS.update({
    "q2_a": q2_search_job("q2_a", "VCP Q2 search toxicity", ["toxic_0.5", "toxic_0.8"]),
    "q2_b": q2_search_job("q2_b", "VCP Q2 search viability treated", ["viability", "treated_vs_dmso"]),
    "q2_assess": lambda: kernel("q2_assess", "VCP Q2 assess", [md("Q2 assessment over all 16 searches."), *q2_setup(),
        code("import shutil\nfor f in glob.glob('/kaggle/input/**/q2/searches/*', recursive=True):\n"
             "    os.makedirs('/kaggle/working/q2/searches', exist_ok=True); shutil.copy(f, '/kaggle/working/q2/searches/')\n"
             "print(len(os.listdir('/kaggle/working/q2/searches')), 'search files')"),
        sh("python -m sweep.q2_search assess")],
        gpu=False, dataset_sources=["tiagomgfernandes/vcp-artifacts"],
        kernel_sources=["tiagomgfernandes/vcp-q2-search-toxicity", "tiagomgfernandes/vcp-q2-search-viability-treated"]),
})

Q4_FILES = ["sweep/__init__.py", "sweep/engine.py", "sweep/stats.py", "sweep/q4_parts.py", "sweep/q4_chip_qc.py",
            "sweep/configs/q4_chip_qc.yaml"]
Q3_FILES = ["config.py", "data.py", "model.py", "evaluate.py", "sweep/__init__.py", "sweep/engine.py", "sweep/stats.py",
            "sweep/q3_single_cell.py", "sweep/q3_search.py", "sweep/configs/q3_single_cell.yaml"]


def setup(files, pip):
    return [code(f"!pip -q install {pip} 2>&1 | tail -1\nimport os, glob, shutil, subprocess; os.makedirs('sweep/configs', exist_ok=True)"),
            *[nbf.v4.new_code_cell(f"%%writefile {f}\n" + (CODE / f).read_text()) for f in files]]


Q4_ENV = code("virt = glob.glob('/kaggle/input/**/chip1/chip_s2/chip_features.csv', recursive=True)[0]\n"
              "os.environ.update(OMP_NUM_THREADS='1')\nprint(virt)")
Q3_ENV = code("def find(name):\n    h = glob.glob(f'/kaggle/input/**/{name}', recursive=True); assert h, name; return os.path.dirname(h[0])\n"
              "a549, base = find('BR00117017.npy'), find('base/best.pt')\n"
              "os.environ.update(VCP_DATA=a549, VCP_RUNS=os.path.dirname(base), Q3_OUT='/kaggle/working/q3', "
              "Q3_CACHE='/tmp/q3cache', OMP_NUM_THREADS='1')\nprint(a549, base)")

JOBS.update({
    "q4_gpu": lambda: kernel("q4_gpu", "VCP Q4 embeddings and finetune",
        [md("Q4 GPU: embeddings for all representations x preprocessings, then the fine-tuned ResNet18 search (24 paths)."),
         *setup(Q4_FILES, "optuna open_clip_torch timm xgboost"), Q4_ENV,
         sh("python -m sweep.q4_chip_qc embed --emb-dir /kaggle/working/q4/emb --virt-csv ' + virt + '"),
         sh("python -m sweep.q4_chip_qc finetune --emb-dir /kaggle/working/q4/emb --out /kaggle/working/q4/finetune --budget 24 --workers 2 --hours 3")],
        gpu=True, dataset_sources=["tiagomgfernandes/vcp-artifacts"], kernel_sources=["tiagomgfernandes/vcp-prep-organ-on-a-chip-images"]),
    "q4_cpu": lambda: kernel("q4_cpu", "VCP Q4 search and assess",
        [md("Q4 CPU: classifier search (1,000 paths) on cached embeddings, then the assessment."),
         *setup(Q4_FILES, "optuna xgboost"), Q4_ENV,
         code("src = glob.glob('/kaggle/input/**/q4/emb', recursive=True)[0]\nshutil.copytree(src, '/kaggle/working/q4/emb', dirs_exist_ok=True)\n"
              "ft = glob.glob('/kaggle/input/**/q4/finetune', recursive=True)\n"
              "ft and shutil.copytree(ft[0], '/kaggle/working/q4/finetune', dirs_exist_ok=True)\nprint(src, ft)"),
         sh("python -m sweep.q4_chip_qc search --emb-dir /kaggle/working/q4/emb --out /kaggle/working/q4/search --budget 1000 --workers 4 --hours 9 --virt-csv ' + virt + '"),
         sh("python -m sweep.q4_chip_qc assess --emb-dir /kaggle/working/q4/emb --out /kaggle/working/q4/search --finetune-dir /kaggle/working/q4/finetune --virt-csv ' + virt + '")],
        gpu=False, dataset_sources=["tiagomgfernandes/vcp-artifacts"],
        kernel_sources=["tiagomgfernandes/vcp-prep-organ-on-a-chip-images", "tiagomgfernandes/vcp-q4-embeddings-and-finetune"]),
    "q3_gpu": lambda: kernel("q3_gpu", "VCP Q3 segmentation search",
        [md("Q3 GPU: virtual stains for 1,536 test fields, segmentation search (50 paths, 2 T4), per-cell features of the winner."),
         *setup(Q3_FILES, "optuna 'cellpose<4' mahotas stardist"), Q3_ENV,
         sh("python -m sweep.q3_search predict"),
         sh("python -m sweep.q3_search segsearch --workers 2 --hours 4"),
         sh("python -m sweep.q3_search features")],
        gpu=True, dataset_sources=["tiagomgfernandes/vcp-artifacts"], kernel_sources=["tiagomgfernandes/vcp-prep-a549-plates"]),
    "q3_feat": lambda: kernel("q3_feat", "VCP Q3 cell features",
        [md("Q3 GPU: virtual stains + per-cell features for the winning segmentation path of the search (F1 0.977)."),
         *setup(Q3_FILES, "optuna 'cellpose<4' mahotas"), Q3_ENV,
         code("import json\nos.environ['Q3_SEG_PATH'] = json.dumps({'channels': 'all_five', 'preprocessing': 'none', "
              "'seg_model': 'cellpose_cyto3', 'diameter': '30', 'flow_threshold': '0.4', 'cellprob_threshold': '-1'})"),
         sh("python -m sweep.q3_search predict"),
         sh("python -m sweep.q3_search features")],
        gpu=True, dataset_sources=["tiagomgfernandes/vcp-artifacts"], kernel_sources=["tiagomgfernandes/vcp-prep-a549-plates"]),
    "q3_cpu": lambda: kernel("q3_cpu", "VCP Q3 downstream and assess",
        [md("Q3 CPU: downstream search (200 paths), supervised per-cell search (30 per source), assessment."),
         *setup(Q3_FILES, "optuna umap-learn leidenalg igraph mahotas"), Q3_ENV,
         code("os.makedirs('/kaggle/working/q3', exist_ok=True)\nfor f in ['cells.parquet', 'seg_fields.json']:\n    shutil.copy(glob.glob(f'/kaggle/input/**/q3/{f}', recursive=True)[0], '/kaggle/working/q3/')\n"
              "shutil.copytree(glob.glob('/kaggle/input/**/q3/seg', recursive=True)[0], '/kaggle/working/q3/seg', dirs_exist_ok=True)"),
         sh("python -m sweep.q3_search downstream --workers 4 --hours 5"),
         sh("python -m sweep.q3_search supervised --workers 4 --hours 1.5"),
         sh("python -m sweep.q3_search assess")],
        gpu=False, dataset_sources=["tiagomgfernandes/vcp-artifacts"],
        kernel_sources=["tiagomgfernandes/vcp-prep-a549-plates", "tiagomgfernandes/vcp-q3-segmentation-search",
                        "tiagomgfernandes/vcp-q3-cell-features"]),
})

JOBS["q1_test"] = lambda: kernel(
    "q1_test", "VCP Q1 test evaluation",
    [md("Q1 winner evaluated ONCE on the unseen test plates (BR00117017/19) with its chosen inference mode."),
     *Q1_SETUP, nbf.v4.new_code_cell("%%writefile evaluate.py\n" + (CODE / "evaluate.py").read_text()),
     code("import shutil\nsrc = os.path.dirname(glob.glob('/kaggle/input/**/q1_translation/q1_winner.json', recursive=True)[0])\n"
          "shutil.copytree(src, '/kaggle/working/runs/q1_translation', dirs_exist_ok=True)\n"
          "print(open('/kaggle/working/runs/q1_translation/q1_winner.json').read()[:1500])"),
     sh("python evaluate.py --winner /kaggle/working/runs/q1_translation/q1_winner.json --out /kaggle/working/q1_test --workers 4"),
     code("print(open('/kaggle/working/q1_test/report.json').read()[:4000])")],
    gpu=True, kernel_sources=PREP_SOURCES + ["tiagomgfernandes/vcp-q1-final-seeds"])

U2OS_ENV = code(
    "import glob, os, shutil, json, subprocess, numpy as np\n"
    "def plate_dir(name, n):  # the input folder whose copy of the plate has n fields\n"
    "    for h in glob.glob(f'/kaggle/input/**/{name}.npy', recursive=True):\n"
    "        if len(np.load(h, mmap_mode='r')) == n:\n"
    "            return os.path.dirname(h)\n"
    "    raise FileNotFoundError(name)\n"
    "train, test = plate_dir('BR00117011', 1152), plate_dir('BR00117012', 768)  # 3 and 2 sites per well\n"
    "os.environ.update(VCP_DATA=train, VCP_DATA_EXTRA=test, VCP_RUNS='/kaggle/working/runs', VCP_WORKERS='3',\n"
    "                  VCP_TRAIN_PLATES='BR00117011', VCP_TEST_PLATES='BR00117012')\n"
    "print('train', train, 'test', test)")

JOBS["u2os_q1"] = lambda: kernel(
    "u2os_q1", "VCP U2OS replication Q1",
    [md("U2OS replication study (second dataset, separate from the A549 study): the A549 Q1 winning path - same settings, "
        "seeds and inference mode, nothing re-chosen - trained on U2OS 48 h plate BR00117011 and tested once on BR00117012. "
        "Virtual stains of the test plate are saved for Q3."),
     *Q1_SETUP[:-1], nbf.v4.new_code_cell("%%writefile evaluate.py\n" + (CODE / "evaluate.py").read_text()), U2OS_ENV,  # [:-1]: no A549 env
     code("src = os.path.dirname(glob.glob('/kaggle/input/**/q1_translation/q1_winner.json', recursive=True)[0])\n"
          "shutil.copytree(src, '/kaggle/working/runs/q1_translation', dirs_exist_ok=True)\n"
          "w = json.load(open('/kaggle/working/runs/q1_translation/q1_winner.json'))\n"
          "seeds = [json.load(open(os.path.join(d, 'train_log.json')))['args']['seed'] for d in w['run_dirs']]\n"
          "print(w['config'], w['inference'], 'seeds', seeds)\n"
          "WIN = '/kaggle/working/runs/q1_translation/q1_winner.json'\n"
          "Q1 = 'python -m sweep.q1_translation replicate --winner ' + WIN\n"
          "cmds = [f'{Q1} --gpu {g} --seeds ' + ' '.join(map(str, seeds[g::2])) for g in (0, 1) if seeds[g::2]]\n"
          "procs = [subprocess.Popen(c, shell=True, stdout=open(f'/kaggle/working/rep{i}.log', 'w'), stderr=subprocess.STDOUT)\n"
          "         for i, c in enumerate(cmds)]\n"
          "[p.wait() for p in procs]\n"
          "for i in range(len(procs)):\n"
          "    print(''.join(l for l in open(f'/kaggle/working/rep{i}.log') if 'arn' not in l)[-2500:])"),
     sh("python -m sweep.q1_translation replicate --gpu 0 --winner /kaggle/working/runs/q1_translation/q1_winner.json"),
     sh("python evaluate.py --winner /kaggle/working/runs/q1_replicate/q1_winner.json --out /kaggle/working/u2os_test "
        "--workers 4 --save-virtual /kaggle/working/u2os_virtual"),
     code("print(open('/kaggle/working/runs/q1_replicate/q1_winner.json').read())\n"
          "print(open('/kaggle/working/u2os_test/report.json').read()[:3000])\n"
          "shutil.rmtree('/kaggle/working/runs/q1_translation/trials', ignore_errors=True)")],
    gpu=True, kernel_sources=["tiagomgfernandes/vcp-prep-u2os-training-plate", "tiagomgfernandes/vcp-prep-u2os-plates",
                              "tiagomgfernandes/vcp-q1-final-seeds"])

JOBS["u2os_q3"] = lambda: kernel(
    "u2os_q3", "VCP U2OS replication Q3",
    [md("U2OS replication study, Q3: the A549 winning segmentation path on real and virtual U2OS test fields (virtual = "
        "U2OS replication model), then the assessment with the A549 winning downstream and per-cell classifier paths."),
     *setup(Q3_FILES, "optuna 'cellpose<4' mahotas umap-learn leidenalg igraph"), U2OS_ENV,
     code("pf = '/kaggle/working/q3_a549_paths'\n"
          "for f in glob.glob('/kaggle/input/**/q3_*_paths.csv', recursive=True):  # flat dataset -> search folders\n"
          "    sub = {'q3_down': 'down', 'q3_sup_virtual': 'sup_virtual', 'q3_sup_real': 'sup_real'}[os.path.basename(f).rsplit('_paths', 1)[0]]\n"
          "    os.makedirs(f'{pf}/{sub}', exist_ok=True); shutil.copy(f, f'{pf}/{sub}/')\n"
          "seg = open(glob.glob('/kaggle/input/**/q3_seg_winner.json', recursive=True)[0]).read()\n"
          "virt = os.path.dirname(glob.glob('/kaggle/input/**/u2os_virtual/BR00117012_0000.npy', recursive=True)[0])\n"
          "os.makedirs('/tmp/q3cache', exist_ok=True); os.path.exists('/tmp/q3cache/virtual') or os.symlink(virt, '/tmp/q3cache/virtual')\n"
          "os.environ.update(Q3_OUT='/kaggle/working/u2os_q3', Q3_CACHE='/tmp/q3cache', Q3_SEG_PATH=seg, OMP_NUM_THREADS='1')\n"
          "print(seg, virt, os.listdir(pf))"),
     sh("python -m sweep.q3_search features"),
     sh("python -m sweep.q3_search assess --paths-from /kaggle/working/q3_a549_paths")],
    gpu=True, dataset_sources=["tiagomgfernandes/vcp-q3-a549-v2-paths"],
    kernel_sources=["tiagomgfernandes/vcp-prep-u2os-training-plate", "tiagomgfernandes/vcp-prep-u2os-plates",
                    "tiagomgfernandes/vcp-u2os-replication-q1"])

JOBS["q3_v2_gpu"] = lambda: kernel(
    "q3_v2_gpu", "VCP Q3 segmentation search on Q1 winner",
    [md("Q3 rerun on the Q1 winner's virtual stains (seed ensemble + TTA): virtual stains for the 1,536 A549 test fields, "
        "segmentation search (50 paths, 2 T4, same settings as the first Q3 search), per-cell features of the winner."),
     *setup(Q3_FILES, "optuna 'cellpose<4' mahotas stardist segmentation-models-pytorch"),
     code("def find(name):\n    h = glob.glob(f'/kaggle/input/**/{name}', recursive=True); assert h, name; return os.path.dirname(h[0])\n"
          "src = find('q1_translation/q1_winner.json')\n"
          "shutil.copytree(src, '/kaggle/working/runs/q1_translation', dirs_exist_ok=True)\n"
          "os.environ.update(VCP_DATA=find('BR00117017.npy'), VCP_RUNS='/kaggle/working/runs', Q3_OUT='/kaggle/working/q3', "
          "Q3_CACHE='/tmp/q3cache', OMP_NUM_THREADS='1')\n"
          "print(open('/kaggle/working/runs/q1_translation/q1_winner.json').read()[:600])"),
     sh("python -m sweep.q3_search predict --winner /kaggle/working/runs/q1_translation/q1_winner.json"),
     sh("python -m sweep.q3_search segsearch --workers 2 --hours 4"),
     sh("python -m sweep.q3_search features"),
     code("shutil.rmtree('/kaggle/working/runs', ignore_errors=True)  # weights are in the Q1 final notebook already")],
    gpu=True, kernel_sources=["tiagomgfernandes/vcp-prep-a549-plates", "tiagomgfernandes/vcp-q1-final-seeds"])

if __name__ == "__main__":
    for name in sys.argv[1:] or JOBS:
        JOBS[name]()
