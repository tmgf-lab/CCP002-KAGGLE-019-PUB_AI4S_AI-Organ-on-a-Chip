"""Assemble the demo video (max 5 min): real terminal runs, a real JupyterLab session (repository tour + walkthrough notebook),
the web demo, report figures and narration (Kokoro TTS, voice am_puck), with English subtitles burned in and English + Chinese
subtitle files for YouTube.

Nothing shown is generated imagery: every visual is a recording of a real run, a screen recording, a report figure, or a
title card with text. Recordings are replayed as recorded; only the speed changes, and the speed is shown on screen.
    bash demo_run/record_all.sh                          # real terminal runs -> /tmp/demo/*.cast (~12 min)
    ~/.venvs/gradio/bin/python demo_run/record_lab.py    # JupyterLab session -> video/lab_raw.mp4 + lab_events.json
    ~/.venvs/gradio/bin/python demo_run/record_demo.py            # web demo -> video/demo_recording.mp4
    ~/.venvs/kokoro/bin/python demo_run/build_video.py            # -> video/demo_video.mp4 (English burned in), youtube_captions_en.srt, youtube_captions_zh-Hans.srt
"""
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
VID = ROOT / "video"  # git-ignored
WORKV = VID / "build"
WORKV.mkdir(parents=True, exist_ok=True)
CASTS = Path("/tmp/demo")
R = json.loads((ROOT / "results" / "results.json").read_text())
W, H, FPS = 1920, 1080, 30
PIC_H = 950                      # picture area; the band below holds the burned-in subtitles
BG = "0xfcfcfb"
VOICE = "am_puck"
NICE = [1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def g(k):
    v = R.get(k, "—")
    return v.split(" [")[0] if isinstance(v, str) else v


# Each scene: visual + narration as (English, Chinese) sentence pairs (one subtitle per sentence).
SCENES = [
    ("card:Label-free, non-destructive readout\nof organ-on-a-chip experiments|Virtual staining · drug response · single cells · chip quality control",
     [("What if an organ-on-a-chip experiment could be read every day, without ever staining it?",
       "如果每天都能读取器官芯片实验的结果，而无需对其染色，会怎样？")]),
    ("card:The problem|Stains such as Cell Painting need fixed cells.\nUsually one snapshot per chip, at the end.\nBright-field is label-free, but hard to read.",
     [("Organ-on-a-chip systems test drugs on human-like tissue over days.", "器官芯片系统可在类人组织上连续多日测试药物。"),
      ("But they are usually read with fluorescent stains that need fixed cells, so each chip typically gives one snapshot.", "但它们通常使用需要固定细胞的荧光染色来读取，因此每块芯片通常只能得到一张快照。"),
      ("Bright-field imaging is label-free and non-destructive, but on its own it is hard to interpret.", "明场成像无需标记、不破坏样本，但单凭它很难解读。")]),
    ("card:One system, four questions|Q1  Predict five Cell Painting stains from bright-field\nQ2  Toxicity and drug response\n"
     "Q3  Single-cell phenotypes\nQ4  Chip quality control on real organ-on-a-chip images",
     [("One system was built around four questions, each answered by the same search engine.", "围绕四个问题构建了一个系统，每个问题都由同一个搜索引擎来回答。"),
      ("Stains, drug response, single cells and chip quality: all from bright-field alone.", "染色、药物反应、单细胞和芯片质量：全部仅基于明场图像。")]),
    ("lab:tour|Repository tour (real JupyterLab session)",
     [("The README gives a quick start, the inputs and outputs, and how to reproduce every result.", "README 提供了快速入门、输入与输出说明，以及如何复现每一个结果。"),
      ("Each question's search tree is a configuration file, and the score of every path is saved in the results folder.", "每个问题的搜索树都是一个配置文件，每条路径的得分都保存在 results 文件夹中。")]),
    ("cast:step1_data|Data · stream an unseen plate from the public Cell Painting Gallery",
     [("Here the pipeline streams wells of a plate the model never saw, straight from the Cell Painting Gallery.", "这里，流程直接从 Cell Painting Gallery 流式读取模型从未见过的一块培养板的孔位图像。"),
      ("Each field has five stains and three bright-field planes.", "每个视野包含五种染色和三个明场焦平面。")]),
    ("cast:step2_q1search|Q1 · guided path search (small demo tree on a laptop GPU)",
     [("Question one: predict the stains from bright-field.", "问题一：从明场图像预测染色。"),
      ("A path is one complete pipeline, one option per step; here, each is a network variant, trained first on a quarter of the data.", "一条路径是一个完整的流程，每一步选一个选项；这里每条路径是网络的一个变体，先在四分之一的数据上训练。"),
      ("The worse half is stopped; survivors move on to half, then all of the data, and a repeated path is skipped.", "较差的一半被停止；保留下来的路径继续使用一半、然后全部数据，重复的路径会被跳过。")]),
    ("cast:step3_q1test|Q1 · the full-scale winner on the unseen test plates",
     [("The winner of the full search now runs on the unseen test plates.", "完整搜索的最优模型现在在未见过的测试板上运行。"),
      ("On compounds never seen in training, virtual and real stains agree at a pixel correlation above 0.95 on these fields, and nuclei counts nearly match.",
       "对于训练中从未见过的化合物，在这些视野上虚拟染色与真实染色的像素相关系数高于 0.95，细胞核计数也几乎一致。")]),
    ("fig:fig_examples.png",
     [("Real stains above, stains predicted from bright-field alone below.", "上方为真实染色，下方为仅由明场图像预测的染色。"),
      ("A549 cells on top; a second cell line, U2OS, below: four stains transferred well, mitochondria did not.", "上面是 A549 细胞；下面是第二种细胞系 U2OS：四种染色迁移良好，线粒体则没有。")]),
    ("cast:step4_q2|Q2 · guided search: drug response from virtual-stain features",
     [("Question two: do virtual stains keep the drug response?", "问题二：虚拟染色能否保留药物反应？"),
      ("The same engine searches cleaning, variable selection, models and calibration.", "同一个引擎搜索数据清洗、变量选择、模型和校准方法。"),
      ("Each line is one path with its score at each stage; options that fail four times are excluded.", "每一行是一条路径及其在各阶段的得分；失败四次的选项会被排除。")]),
    ("fig:fig_q2_summary.png",
     [("Strong toxicity was detected even in bright-field.", "强毒性即使在明场图像中也能被检测到。"),
      (f"But for non-toxic drugs that still change the cells, bright-field reached an AUC of {g('q2_treated_vs_dmso_brightfield')}, "
       f"and virtual stains {g('q2_treated_vs_dmso_virtual')}, with no significant difference from real stains.",
       f"但对于无毒却仍改变细胞的药物，明场的 AUC 为 {g('q2_treated_vs_dmso_brightfield')}，虚拟染色为 {g('q2_treated_vs_dmso_virtual')}，与真实染色相比未检测到显著差异。")]),
    ("cast:step5_q3|Q3 · the same cells on real and virtual stains",
     [("Question three: single cells.", "问题三：单细胞。"),
      ("The winning segmentation runs on real and virtual stains of the same fields, and the cells are matched: an F1 of 0.98, with closely matching features and cell states.",
       "最优分割方法在同一视野的真实染色和虚拟染色上运行，并对细胞进行匹配：F1 为 0.98，细胞特征和细胞状态也高度一致。")]),
    ("cast:step6_q4|Q4 · chip quality control on 3,072 real organ-on-a-chip images",
     [("Question four uses 3,072 real organ-on-a-chip images, labelled good or bad.", "问题四使用 3,072 张真实器官芯片图像，标注为合格或不合格。"),
      ("Among the representations searched, self-supervised DINOv2 features came out on top.", "在搜索的图像表征中，自监督的 DINOv2 特征表现最好。"),
      (f"The full search reached an AUC of {g('q4_best')}, and {g('q4_loco')} on cell types it never saw.",
       f"完整搜索的 AUC 达到 {g('q4_best')}，在从未见过的细胞类型上为 {g('q4_loco')}。")]),
    ("fig:fig_replication.png",
     [("The winning paths were then repeated on a second cell line, U2OS, with nothing changed.", "随后，最优路径在第二种细胞系 U2OS 上被重复，未做任何改动。"),
      ("Toxicity and single-cell results held, at a lower level; mitochondria and the subtle drug effect did not replicate on a single plate.",
       "毒性和单细胞结果依然成立，但水平较低；线粒体和细微的药物效应在单块培养板上未能复现。")]),
    ("cast:step7_system@32|End-to-end system · one command, Q1 to Q4 (plate never used to fit any model)",
     [("Finally, everything runs as one system: bright-field images in, all four answers out.",
       "最后，所有部分作为一个系统运行：输入明场图像，输出全部四个问题的答案。"),
      ("On a plate no model was fitted on, it flags the toxic drug, estimates viability close to the real stains, and finds the cells and their states.",
       "在没有任何模型用其拟合的培养板上，它识别出有毒药物，估计的存活率接近真实染色结果，并找出细胞及其状态。"),
      ("Of two random held-out chips, it judges one correctly and misses the other, consistent with its held-out AUC of 0.80.",
       "对于两张随机选取的留出芯片，它正确判断了一张，误判了另一张，这与其在留出数据上 0.80 的 AUC 相符。")]),
    ("screenstart:report_recording.mp4",
     [("Every run writes a report: for each well the drug response and cell states, and for each field the input, the virtual stains with the cells, and the real stains as a check.",
       "每次运行都会生成一份报告：每个孔的药物反应和细胞状态，以及每个视野的输入图像、带细胞轮廓的虚拟染色和作为对照的真实染色。")]),
    ("lab:notebook|Walkthrough notebook (real JupyterLab session)",
     [("Reviewers can follow everything in the walkthrough notebook.", "评审者可以通过演练笔记本跟随全部流程。"),
      ("Each section runs a small real version of the same commands, then loads the full results.", "每个部分先运行相同命令的小规模真实版本，然后加载完整结果。"),
      ("It runs on a free CPU in under twenty minutes; here, long steps are fast-forwarded.", "它可以在免费 CPU 上运行，用时不到二十分钟；这里较长的步骤已加速播放。")]),
    ("app:demo_recording.mp4",
     [("The same system also runs as a small web app on your own computer: pick a compound for the stains, the drug response and the cells, or a chip photo for its quality.",
       "同一系统也可以作为小型网页应用在本地电脑上运行：选择一种化合物，即可得到染色、药物反应和细胞结果；或选择一张芯片照片，检查其质量。")]),
    ("card:Limits and next step|A new cell line needed its own training plate\nReal chips would likely need a stained calibration set\n"
     "Strong toxicity was detected without stains; subtle effects needed them",
     [("Virtual staining was learned per cell line, mitochondria did not transfer, and real chips would likely need a small stained calibration set.",
       "虚拟染色是针对每种细胞系学习的，线粒体未能迁移，真实芯片可能还需要一小组染色校准样本。")]),
    ("card:Image every day, stain once to calibrate|Team tmgf · code, weights, notebook and web demo: links in the Writeup",
     [("The proposed use, still to be tested on chips: image every chip every day, and stain once to calibrate.", "建议的用法（尚需在芯片上验证）：每天对每块芯片成像，只染色一次用于校准。"),
      ("Code, weights, the notebook and the web demo are all open. Thank you.", "代码、模型权重、笔记本和网页演示全部开放。谢谢。")]),
]


# ------------------------------------------------------------------------------------------ narration
def tts_lines(lines, out):
    """One TTS call per sentence; returns the scene audio length and each sentence's (start, end) for the subtitles."""
    from kokoro import KPipeline
    import soundfile as sf
    if not hasattr(tts_lines, "p"):
        tts_lines.p = KPipeline(lang_code="a")
    sr, gap, lead = 24000, 0.25, 0.3
    parts, times, t = [np.zeros(int(lead * sr), np.float32)], [], lead
    for en, _ in lines:
        a = np.concatenate([x.numpy() if hasattr(x, "numpy") else x for _, _, x in tts_lines.p(en, voice=VOICE)]).astype(np.float32)
        times.append((t, t + len(a) / sr))
        t += len(a) / sr + gap
        parts += [a, np.zeros(int(gap * sr), np.float32)]
    sf.write(out, np.concatenate(parts), sr)
    return t, times


# ------------------------------------------------------------------------------------------ visuals
def card(text, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    title, _, body = text.partition("|")
    fig = plt.figure(figsize=(W / 100, PIC_H / 100), dpi=100, facecolor="#fcfcfb")
    t = fig.text(0.08, 0.60, title, fontsize=50, fontweight="bold", color="#0b0b0b", va="bottom")
    r = fig.canvas.get_renderer()
    while t.get_window_extent(r).width > 0.86 * W and t.get_fontsize() > 20:  # never cut the title at the right edge
        t.set_fontsize(t.get_fontsize() - 2)
    fig.text(0.08, 0.53, body, fontsize=32, color="#52514e", va="top", linespacing=1.6)
    fig.add_artist(plt.Rectangle((0.08, 0.575), 0.12, 0.009, color="#eb6834", transform=fig.transFigure))
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


FIT = f"scale={W}:{PIC_H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(({PIC_H}-ih)/2):color={BG},setsar=1"


def badge(text):
    t = text.replace(":", r"\:")
    return (f"drawtext=fontfile={BOLD}:text='{t}':fontsize=26:fontcolor=white:box=1:boxcolor=0xeb6834:boxborderw=12:"
            f"x=w-tw-44:y={PIC_H}-th-36")


def run(cmd):
    subprocess.run(cmd, check=True)


def enc(out):
    return ["-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)]


def still(img, dur, out):
    run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(img), "-t", f"{dur:.2f}", "-vf", FIT, *enc(out)])


def app_scene(src, dur, out):
    """The web-demo recording from the moment the Q1-Q3 result appears (the CPU wait is skipped and said so), sped up to
    fit the narration; the speed is shown."""
    ev = json.loads((VID / "demo_events.json").read_text())
    start = max(0.0, ev["tab1_result"] - 1.0)
    total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                                 capture_output=True, text=True).stdout) - start
    speed = max(1.0, total / dur)
    label = f"REAL SESSION  ·  {ev['tab1_result']:.0f} s CPU wait skipped" + (f"  ·  x{speed:.1f} speed" if speed > 1.05 else "")
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{total:.2f}", "-vf",
         f"setpts=PTS/{speed:.3f},tpad=stop_mode=clone:stop_duration={dur:.2f},{FIT},{badge(label)}", "-t", f"{dur:.2f}", *enc(out)])


def screen(src, dur, out):
    """The last `dur` seconds of a screen recording (where the result is on screen); frozen first frame if it is shorter."""
    total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                                 capture_output=True, text=True).stdout)
    start = max(0.0, total - dur)
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{dur:.2f}",
         "-vf", f"tpad=start_mode=clone:start_duration={max(0.0, dur - total):.2f},{FIT}", *enc(out)])


def screen_start(src, dur, out):
    total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                                 capture_output=True, text=True).stdout)
    speed = max(1.0, total / dur)
    vf = f"setpts=PTS/{speed:.3f},tpad=stop_mode=clone:stop_duration={dur:.2f},{FIT}"
    if speed > 1.05:
        vf += "," + badge(f"REAL REPORT  ·  x{speed:.1f} speed")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", f"{dur:.2f}", "-vf", vf, *enc(out)])


def nice(x):
    return next((v for v in NICE if v >= x - 1e-9), NICE[-1])


def cast(name, title, dur, out):
    name, _, forced = name.partition("@")  # "step7_system@32": a fixed speed, so the final output stays on screen longer
    lines = (CASTS / f"{name}.cast").read_text().splitlines()
    real = json.loads(lines[-1])[0]
    speed = float(forced) if forced else nice(real / max(dur - 1.0, 1.0))
    hold = max(1.0, dur - real / speed)
    raw = WORKV / f"{name}_raw.mp4"
    run([str(Path.home() / ".venvs/rec/bin/python"), str(ROOT / "demo_run" / "render_cast.py"), str(CASTS / f"{name}.cast"),
         str(raw), "--title", title, "--speed", str(speed), "--hold", f"{hold:.2f}"])
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-t", f"{dur:.2f}", "-vf", FIT, *enc(out)])


def lab_piece(t0, t1, speed, label, out):
    src = VID / "lab_raw.mp4"
    vf = f"setpts=(PTS-STARTPTS)/{speed},{FIT},{badge(label + (f'  ·  x{speed:g} speed' if speed != 1 else '  ·  real time'))}"
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}", "-i", str(src), "-vf", vf, *enc(out)])
    return (t1 - t0) / speed


def lab(part, dur, out):
    ev = json.loads((VID / "lab_events.json").read_text())
    start = next(e["t"] for e in ev if e["kind"] == "start" and e["name"] == part)
    end = next(e["t"] for e in ev if e["kind"] == "end" and e["name"] == part)
    pieces = []
    if part == "tour":
        speed = max(1, nice((end - start) / dur))
        pieces.append(lab_piece(start, end, speed, "REAL SESSION", WORKV / "lab_tour_0.mp4"))
    else:
        cells = [(a["t"], b["t"]) for a, b in zip([e for e in ev if e["kind"] == "cell_start"], [e for e in ev if e["kind"] == "cell_end"])]
        short = [c for c in cells if c[1] - c[0] <= 8]
        long_ = [c for c in cells if c[1] - c[0] > 8]
        s_short = nice(max(1, sum(b - a for a, b in short) / (0.45 * dur)))
        rest = dur - sum(b - a for a, b in short) / s_short
        s_long = nice(sum(b - a for a, b in long_) / max(rest, 1))
        for k, (a, b) in enumerate(cells):
            sp = s_long if (b - a) > 8 else s_short
            pieces.append(lab_piece(a, b, sp, "REAL SESSION", WORKV / f"lab_nb_{k:02d}.mp4"))
    files = sorted(WORKV.glob(f"lab_{'tour' if part == 'tour' else 'nb'}_*.mp4"))
    (WORKV / "lab_list.txt").write_text("".join(f"file '{f}'\n" for f in files))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(WORKV / "lab_list.txt"),
         "-vf", f"tpad=stop_mode=clone:stop_duration={max(0.0, dur - sum(pieces)):.2f}", "-t", f"{dur:.2f}", *enc(out)])


# ------------------------------------------------------------------------------------------ assembly
def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    for f in WORKV.glob("lab_*.mp4"):
        f.unlink()
    segs, en, zh, t0, skipped = [], [], [], 0.0, []
    for i, (visual, lines) in enumerate(SCENES):
        kind, _, arg = visual.partition(":")
        need = {"cast": CASTS / f"{arg.split('|')[0].split('@')[0]}.cast", "lab": VID / "lab_events.json", "fig": FIG / arg,
                "screen": VID / arg, "screenstart": VID / arg, "app": VID / arg}.get(kind)
        if need is not None and not need.exists():
            skipped.append(visual)
            continue
        wav = WORKV / f"s{i:02d}.wav"
        a_len, times = tts_lines(lines, wav)
        dur = a_len + 0.6
        v = WORKV / f"s{i:02d}_v.mp4"
        if kind == "card":
            card(arg, WORKV / f"s{i:02d}.png")
            still(WORKV / f"s{i:02d}.png", dur, v)
        elif kind == "fig":
            still(FIG / arg, dur, v)
        elif kind == "screen":
            screen(VID / arg, dur, v)
        elif kind == "app":
            app_scene(VID / arg, dur, v)
        elif kind == "screenstart":  # from the start of the recording (sped up to fit the narration if longer)
            screen_start(VID / arg, dur, v)
        elif kind == "cast":
            name, title = arg.split("|", 1)
            cast(name, title, dur, v)
        elif kind == "lab":
            lab(arg.split("|")[0], dur, v)
        seg = WORKV / f"s{i:02d}.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(v), "-i", str(wav), "-filter_complex", "[1:a]apad[a]",
             "-map", "0:v", "-map", "[a]", "-t", f"{dur:.2f}", "-c:v", "copy", "-c:a", "aac", "-ar", "24000", str(seg)])
        segs.append(seg)
        for (a, b), (e, z) in zip(times, lines):
            en.append((t0 + a, t0 + b + 0.15, e))
            zh.append((t0 + a, t0 + b + 0.15, z))
        t0 += dur
    (WORKV / "list.txt").write_text("".join(f"file '{p}'\n" for p in segs))
    joined = WORKV / "joined.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(WORKV / "list.txt"), "-c", "copy", str(joined)])
    for lang, subs in (("en", en), ("zh-Hans", zh)):
        (VID / f"youtube_captions_{lang}.srt").write_text("\n".join(f"{k + 1}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n"
                                                              for k, (a, b, txt) in enumerate(subs)), encoding="utf-8")
    style = ("FontName=DejaVu Sans,FontSize=9.5,PrimaryColour=&H00FFFFFF,BackColour=&H99000000,BorderStyle=3,Outline=6,"
             "Shadow=0,MarginV=6,MarginL=40,MarginR=40")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined), "-vf",
         f"subtitles={VID / 'youtube_captions_en.srt'}:force_style='{style}'", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "copy", str(VID / "demo_video.mp4")])
    for f in VID.glob("demo_video*.srt"):  # no caption file named like the video: players would load it on top of the burned-in text
        f.unlink()
    print(f"video: {t0:.0f} s ({t0 / 60:.1f} min), {len(segs)} scenes -> {VID / 'demo_video.mp4'}; skipped: {skipped}")


if __name__ == "__main__":
    main()
