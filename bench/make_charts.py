"""Render the README charts from the measured benchmark JSON. Regenerate with: python bench/make_charts.py"""
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "assets")
INK, MUTED, WIN, LOSE, GRID = "#15201c", "#5b6a64", "#0e7a6b", "#9aa9a3", "#d8dfdb"
NAMES = {"soliton+arena": "Soliton\n(arena)", "soliton": "Soliton", "pytorch": "PyTorch", "jax": "JAX",
         "tensorflow": "TensorFlow"}


def load(regime, exp):
    with open(os.path.join(HERE, "results", regime, f"{exp}.json")) as f:
        return json.load(f)


def style(ax, title, subtitle):
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, loc="left", pad=14)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def bars(ax, labels, values, fmt, best="min"):
    win = (min if best == "min" else max)(values)
    colors = [WIN if v == win else LOSE for v in values]
    b = ax.bar(labels, values, color=colors, width=0.62)
    for rect, v in zip(b, values):
        ax.text(rect.get_x() + rect.get_width() / 2, v, fmt(v), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(0, max(values) * 1.18)


def chart_memory():
    rows = {r["framework"]: r for r in load("fp32", "e1") if r["batch"] == 8}
    order = ["soliton+arena", "soliton", "pytorch", "tensorflow", "jax"]
    vals, labels = [], []
    for k in order:
        r = rows.get(k)
        if not r:
            continue
        v = r.get("native_peak") or r.get("gpu_peak")
        vals.append(v / 2**30)
        labels.append(NAMES[k] + ("\n(process peak)" if k == "jax" else ""))
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
    bars(ax, labels, vals, lambda v: f"{v:.1f}", best="min")
    style(ax, "GPU memory for one GPT-2 124M training step", "batch 8 × 1024, fp32 — peak reserved, GiB (lower is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart-memory.png"), facecolor="white")


def chart_maxbatch():
    rows = {r["framework"]: r for r in load("fp32", "e2")}
    e3 = load("fp32", "e3")
    labels, vals = [], []
    if e3:
        labels.append("Soliton\n(auto-fit)")
        vals.append(e3[0]["max_batch"])
    for k in ["soliton+arena", "soliton", "jax", "pytorch", "tensorflow"]:
        if k in rows:
            labels.append(NAMES[k])
            vals.append(rows[k]["true_max"])
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
    bars(ax, labels, vals, lambda v: f"{v:.0f}", best="max")
    style(ax, "Largest batch that trains under a 24 GiB budget",
          "GPT-2 124M, sequence 1024 — verified by training, not estimated (higher is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart-maxbatch.png"), facecolor="white")


def chart_throughput():
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
    order = ["jax", "soliton", "pytorch", "tensorflow"]
    fp32 = {r["framework"]: r for r in load("fp32", "e1") if r["batch"] == 8}
    tf32 = {r["framework"]: r for r in load("tf32", "e1") if r["batch"] == 8}
    x = range(len(order))
    a = [(fp32[k].get("tok_s") or 0) / 1000 for k in order]
    b = [(tf32[k].get("tok_s") or 0) / 1000 for k in order]
    ax.bar([i - 0.19 for i in x], a, width=0.36, color=LOSE, label="fp32")
    ax.bar([i + 0.19 for i in x], b, width=0.36, color=WIN, label="TF32")
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(i - 0.19, va, f"{va:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(i + 0.19, vb, f"{vb:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([NAMES[k] for k in order])
    ax.set_ylim(0, max(a + b) * 1.2)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=MUTED)
    style(ax, "Training throughput, both precisions", "GPT-2 124M, batch 8 — thousands of tokens/second (higher is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart-throughput.png"), facecolor="white")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    chart_memory()
    chart_maxbatch()
    chart_throughput()
    docs = os.path.join(os.path.dirname(HERE), "docs", "source", "_static")
    if os.path.isdir(docs):  # the docs site serves its own copy
        for f in os.listdir(OUT):
            if f.startswith("chart-"):
                shutil.copy(os.path.join(OUT, f), docs)
    print("wrote", ", ".join(sorted(f for f in os.listdir(OUT) if f.startswith("chart-"))))
