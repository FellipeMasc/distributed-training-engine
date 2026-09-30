"""Plot benchmark summaries written by training/train.py (training/metrics.py).

Per run (``plot_run``):
    loss_curve.png     loss per optimizer step
    step_time.png      per-rank step time and per-step throughput (two panels)
    peak_memory.png    peak memory per rank

Across runs (``plot_comparison``, needs >= 2 summaries):
    scaling.png        throughput vs device count with the ideal-linear line,
                       and scaling efficiency relative to the smallest run
    loss_comparison.png  loss curves of every run overlaid

Usage:
    python tools/plot_metrics.py benchmarks/*.json            # -> benchmarks/plots/
    python tools/plot_metrics.py run.json --out some/dir

train.py calls ``plot_run`` automatically after writing the JSON unless
``--no-plots`` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (fixed order, never cycled): blue, orange,
# aqua, yellow, magenta, green, violet, red.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
WARMUP_FILL = "#f0efec"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "font.size": 10,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def run_label(summary: dict) -> str:
    p = summary["config"]["parallelism"]
    parts = [f"{k}{v}" for k, v in p.items() if v > 1]
    layers = summary["config"].get("num_hidden_layers")
    label = f"{summary['config']['world_size']} dev " + (" ".join(parts) if parts else "single")
    return f"{label} L{layers}" if layers is not None else label


def _fmt_bytes(b: float) -> str:
    return f"{b / 2**30:.2f} GiB" if b >= 2**30 else f"{b / 2**20:.0f} MiB"


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Per-run charts ────────────────────────────────────────────────
def plot_loss_curve(summary: dict, out_dir: str) -> str | None:
    curve = summary["loss_convergence"].get("curve")
    if not curve:
        return None
    steps = list(range(1, len(curve) + 1))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(steps, curve, color=SERIES[0], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.annotate(
        f"{curve[-1]:.3f}", (steps[-1], curve[-1]), xytext=(6, 0),
        textcoords="offset points", va="center", color=INK, fontsize=9,
    )
    ax.set_title(f"Training loss  ·  {run_label(summary)}")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_xlim(0.5, steps[-1] + 0.5 + len(steps) * 0.08)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    return _save(fig, out_dir, "loss_curve.png")


def plot_step_time(summary: dict, out_dir: str) -> str | None:
    thr = summary["throughput"]
    per_rank = thr.get("per_rank_step_times_sec")
    if not per_rank:
        return None
    warmup = thr["warmup_steps"]
    tokens = summary["config"]["tokens_per_step"]
    n = len(per_rank[0])
    steps = list(range(1, n + 1))
    # Slowest rank per step is what the step actually costs.
    slowest = [max(r[i] for r in per_rank) for i in range(n)]
    tok_s = [tokens / t for t in slowest]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"hspace": 0.35})
    for ax in (ax1, ax2):
        if warmup:
            ax.axvspan(0.5, warmup + 0.5, color=WARMUP_FILL, lw=0, zorder=0)
    if warmup:
        ax1.text(0.6, 0.03, "warmup (excluded)", transform=ax1.get_xaxis_transform(),
                 fontsize=8, color=INK_2, va="bottom")

    show_ranks = len(per_rank) <= 8
    for i, r in enumerate(per_rank if show_ranks else []):
        ax1.plot(steps, [t * 1000 for t in r], color=SERIES[i], marker="o",
                 markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"rank {i}")
    if not show_ranks:
        ax1.plot(steps, [t * 1000 for t in slowest], color=SERIES[0], marker="o",
                 markeredgecolor=SURFACE, markeredgewidth=1.5, label="slowest rank")
    if len(per_rank) > 1:
        ax1.legend(loc="upper right", ncol=min(4, len(per_rank)))
    ax1.set_title(f"Step time  ·  {run_label(summary)}")
    ax1.set_ylabel("ms / step")
    ax1.set_ylim(bottom=0)

    ax2.plot(steps, tok_s, color=SERIES[0], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5)
    if thr.get("tokens_per_sec"):
        ax2.axhline(thr["tokens_per_sec"], color=INK_2, lw=1, ls=(0, (4, 3)))
        ax2.text(0.6, thr["tokens_per_sec"], f" mean {thr['tokens_per_sec']:,.0f} tok/s",
                 fontsize=8, color=INK_2, va="bottom")
    ax2.set_title("Throughput (global tokens / slowest-rank step time)")
    ax2.set_ylabel("tokens / s")
    ax2.set_xlabel("optimizer step")
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(0.5, n + 0.5 + n * 0.1)
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    return _save(fig, out_dir, "step_time.png")


def plot_peak_memory(summary: dict, out_dir: str) -> str | None:
    mem = summary["peak_memory_bytes"]
    if not mem:
        return None
    keys = list(mem)
    ranks = len(mem[keys[0]]["per_rank"])
    fig, ax = plt.subplots(figsize=(7, 1.4 + 0.4 * ranks * len(keys)))
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    height = 0.6 / len(keys)
    ax.set_ylim(ranks - 0.5, -0.5)
    for k_i, k in enumerate(keys):
        vals = [v / 2**30 for v in mem[k]["per_rank"]]
        ys = [r + (k_i - (len(keys) - 1) / 2) * height for r in range(ranks)]
        bars = ax.barh(ys, vals, height=height * 0.9, color=SERIES[k_i], label=f"peak {k}")
        for b, v in zip(bars, mem[k]["per_rank"]):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {_fmt_bytes(v)}",
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(range(ranks), [f"rank {r}" for r in range(ranks)])
    ax.set_xlabel("GiB")
    ax.set_xlim(0, max(max(v["per_rank"]) for v in mem.values()) / 2**30 * 1.25)
    ax.set_title(f"Peak memory per rank  ·  {run_label(summary)}")
    if len(keys) > 1:
        ax.legend(loc="lower right")
    return _save(fig, out_dir, "peak_memory.png")


def plot_run(summary: dict, out_dir: str) -> list[str]:
    return [
        p for p in (
            plot_loss_curve(summary, out_dir),
            plot_step_time(summary, out_dir),
            plot_peak_memory(summary, out_dir),
        ) if p
    ]


# ── Cross-run charts ──────────────────────────────────────────────
def plot_scaling(summaries: list[dict], out_dir: str) -> str | None:
    runs = [s for s in summaries if s["throughput"].get("tokens_per_sec")]
    if len(runs) < 2:
        return None
    runs.sort(key=lambda s: s["config"]["world_size"])
    base = runs[0]
    bws, btps = base["config"]["world_size"], base["throughput"]["tokens_per_sec"]
    ws = [s["config"]["world_size"] for s in runs]
    tps = [s["throughput"]["tokens_per_sec"] for s in runs]
    eff = [(t / btps) / (w / bws) * 100 for w, t in zip(ws, tps)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"wspace": 0.3})
    ideal = [btps * w / bws for w in ws]
    ax1.plot(ws, ideal, color=INK_2, lw=1, ls=(0, (4, 3)), label="ideal linear")
    ax1.plot(ws, tps, color=SERIES[0], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5,
             label="measured")
    for s, w, t in zip(runs, ws, tps):
        ax1.annotate(run_label(s).split(" dev ")[1], (w, t), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax1.set_title("Throughput vs devices")
    ax1.set_xlabel("devices")
    ax1.set_ylabel("tokens / s")
    ax1.set_ylim(bottom=0)
    ax1.set_xticks(ws)
    ax1.legend(loc="upper left")

    bars = ax2.bar(range(len(ws)), eff, color=SERIES[0], width=0.6)
    ax2.axhline(100, color=INK_2, lw=1, ls=(0, (4, 3)))
    for b, e in zip(bars, eff):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{e:.0f}%", ha="center",
                 va="bottom", fontsize=9, color=INK)
    ax2.set_xticks(range(len(ws)), [run_label(s).replace(" dev ", " dev\n") for s in runs], fontsize=8)
    ax2.set_ylabel("scaling efficiency (%)")
    ax2.set_title(f"Efficiency vs {bws}-device baseline")
    ax2.set_ylim(0, max(110, max(eff) * 1.15))
    return _save(fig, out_dir, "scaling.png")


def plot_loss_comparison(summaries: list[dict], out_dir: str) -> str | None:
    runs = [s for s in summaries if s["loss_convergence"].get("curve")]
    if len(runs) < 2:
        return None
    runs = runs[: len(SERIES)]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, s in enumerate(runs):
        c = s["loss_convergence"]["curve"]
        x = list(range(1, len(c) + 1))
        ax.plot(x, c, color=SERIES[i], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=run_label(s))
        if len(runs) <= 4:
            ax.annotate(run_label(s), (x[-1], c[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
    ax.set_title("Loss convergence across runs")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    return _save(fig, out_dir, "loss_comparison.png")


def plot_comparison(summaries: list[dict], out_dir: str) -> list[str]:
    return [p for p in (plot_scaling(summaries, out_dir), plot_loss_comparison(summaries, out_dir)) if p]


# ── CLI ───────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summaries", nargs="+", help="Benchmark JSON files written by train.py")
    ap.add_argument("--out", default=None, help="Output dir (default: <dir of first json>/plots)")
    args = ap.parse_args(argv)

    summaries = []
    for path in args.summaries:
        with open(path) as f:
            s = json.load(f)
        s["_path"] = path
        summaries.append(s)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.summaries[0])), "plots")

    written = []
    for s in summaries:
        stem = os.path.splitext(os.path.basename(s["_path"]))[0]
        written += plot_run(s, os.path.join(out, stem))
    written += plot_comparison(summaries, out)
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
