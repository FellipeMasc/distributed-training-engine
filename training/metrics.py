"""Training benchmark metrics: throughput, peak memory, scaling, convergence.

Every rank records its own step times, loss (``None`` on ranks that do not
own the loss, e.g. non-last pipeline stages) and peak memory. At the end the
per-rank records are gathered to global rank 0, which merges them into one
summary and optionally compares it with a baseline run to compute scaling
efficiency.

Definitions
-----------
tokens/sec
    Global tokens consumed per optimizer step divided by the step wall time,
    averaged over the steps after ``warmup_steps``. Global tokens per step is
    ``micro_batch_size * seq_length * dp_size`` since every data-parallel
    replica sees a different batch while tp/pp ranks share one.
peak memory
    ``torch.cuda.max_memory_allocated`` / ``max_memory_reserved`` per rank on
    CUDA; the process RSS high-water mark otherwise. The summary keeps the
    per-rank values and the max across ranks.
scaling efficiency
    ``(tokens_per_sec / baseline_tokens_per_sec) / (world_size / baseline_world_size)``.
    1.0 is perfect linear scaling relative to the baseline run. Strong scaling
    means the baseline used the same global batch; weak scaling means the same
    per-device batch. The summary reports which one applies to the comparison.
loss convergence
    Per-step loss curve (averaged across the ranks that own a loss), first and
    last loss, best loss, relative improvement, and the least-squares slope of
    ``log(loss)`` over steps (negative means the loss is decaying; the closer to
    zero, the flatter the curve).
"""

from __future__ import annotations

import json
import math
import os
import platform
import resource
import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_bytes(device: torch.device) -> dict[str, int]:
    """Peak memory of this process so far, in bytes."""
    if device.type == "cuda":
        return {
            "allocated": int(torch.cuda.max_memory_allocated(device)),
            "reserved": int(torch.cuda.max_memory_reserved(device)),
        }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    if platform.system() != "Darwin":
        rss *= 1024
    return {"rss": int(rss)}


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _log_loss_slope(losses: list[float]) -> float | None:
    """Least-squares slope of log(loss) vs step index (per-step decay rate)."""
    pts = [(i, math.log(l)) for i, l in enumerate(losses) if l is not None and l > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    var = sum((x - mx) ** 2 for x, _ in pts)
    if var == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / var


def loss_convergence(losses: list[float | None]) -> dict:
    valid = [l for l in losses if l is not None]
    if not valid:
        return {"steps_with_loss": 0}
    k = max(1, min(5, len(valid) // 4))  # window for first/last averages
    first, last = valid[0], valid[-1]
    return {
        "steps_with_loss": len(valid),
        "first_loss": first,
        "last_loss": last,
        "min_loss": min(valid),
        "min_loss_step": losses.index(min(valid)) + 1,
        "first_window_mean": _mean(valid[:k]),
        "last_window_mean": _mean(valid[-k:]),
        "window": k,
        "relative_improvement": (first - last) / first if first else None,
        "log_loss_slope_per_step": _log_loss_slope(valid),
        "curve": valid,
    }


def scaling_efficiency(summary: dict, baseline: dict) -> dict:
    """Compare this run's throughput against a baseline run's JSON summary."""
    cfg, bcfg = summary["config"], baseline["config"]
    ws, bws = cfg["world_size"], bcfg["world_size"]
    tps, btps = summary["throughput"]["tokens_per_sec"], baseline["throughput"]["tokens_per_sec"]
    if not tps or not btps:
        return {"error": "missing throughput in run or baseline"}
    speedup = tps / btps
    ideal = ws / bws
    same_global = cfg["tokens_per_step"] == bcfg["tokens_per_step"]
    same_per_device = cfg["tokens_per_step"] / ws == bcfg["tokens_per_step"] / bws
    if same_global and same_per_device:
        mode = "identical_batch"
    elif same_global:
        mode = "strong"
    elif same_per_device:
        mode = "weak"
    else:
        mode = "mixed (global and per-device batch both differ; interpret with care)"
    return {
        "baseline_path": baseline.get("_path"),
        "baseline_world_size": bws,
        "baseline_parallelism": bcfg["parallelism"],
        "baseline_tokens_per_sec": btps,
        "speedup": speedup,
        "ideal_speedup": ideal,
        "efficiency": speedup / ideal,
        "per_device_throughput_ratio": (tps / ws) / (btps / bws),
        "scaling_mode": mode,
    }


@dataclass
class TrainingMetrics:
    """Per-rank step recorder; see module docstring for the metric definitions."""

    tokens_per_step: int  # global tokens per optimizer step
    device: torch.device
    config: dict
    warmup_steps: int = 1
    is_log_rank: bool = False
    step_times: list[float] = field(default_factory=list)
    losses: list[float | None] = field(default_factory=list)
    _t0: float | None = None
    _train_start: float | None = None

    def start_step(self) -> None:
        _sync(self.device)
        now = time.perf_counter()
        if self._train_start is None:
            self._train_start = now
        self._t0 = now

    def end_step(self, loss: float | None) -> float:
        _sync(self.device)
        dt = time.perf_counter() - self._t0
        self.step_times.append(dt)
        self.losses.append(loss)
        return dt

    # ── Live logging ─────────────────────────────────────────────
    def step_line(self, step: int, max_steps: int) -> str:
        dt = self.step_times[-1]
        loss = self.losses[-1]
        loss_s = f"{loss:.4f}" if loss is not None else "n/a"
        mem = peak_memory_bytes(self.device)
        mem_key = "allocated" if "allocated" in mem else "rss"
        return (
            f"step {step}/{max_steps} | loss {loss_s} | "
            f"{dt * 1000:.1f} ms/step | {self.tokens_per_step / dt:,.0f} tok/s | "
            f"peak {mem_key} {mem[mem_key] / 2**30:.2f} GiB"
        )

    # ── Per-rank summary ─────────────────────────────────────────
    def local_summary(self) -> dict:
        timed = self.step_times[self.warmup_steps :]
        return {
            "rank": dist.get_rank() if dist.is_initialized() else 0,
            "steps": len(self.step_times),
            "step_times": self.step_times,
            "timed_wall_time": sum(timed),
            "losses": self.losses,
            "peak_memory_bytes": peak_memory_bytes(self.device),
        }

    # ── Cross-rank merge (rank 0 only returns the merged dict) ───
    def gather_summary(self) -> dict | None:
        local = self.local_summary()
        if dist.is_initialized() and dist.get_world_size() > 1:
            gathered: list[dict | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            if dist.get_rank() != 0:
                return None
        else:
            gathered = [local]
        return self._merge(gathered)

    def _merge(self, per_rank: list[dict]) -> dict:
        steps = per_rank[0]["steps"]
        timed_steps = max(0, steps - self.warmup_steps)
        # Ranks are synchronized by collectives every step; the slowest rank's
        # wall time is what the run actually costs.
        wall = max(r["timed_wall_time"] for r in per_rank)
        world_size = len(per_rank)
        tps = self.tokens_per_step * timed_steps / wall if wall > 0 and timed_steps else None

        # Loss curve: average over the ranks that own a loss at each step.
        curve: list[float | None] = []
        for i in range(steps):
            vals = [r["losses"][i] for r in per_rank if r["losses"][i] is not None]
            curve.append(_mean(vals))

        mem_keys = per_rank[0]["peak_memory_bytes"].keys()
        peak = {
            k: {
                "max_over_ranks": max(r["peak_memory_bytes"][k] for r in per_rank),
                "per_rank": [r["peak_memory_bytes"][k] for r in per_rank],
            }
            for k in mem_keys
        }

        return {
            "config": {**self.config, "world_size": world_size, "tokens_per_step": self.tokens_per_step},
            "throughput": {
                "warmup_steps": self.warmup_steps,
                "timed_steps": timed_steps,
                "timed_wall_time_sec": wall,
                "mean_step_time_sec": wall / timed_steps if timed_steps else None,
                "tokens_per_sec": tps,
                "tokens_per_sec_per_device": tps / world_size if tps else None,
                "per_rank_mean_step_time_sec": [
                    _mean(r["step_times"][self.warmup_steps :]) for r in per_rank
                ],
                "per_rank_step_times_sec": [r["step_times"] for r in per_rank],
            },
            "peak_memory_bytes": peak,
            "loss_convergence": loss_convergence(curve),
        }


# ── Reporting ────────────────────────────────────────────────────
def load_baseline(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    data["_path"] = path
    return data


def write_summary(summary: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def format_summary(summary: dict) -> str:
    cfg, thr, conv = summary["config"], summary["throughput"], summary["loss_convergence"]
    p = cfg["parallelism"]
    lines = [
        "=" * 64,
        f"Benchmark summary  (world_size={cfg['world_size']}, dp={p['dp']} tp={p['tp']} "
        f"pp={p['pp']}, dtype={cfg['dtype']}, layers={cfg['num_hidden_layers']})",
        "-" * 64,
        f"tokens/step (global)     : {cfg['tokens_per_step']:,}",
        f"timed steps              : {thr['timed_steps']} (after {thr['warmup_steps']} warmup)",
    ]
    if thr["tokens_per_sec"]:
        lines += [
            f"mean step time           : {thr['mean_step_time_sec'] * 1000:.1f} ms",
            f"throughput               : {thr['tokens_per_sec']:,.0f} tok/s "
            f"({thr['tokens_per_sec_per_device']:,.0f} tok/s/device)",
        ]
    for k, v in summary["peak_memory_bytes"].items():
        per = ", ".join(f"{x / 2**30:.2f}" for x in v["per_rank"])
        lines.append(f"peak memory {k:<12} : {v['max_over_ranks'] / 2**30:.2f} GiB max  [per rank: {per}]")
    if conv.get("steps_with_loss"):
        slope = conv["log_loss_slope_per_step"]
        lines += [
            f"loss first -> last       : {conv['first_loss']:.4f} -> {conv['last_loss']:.4f} "
            f"({conv['relative_improvement'] * 100:+.1f}% improvement)",
            f"loss min                 : {conv['min_loss']:.4f} at step {conv['min_loss_step']}",
            f"log-loss slope / step    : {slope:+.4f}" if slope is not None else "log-loss slope / step    : n/a",
        ]
    else:
        lines.append("loss                     : n/a (no rank reported a loss)")
    if "scaling" in summary:
        s = summary["scaling"]
        if "error" in s:
            lines.append(f"scaling                  : {s['error']}")
        else:
            lines += [
                f"scaling vs baseline      : {s['speedup']:.2f}x speedup over "
                f"{s['baseline_world_size']} device(s) (ideal {s['ideal_speedup']:.2f}x) "
                f"-> efficiency {s['efficiency'] * 100:.1f}% [{s['scaling_mode']}]",
            ]
    lines.append("=" * 64)
    return "\n".join(lines)
