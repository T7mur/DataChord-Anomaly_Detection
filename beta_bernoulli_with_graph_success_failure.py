# -*- coding: utf-8 -*-
"""
beta_bernoulli_with_graph_success_failure (Exercise).py
(RECREATED from Final_Report_TT.docx, sections 8.1, 9.2.1)

Beta-Bernoulli firewall risk model on a (source -> destination) graph.

For every minute t the script looks at a rolling window of TIME_WINDOW seconds ending at t:
  * DENY outcomes are "failures", ALLOW outcomes are "successes" for each edge (src, dst)
  * posterior mean   p = alpha / (alpha + beta)
        alpha = ALPHA0 + FAIL_WEIGHT * fails + src_consecutive_bonus
        beta  = BETA0  + SUCC_WEIGHT * successes
    (without the streak bonus, 2 fails + 1 allow with prior 1/1 gives 5/7 = 0.714286, the peak value
    quoted in the report; the exact bonus/clamping details are a best-effort reconstruction)
  * src_consecutive_bonus = SRC_CONSEC_WEIGHT * min(longest DENY streak of the source in the
    window, SRC_CONSEC_MAX_BONUS); applied to the edge only (SRC_BONUS_SCOPE = "edge_only")
  * every simple path (<= PATH_CUTOFF nodes) in the window graph is scored by the
    PATH_AGGREGATOR ("geometric" mean of edge posteriors); the best path of the minute is kept
  * minutes with best score >= THRESHOLD are merged into suspicious windows

Outputs: timeline.csv, suspicious_windows.csv, source_rankings.csv, best_path_anim.gif
"""
import itertools
import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- parameters
CSV_PATH = "firewall_logs.csv"          # columns: timestamp, source_ip, destination_ip, action
OUTDIR = "outputs"

#Rolling window length in seconds
TIME_WINDOW: Optional[int] = 240
#Suspicion threshold (0-1).
THRESHOLD: Optional[float] = .7

# Beta prior
ALPHA0: Optional[float] = 1.0
BETA0: Optional[float] = 1.0

#Evidence weights
FAIL_WEIGHT: Optional[float] = 2.0
# Success weight
SUCC_WEIGHT: float = 1.0

#Consecutive-source bonus
SRC_CONSEC_WEIGHT: Optional[float] = 0.35

SRC_CONSEC_MAX_BONUS: Optional[float] = 8 #

#Scope of the src bonus.
SRC_BONUS_SCOPE: Optional[str] = "edge_only"

#Path score aggregator across edges.
PATH_AGGREGATOR: Optional[str] = "geometric"

# This is about reporting/aggregation for the timeline output file (like timeline.csv).
TIMELINE_WINDOW_MINUTES: Optional[int] = 5

# Keep animation readable on busy graphs
MAX_EDGES = 60     # top-K edges by activity; set 0/None to disable trimming
PATH_CUTOFF = 7    # max simple-path length (in nodes)

# Posteriors clamped for stability/legibility
P_MIN, P_MAX = 0.01, 0.99

# animation
ANIMATE = True
ANIM_FILE = "best_path_anim.gif"
ANIM_FRAMES = 120
ANIM_FPS = 4

FAIL_ACTIONS = {"deny", "denied", "drop", "block", "blocked", "reject", "fail", "failed"}


# ----------------------------------------------------------------------------- loading
def find_col(df, cands):
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in low:
            return low[c]
    raise KeyError(f"none of {cands} in {list(df.columns)}")


def load(path):
    df = pd.read_csv(path)
    ts = find_col(df, ["timestamp", "time", "datetime"])
    src = find_col(df, ["source_ip", "src_ip", "src"])
    dst = find_col(df, ["destination_ip", "dst_ip", "dst"])
    act = find_col(df, ["action", "decision", "result", "status"])
    d = pd.DataFrame({"ts": pd.to_datetime(df[ts], errors="coerce"), "src": df[src],
                      "dst": df[dst], "act": df[act].astype(str).str.lower().str.strip()})
    d = d.dropna(subset=["ts", "src", "dst"]).sort_values("ts").reset_index(drop=True)
    d["fail"] = d["act"].isin(FAIL_ACTIONS).astype(int)      # null/unknown evidence counts as 0 (report 6.6 A)
    d["succ"] = 1 - d["fail"]
    t0 = d["ts"].min().floor("min")
    d["sec"] = (d["ts"] - t0).dt.total_seconds()
    return d, t0


# ----------------------------------------------------------------------------- core model
def longest_fail_streak(fails):
    best = cur = 0
    for f in fails:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def window_edges(win):
    """Beta-Bernoulli posterior for every (src,dst) edge in a window of events."""
    streak = {s: longest_fail_streak(g["fail"].values) for s, g in win.groupby("src", sort=False)}
    rows = []
    for (s, t), g in win.groupby(["src", "dst"], sort=False):
        F, S = int(g["fail"].sum()), int(g["succ"].sum())
        bonus = SRC_CONSEC_WEIGHT * min(streak[s], SRC_CONSEC_MAX_BONUS)
        if SRC_BONUS_SCOPE == "edge_only" and F == 0:
            bonus = 0.0                                   # bonus only applies to edges that actually failed
        w_fail = FAIL_WEIGHT * F + bonus
        alpha = ALPHA0 + w_fail
        beta = BETA0 + SUCC_WEIGHT * S
        post = min(max(alpha / (alpha + beta), P_MIN), P_MAX)
        lik = (w_fail + 1) / (w_fail + SUCC_WEIGHT * S + 2)       # evidence-only mean (0.5 with no evidence)
        rows.append({"src": s, "dst": t, "F": F, "S": S, "W": bonus, "prior": ALPHA0 / (ALPHA0 + BETA0),
                     "lik": lik, "post": post, "n": F + S})
    e = pd.DataFrame(rows)
    if MAX_EDGES and len(e) > MAX_EDGES:
        e = e.sort_values("n", ascending=False).head(MAX_EDGES)
    return e


def aggregate(vals):
    vals = np.asarray(vals, float)
    if PATH_AGGREGATOR == "geometric":
        return float(np.exp(np.log(vals).mean()))
    if PATH_AGGREGATOR == "min":
        return float(vals.min())
    return float(vals.mean())


def best_path(edges):
    """Best simple path (<= PATH_CUTOFF nodes) in the window graph under the aggregator."""
    if edges.empty:
        return [], 0.5
    G = nx.DiGraph()
    for r in edges.itertuples():
        G.add_edge(r.src, r.dst, post=r.post)
    best, best_score = [], -1.0
    for a in G.nodes:
        if G.out_degree(a) == 0:
            continue
        for b in G.nodes:
            if a == b:
                continue
            for p in itertools.islice(nx.all_simple_paths(G, a, b, cutoff=PATH_CUTOFF - 1), 2000):
                sc = aggregate([G[u][v]["post"] for u, v in zip(p[:-1], p[1:])])
                if sc > best_score + 1e-12 or (abs(sc - best_score) <= 1e-12 and len(p) < len(best)):
                    best, best_score = p, sc
    return (best, best_score) if best else ([], 0.5)


def run_timeline(d, t0):
    n_min = int(d["sec"].max() // 60) + 1
    secs = d["sec"].values
    rows, per_src = [], {}
    edge_cache = {}
    for m in range(n_min):
        end = (m + 1) * 60
        lo = np.searchsorted(secs, end - TIME_WINDOW, side="left")
        hi = np.searchsorted(secs, end, side="left")
        win = d.iloc[lo:hi]
        if win.empty:
            rows.append({"t_min": m, "time": t0 + pd.Timedelta(minutes=m), "best_path": "", "score": 0.5})
            continue
        e = window_edges(win)
        path, score = best_path(e)
        rows.append({"t_min": m, "time": t0 + pd.Timedelta(minutes=m),
                     "best_path": " -> ".join(path), "score": score})
        for s, g in e.groupby("src"):
            per_src.setdefault(s, []).append(g["post"].max())
        edge_cache[m] = (e, path, score)
    return pd.DataFrame(rows), per_src, edge_cache, n_min


def suspicious_windows(tl):
    hot = tl[tl["score"] >= THRESHOLD]
    wins, cur = [], None
    for r in hot.itertuples():
        if cur and r.t_min - cur["end"] <= 1:
            cur["end"] = r.t_min
            if r.score > cur["peak"]:
                cur.update(peak=r.score, peak_t=r.t_min, peak_time=r.time, path=r.best_path)
        else:
            if cur: wins.append(cur)
            cur = dict(start=r.t_min, end=r.t_min, peak=r.score, peak_t=r.t_min,
                       peak_time=r.time, path=r.best_path)
    if cur: wins.append(cur)
    w = pd.DataFrame(wins)
    if not w.empty:
        w = w.rename(columns={"start": "start_min", "end": "end_min", "peak": "peak_score"})
    return w


def source_rankings(per_src, n_min):
    rows = []
    for s, vals in per_src.items():
        v = np.array(vals)
        rows.append({"src": s, "minutes_active": len(v), "mean_posterior": v.mean(),
                     "max_posterior": v.max(),
                     "share_minutes_ge_thresh": float((v >= THRESHOLD).sum() / len(v))})
    return pd.DataFrame(rows).sort_values(["share_minutes_ge_thresh", "mean_posterior"], ascending=False)


# ----------------------------------------------------------------------------- animation
def animate(edge_cache, n_min, path):
    from PIL import Image
    import io
    minutes = sorted(edge_cache)
    if not minutes:
        return
    step = max(1, len(minutes) // ANIM_FRAMES)
    frames = []
    for m in minutes[::step][:ANIM_FRAMES]:
        e, p, sc = edge_cache[m]
        G = nx.DiGraph()
        for r in e.itertuples():
            G.add_edge(r.src, r.dst, lab=f"W={r.W:.1f} F={r.F}\nPr={r.prior:.2f} L={r.lik:.2f} Po={r.post:.2f}")
        pos = nx.spring_layout(G, seed=42)
        pe = set(zip(p[:-1], p[1:]))
        fig, ax = plt.subplots(figsize=(11, 7))
        nx.draw_networkx_edges(G, pos, ax=ax, width=[4 if ed in pe else 1 for ed in G.edges],
                               edge_color=["#0d1b2a" if ed in pe else "#9aa5b1" for ed in G.edges], arrows=True)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#d0e7ff", edgecolors="#1b263b", node_size=500)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "lab"), font_size=6, ax=ax)
        ax.set_title(f"t={m * 60}s — Best path score={sc:.3f} (thr={THRESHOLD:.2f}, agg={PATH_AGGREGATOR})")
        ax.axis("off")
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90); plt.close(fig)
        frames.append(Image.open(io.BytesIO(buf.getvalue())).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / ANIM_FPS), loop=0)


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d, t0 = load(CSV_PATH)
    tl, per_src, cache, n_min = run_timeline(d, t0)
    tl.to_csv(os.path.join(OUTDIR, "timeline.csv"), index=False)
    win = suspicious_windows(tl)
    win.to_csv(os.path.join(OUTDIR, "suspicious_windows.csv"), index=False)
    source_rankings(per_src, n_min).to_csv(os.path.join(OUTDIR, "source_rankings.csv"), index=False)
    print(f"{n_min} minutes, {len(win)} suspicious windows")
    if not win.empty:
        print(win.to_string(index=False))
    if ANIMATE:
        animate(cache, n_min, os.path.join(OUTDIR, ANIM_FILE))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        CSV_PATH = sys.argv[1]
    main()
