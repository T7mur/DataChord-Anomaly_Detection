# -*- coding: utf-8 -*-
"""
ids_traversal_with_svd.py   (RECREATED from Final_Report_TT.docx, sections 7, 8.2, 9.2)

Severity-weighted IDS graph with SVD imputation of missing severities,
degree-tail ("chatty node") detection, centralities, BFS / DFS / Dijkstra
traversal and a BFS animation.

Inputs : ids_logs_ttt.csv  (timestamp, source_ip, destination_ip, severity)
Outputs: edge_probs.csv, node_stats.csv, chatty_nodes.csv, bfs_paths.csv,
         dfs_paths.csv, best_path_dijkstra.csv, algo_compare.csv,
         centralities.csv, quick_ids_graph.png, degree_in.png, degree_out.png,
         chatty_nodes_subgraph.png, bfs_anim.gif
"""
import math
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

EPS = 1e-12

CFG = {
    # --- Data mapping ---
    "CSV_PATH": "ids_logs_ttt.csv",
    "COL_TS":  ["timestamp", "time", "datetime"],
    "COL_SRC": ["source_ip", "src_ip", "src"],
    "COL_DST": ["destination_ip", "dst_ip", "dst"],
    "COL_SEV": ["severity", "sev", "alert_severity"],   # optional

    # --- Graph building & SVD imputation ---
    "USE_SVD_IMPUTE": True,
    "SVD_RANK": 6,
    "SVD_MAX_ITERS": 40,
    "SVD_TOL": 1e-6,
    "SEV_TO_WEIGHT_MIN": 1.0,
    "SEV_TO_WEIGHT_MAX": 1.6,

    # --- Degree distribution fitting & chatty-node detection ---
    "DEGREE_MODELS": ["lognormal"],

    # Tail tests for chatty nodes, ranked by min_tail_p
    "ALPHA_CHAT": 0.01,
    "TOP_N_CHAT": 20,

    # --- Centralities ---
    "CALC_CENTRALITIES": True,
    "CENTRALITIES": ["degree", "betweenness", "closeness", "pagerank"],
    "PAGERANK_DAMPING": 0.85,

    # --- Traversal algorithms & animation ---
    "START_NODE": None,        # None = choose max out-degree node
    "TARGET_NODE": None,       # None = choose max in-degree node != start
    "MAX_DEPTH": 4,            # BFS/DFS edges
    "ANIMATE": True,           # BFS animation
    "ANIM_FPS": 2,             # frames per second
    "ANIM_FILE": "bfs_anim.gif",

    # --- Plotting & random seeds (layouts) ---
    "MAKE_PLOTS": True,
    "SEED_LAYOUT": 42,
    "PLOT_GRAPH_FILE": "quick_ids_graph.png",
    "OUTDIR": ".",
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def pick_col(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")
    return None


def out(name):
    os.makedirs(CFG["OUTDIR"], exist_ok=True)
    return os.path.join(CFG["OUTDIR"], name)


def load_data(path):
    df = pd.read_csv(path)
    c_src = pick_col(df, CFG["COL_SRC"])
    c_dst = pick_col(df, CFG["COL_DST"])
    c_sev = pick_col(df, CFG["COL_SEV"], required=False)
    keep = {c_src: "src", c_dst: "dst"}
    if c_sev:
        keep[c_sev] = "sev"
    df = df.rename(columns=keep)[list(keep.values())]
    df = df.dropna(subset=["src", "dst"])           # null IPs cannot form an edge
    if "sev" in df:
        df["sev"] = pd.to_numeric(df["sev"], errors="coerce")
        df.loc[df["sev"] < 0, "sev"] = np.nan        # -1 = unknown severity
    else:
        df["sev"] = np.nan
    return df


# ----------------------------------------------------------------------------
# SVD imputation of the (src x dst) mean-severity matrix
# ----------------------------------------------------------------------------
def svd_impute_matrix(M, mask_obs, rank, max_iters, tol):
    """Iterative low-rank (hard-impute) completion.  mask_obs=True where M is observed."""
    M = np.array(M, dtype=float)
    if mask_obs.sum() == 0:
        return np.full_like(M, 1.0)
    fill = np.nanmean(np.where(mask_obs, M, np.nan))
    X = np.where(mask_obs, M, fill)
    for _ in range(max_iters):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        r = min(rank, len(s))
        L = (U[:, :r] * s[:r]) @ Vt[:r, :]
        X_new = np.where(mask_obs, M, L)
        delta = np.linalg.norm(X_new - X) / (np.linalg.norm(X) + EPS)
        X = X_new
        if delta < tol:
            break
    return X


def sev_to_weight(sev, smin, smax):
    wmin, wmax = CFG["SEV_TO_WEIGHT_MIN"], CFG["SEV_TO_WEIGHT_MAX"]
    if smax - smin < EPS:
        return wmin
    z = np.clip((sev - smin) / (smax - smin), 0, 1)
    return wmin + (wmax - wmin) * z


# ----------------------------------------------------------------------------
# graph construction
# ----------------------------------------------------------------------------
def build_graph(df):
    edges = (df.groupby(["src", "dst"])
               .agg(count=("src", "size"), sev_mean=("sev", "mean"))
               .reset_index())
    out_total = edges.groupby("src")["count"].transform("sum")
    edges["p"] = edges["count"] / out_total

    nodes = sorted(set(edges["src"]) | set(edges["dst"]))
    idx = {n: i for i, n in enumerate(nodes)}
    M = np.full((len(nodes), len(nodes)), np.nan)
    for r in edges.itertuples():
        M[idx[r.src], idx[r.dst]] = r.sev_mean
    mask_obs = ~np.isnan(M)

    if CFG["USE_SVD_IMPUTE"] and mask_obs.any():
        M_imp = svd_impute_matrix(np.nan_to_num(M), mask_obs, CFG["SVD_RANK"],
                                  CFG["SVD_MAX_ITERS"], CFG["SVD_TOL"])
    else:
        gm = np.nanmean(M) if mask_obs.any() else 1.0
        M_imp = np.where(mask_obs, M, gm)

    smin, smax = np.nanmin(M_imp), np.nanmax(M_imp)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    rows = []
    for r in edges.itertuples():
        w = sev_to_weight(M_imp[idx[r.src], idx[r.dst]], smin, smax)
        p = r.p
        p_adj = max(p * w, EPS)
        cost = -math.log(p_adj)
        G.add_edge(r.src, r.dst, count=int(r.count), p=p, sev_weight=w,
                   p_adj=p_adj, cost=cost)
        rows.append((r.src, r.dst, int(r.count), p, w, p_adj, cost))
    edge_df = pd.DataFrame(rows, columns=["src", "dst", "count", "p", "sev_weight", "p_adj", "cost"])
    return G, edge_df


# ----------------------------------------------------------------------------
# degree distributions & chatty nodes
# ----------------------------------------------------------------------------
def fit_degree(deg_values, models):
    """Return {model: (frozen_dist, label)} fitted to the degree values."""
    x = np.asarray(deg_values, dtype=float)
    x = x[x > 0]
    fits = {}
    if "lognormal" in models and len(x) > 2:
        sigma, loc, scale = stats.lognorm.fit(x, floc=0)
        mu = math.log(scale)
        fits["lognormal"] = (stats.lognorm(sigma, loc=0, scale=scale),
                             f"LogNormal fit μ={mu:.2f}, σ={sigma:.2f}")
    if "pareto" in models and len(x) > 2:
        b, loc, scale = stats.pareto.fit(x, floc=0)
        fits["pareto"] = (stats.pareto(b, loc=0, scale=scale), f"Pareto fit b={b:.2f}")
    return fits


def node_statistics(G):
    nodes = list(G.nodes())
    indeg = np.array([G.in_degree(n) for n in nodes])
    outdeg = np.array([G.out_degree(n) for n in nodes])
    fit_in = fit_degree(indeg, CFG["DEGREE_MODELS"])
    fit_out = fit_degree(outdeg[outdeg > 0], CFG["DEGREE_MODELS"])

    def tail_p(values, fits):
        ps = []
        for v in values:
            cand = [d.sf(v) for d, _ in fits.values()] if v > 0 else [1.0]
            ps.append(min(cand) if cand else 1.0)
        return np.array(ps)

    tail_in = tail_p(indeg, fit_in)
    tail_out = tail_p(outdeg, fit_out)
    ns = pd.DataFrame({
        "node": nodes, "in_degree": indeg, "out_degree": outdeg,
        "tail_p_in": tail_in, "tail_p_out": tail_out,
    })
    ns["min_tail_p"] = ns[["tail_p_in", "tail_p_out"]].min(axis=1)
    ns["flag_in_chatty"] = ns["tail_p_in"] < CFG["ALPHA_CHAT"]
    ns["flag_out_chatty"] = ns["tail_p_out"] < CFG["ALPHA_CHAT"]
    ns = ns.sort_values("min_tail_p").reset_index(drop=True)
    return ns, fit_in, fit_out, indeg, outdeg


# ----------------------------------------------------------------------------
# centralities
# ----------------------------------------------------------------------------
def compute_centralities(G):
    want = CFG["CENTRALITIES"]
    d = {"node": list(G.nodes())}
    if "degree" in want:
        dc = nx.degree_centrality(G); d["degree"] = [dc[n] for n in G]
    if "betweenness" in want:
        bet = nx.betweenness_centrality(G, normalized=True); d["betweenness"] = [bet[n] for n in G]
    if "closeness" in want:
        cl = nx.closeness_centrality(G); d["closeness"] = [cl[n] for n in G]
    if "pagerank" in want:
        pr = nx.pagerank(G, alpha=CFG["PAGERANK_DAMPING"]); d["pagerank"] = [pr[n] for n in G]
    if "eigenvector" in want:
        try:
            eig = nx.eigenvector_centrality_numpy(G); d["eigenvector"] = [eig[n] for n in G]
        except Exception:
            pass
    return pd.DataFrame(d)


# ----------------------------------------------------------------------------
# traversal
# ----------------------------------------------------------------------------
def path_metrics(G, path):
    total_cost, total_logp = 0.0, 0.0
    for u, v in zip(path[:-1], path[1:]):
        d = G[u][v]
        total_cost += d.get("cost", 0.0)
        total_logp += math.log(max(d.get("p_adj", EPS), EPS))
    return total_cost, math.exp(total_logp)


def bfs_paths(G, start, max_depth):
    """BFS tree paths from start up to max_depth hops."""
    seen = {start: [start]}
    q = deque([(start, 0)])
    rows = []
    while q:
        u, depth = q.popleft()
        if depth >= max_depth:
            continue
        for v in G.successors(u):
            if v not in seen:
                seen[v] = seen[u] + [v]
                c, p = path_metrics(G, seen[v])
                rows.append({"target": v, "hops": depth + 1, "path": " -> ".join(seen[v]),
                             "cost": c, "prob": p})
                q.append((v, depth + 1))
    return rows


def dfs_paths(G, start, max_depth):
    """DFS tree paths (depth limited)."""
    rows, seen = [], {start}
    def rec(u, path):
        if len(path) - 1 >= max_depth:
            return
        for v in G.successors(u):
            if v in seen:
                continue
            seen.add(v)
            np_ = path + [v]
            c, p = path_metrics(G, np_)
            rows.append({"target": v, "hops": len(np_) - 1, "path": " -> ".join(np_),
                         "cost": c, "prob": p})
            rec(v, np_)
    sys.setrecursionlimit(10000)
    rec(start, [start])
    return rows


def dijkstra_best(G, start, target):
    try:
        path = nx.dijkstra_path(G, start, target, weight="cost")
    except nx.NetworkXNoPath:
        return None
    c, p = path_metrics(G, path)
    return {"start": start, "target": target, "hops": len(path) - 1,
            "path": " -> ".join(path), "cost": c, "prob": p}


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
def plot_degree(values, fits, name, fname):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(values, bins=20, density=True, alpha=0.5, label=f"{name} hist")
    xs = np.linspace(min(values), max(values), 300)
    for _, (dist, label) in fits.items():
        ax.plot(xs, dist.pdf(xs), lw=2, label=label)
    ax.set_title(f"{name.capitalize()} distribution with fitted curves")
    ax.set_xlabel(name); ax.set_ylabel("Density"); ax.legend()
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)


def plot_chatty_subgraph(G, chatty_nodes, fname, seed):
    H = G.subgraph(chatty_nodes).copy()
    pos = nx.spring_layout(H, seed=seed)
    fig, ax = plt.subplots(figsize=(8, 6.4))
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color="#cc6600", alpha=0.75, arrows=True, width=1.2)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color="#ffe0c2", edgecolors="#994d00", node_size=220)
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=8)
    ax.set_title("Top 20 Chatty Nodes (min tail-p across models)"); ax.axis("off")
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)


def plot_quick_graph(G, fname, seed, max_nodes=150):
    top = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
    H = G.subgraph(top)
    pos = nx.spring_layout(H, seed=seed)
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx(H, pos, ax=ax, node_size=60, font_size=5, arrows=False, edge_color="#aaaaaa")
    ax.set_title("Quick IDS graph (top-degree nodes)"); ax.axis("off")
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def animate_bfs(G, start, max_depth, fname, fps, seed, frames_max=123):
    """GIF: BFS tree grows from `start`; BLUE = hops 1-2, RED = hop >= 3."""
    from PIL import Image
    import io
    depth = nx.single_source_shortest_path_length(G, start, cutoff=max_depth)
    tree_edges = []
    q, seen = deque([start]), {start}
    while q:
        u = q.popleft()
        if depth[u] >= max_depth:
            continue
        for v in G.successors(u):
            if v not in seen:
                seen.add(v); tree_edges.append((u, v)); q.append(v)
    H = nx.DiGraph(); H.add_edges_from(tree_edges)
    pos = nx.spring_layout(H, seed=seed, k=0.6 / math.sqrt(max(len(H), 1)), iterations=30)
    step = max(1, math.ceil(len(tree_edges) / frames_max))
    frames = []
    for i in range(step, len(tree_edges) + step, step):
        sub = tree_edges[:i]
        nodes = {start} | {v for _, v in sub} | {u for u, _ in sub}
        fig, ax = plt.subplots(figsize=(10, 8))
        cols = ["#4f7cff" if depth[v] <= 2 else "#e03c3c" for _, v in sub]
        nx.draw_networkx_edges(H.edge_subgraph(sub), pos, ax=ax, edge_color=cols, alpha=0.7)
        nx.draw_networkx_nodes(H, pos, nodelist=list(nodes), ax=ax, node_size=140,
                               node_color="#e8efff", edgecolors="#1f3a93")
        ax.set_title(f"BFS from {start} (RED = hop ≥ 3)")
        ax.plot([], [], " ", label="BLUE: hops 1–2   RED: hop ≥ 3"); ax.legend(loc="lower left")
        ax.axis("off")
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
        frames.append(Image.open(io.BytesIO(buf.getvalue())).convert("P", palette=Image.ADAPTIVE))
    if frames:
        frames[0].save(fname, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


# ----------------------------------------------------------------------------
def main():
    df = load_data(CFG["CSV_PATH"])
    G, edge_df = build_graph(df)
    edge_df.to_csv(out("edge_probs.csv"), index=False)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    ns, fit_in, fit_out, indeg, outdeg = node_statistics(G)
    ns.to_csv(out("node_stats.csv"), index=False)
    chatty = ns[(ns["flag_in_chatty"]) | (ns["flag_out_chatty"])].head(CFG["TOP_N_CHAT"])
    chatty.to_csv(out("chatty_nodes.csv"), index=False)
    print("Chatty nodes:", chatty["node"].tolist())

    if CFG["CALC_CENTRALITIES"]:
        compute_centralities(G).to_csv(out("centralities.csv"), index=False)

    # pivot / target selection
    start = CFG["START_NODE"]
    if not start and len(G) > 0:
        start = max(G.out_degree(), key=lambda x: x[1])[0]
    target = CFG["TARGET_NODE"]
    if not target and len(G) > 1:
        target = max((n for n in G.in_degree() if n[0] != start), key=lambda x: x[1])[0]
    print("start =", start, " target =", target)

    bfs_list = bfs_paths(G, start, CFG["MAX_DEPTH"])
    dfs_list = dfs_paths(G, start, CFG["MAX_DEPTH"])
    pd.DataFrame(bfs_list).to_csv(out("bfs_paths.csv"), index=False)
    pd.DataFrame(dfs_list).to_csv(out("dfs_paths.csv"), index=False)

    best = dijkstra_best(G, start, target)
    if best:
        pd.DataFrame([best]).to_csv(out("best_path_dijkstra.csv"), index=False)
    cmp_rows = []
    for name, lst in (("BFS", bfs_list), ("DFS", dfs_list)):
        hit = next((r for r in lst if r["target"] == target), None)
        if hit:
            cmp_rows.append({"algorithm": name, **{k: hit[k] for k in ("hops", "path", "cost", "prob")}})
    if best:
        cmp_rows.append({"algorithm": "Dijkstra", **{k: best[k] for k in ("hops", "path", "cost", "prob")}})
    pd.DataFrame(cmp_rows).to_csv(out("algo_compare.csv"), index=False)
    if best:
        print(f"Best path: {best['path']}  cost={best['cost']:.2f}  p={best['prob']:.4f}")

    if CFG["MAKE_PLOTS"]:
        if fit_in:  plot_degree(indeg, fit_in, "in-degree", out("degree_in.png"))
        if fit_out: plot_degree(outdeg[outdeg > 0], fit_out, "out-degree", out("degree_out.png"))
        if len(chatty):
            plot_chatty_subgraph(G, chatty["node"].tolist(), out("chatty_nodes_subgraph.png"), CFG["SEED_LAYOUT"])
        plot_quick_graph(G, out(CFG["PLOT_GRAPH_FILE"]), CFG["SEED_LAYOUT"])
    if CFG["ANIMATE"] and start:
        animate_bfs(G, start, CFG["MAX_DEPTH"], out(CFG["ANIM_FILE"]), CFG["ANIM_FPS"], CFG["SEED_LAYOUT"])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        CFG["CSV_PATH"] = sys.argv[1]
    main()
