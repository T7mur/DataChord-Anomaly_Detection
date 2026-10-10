# -*- coding: utf-8 -*-
"""
steady_state_anomaly (Exercise).py   (RECREATED from Final_Report_TT.docx, sections 8.3, 9.1)

Markov steady-state deviation test for IDS alerts.
  1. State = "<alert>|<severity>"   (missing severity -> its own state, -1)
  2. Per-entity transition counts between consecutive alerts (time ordered)
  3. Transition matrices are SVD-smoothed (rank 6, clipped non-negative)
  4. Steady state pi via power iteration
  5. Entity pi compared with baseline pi: cosine distance + chi-square GOF
  6. Entities with p < chisq_alpha are flagged

Outputs: steady_baseline_pi.csv, steady_entity_pi.csv, steady_entity_chisq.csv,
         steady_entity_distance_scores.csv,
         fig_anomaly_map.png (Fig 9.1), fig_top_suspects.png (Fig 9.2)
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PARAMS = {
    # Column mapping
    "entity_candidates": {"src_ip", "source_ip", "ip", "src", "user", "username", "account", "host", "device"},
    "alert_candidates":  {"alert", "signature", "rule_name", "event", "event_name", "type"},
    "sev_candidates":    {"severity", "priority", "level", "risk"},
    "time_candidates":   {"timestamp", "time", "ts", "date", "datetime", "event_time", "log_time"},

    # State design
    "state_alert_cap": 40,

    # SVD smoothing
    "svd_rank": 6,
    "svd_clip_nonneg": True,            # keeping negative entries at 0

    # Power iteration
    "max_iter": 1500,

    # Entity filtering & ranking
    "min_transitions": 20,
    "distance_metric": "cosine",
    "top_k": 12,

    # Statistical testing (Chi-square GOF)
    "chisq_alpha": .05,
    "chisq_min_expected": 1e-9,         # epsilon to avoid divide-by-zero on expected counts

    # I/O (not shown in the report's screenshot)
    "csv_path": "ids_logs.csv",
    "outdir": "outputs",
    "n_label": 12,                       # how many points to annotate on the anomaly map
}


def find_col(df, cands, required=True):
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in low:
            return low[c]
    if required:
        raise KeyError(f"no column in {sorted(cands)} among {list(df.columns)}")
    return None


# ----------------------------------------------------------------------------
def build_states(df):
    ent = find_col(df, PARAMS["entity_candidates"])
    alt = find_col(df, PARAMS["alert_candidates"])
    sev = find_col(df, PARAMS["sev_candidates"], required=False)
    tcol = find_col(df, PARAMS["time_candidates"], required=False)

    d = pd.DataFrame({"entity": df[ent], "alert": df[alt].astype(str)})
    if sev:
        s = pd.to_numeric(df[sev], errors="coerce")
        d["sev"] = s.fillna(-1).astype(int)            # unknown severity = its own state
    else:
        d["sev"] = -1
    if tcol:
        d["t"] = pd.to_datetime(df[tcol], errors="coerce")
        d = d.dropna(subset=["t"])
    else:
        d["t"] = np.arange(len(d))
    d = d.dropna(subset=["entity"])

    # cap the number of alert categories: keep the most frequent, lump the rest
    top = d["alert"].value_counts().head(PARAMS["state_alert_cap"]).index
    d["alert"] = np.where(d["alert"].isin(top), d["alert"], "OTHER")
    d["state"] = d["alert"] + "|" + d["sev"].astype(str)
    return d.sort_values(["entity", "t"])


def transition_counts(seq, idx, k):
    C = np.zeros((k, k))
    for a, b in zip(seq[:-1], seq[1:]):
        C[idx[a], idx[b]] += 1
    return C


def svd_smooth(C, rank, clip):
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    r = min(rank, len(s))
    A = (U[:, :r] * s[:r]) @ Vt[:r, :]
    if clip:
        A = np.clip(A, 0, None)
    return A


def row_normalise(A):
    A = np.asarray(A, float)
    rs = A.sum(axis=1, keepdims=True)
    k = A.shape[0]
    P = np.where(rs > 0, A / np.where(rs == 0, 1, rs), 1.0 / k)   # empty rows -> uniform
    return P


def steady_state(P, max_iter, tol=1e-12):
    k = P.shape[0]
    pi = np.full(k, 1.0 / k)
    for _ in range(max_iter):
        nxt = pi @ P
        if np.abs(nxt - pi).sum() < tol:
            pi = nxt
            break
        pi = nxt
    return pi / pi.sum()


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def chisq_gof(pi_e, pi_b, n):
    """Chi-square goodness of fit of entity steady state against baseline, scaled to n transitions."""
    obs = pi_e * n
    exp = np.maximum(pi_b * n, PARAMS["chisq_min_expected"])
    keep = pi_b * n > PARAMS["chisq_min_expected"]
    chi2 = float(((obs[keep] - exp[keep]) ** 2 / exp[keep]).sum())
    dof = int(keep.sum() - 1)
    p = float(stats.chi2.sf(chi2, dof)) if dof > 0 else 1.0
    return chi2, dof, p


# ----------------------------------------------------------------------------
def main():
    os.makedirs(PARAMS["outdir"], exist_ok=True)
    df = pd.read_csv(PARAMS["csv_path"])
    d = build_states(df)

    states = sorted(d["state"].unique())
    idx = {s: i for i, s in enumerate(states)}
    k = len(states)

    seqs = {e: g["state"].tolist() for e, g in d.groupby("entity")}
    counts = {e: transition_counts(s, idx, k) for e, s in seqs.items()}
    ntrans = {e: len(s) - 1 for e, s in seqs.items()}

    # baseline: pooled transition counts of every entity
    C_all = sum(counts.values())
    P_base = row_normalise(svd_smooth(C_all, PARAMS["svd_rank"], PARAMS["svd_clip_nonneg"]))
    pi_base = steady_state(P_base, PARAMS["max_iter"])
    pd.DataFrame({"state": states, "pi": pi_base}).to_csv(
        os.path.join(PARAMS["outdir"], "steady_baseline_pi.csv"), index=False)

    pi_rows, chi_rows, dist_rows = [], [], []
    for e, C in counts.items():
        n = ntrans[e]
        if n < PARAMS["min_transitions"]:
            continue
        P_e = row_normalise(svd_smooth(C, PARAMS["svd_rank"], PARAMS["svd_clip_nonneg"]))
        pi_e = steady_state(P_e, PARAMS["max_iter"])
        pi_rows.append({"entity": e, **dict(zip(states, pi_e))})
        chi2, dof, p = chisq_gof(pi_e, pi_base, n)
        chi_rows.append({"entity": e, "transitions": n, "chi2_stat": chi2, "dof": dof,
                         "p_value": p, "significant": p < PARAMS["chisq_alpha"]})
        sim = cosine_sim(pi_e, pi_base)
        dist_rows.append({"entity": e, "transitions": n, "cosine_similarity": sim,
                          "distance": 1 - sim})

    pd.DataFrame(pi_rows).to_csv(os.path.join(PARAMS["outdir"], "steady_entity_pi.csv"), index=False)
    chi = pd.DataFrame(chi_rows).sort_values("p_value")
    chi.to_csv(os.path.join(PARAMS["outdir"], "steady_entity_chisq.csv"), index=False)
    dist = pd.DataFrame(dist_rows).sort_values("distance", ascending=False)
    dist.to_csv(os.path.join(PARAMS["outdir"], "steady_entity_distance_scores.csv"), index=False)

    print("Significant entities (p < %.2f):" % PARAMS["chisq_alpha"],
          chi.loc[chi["significant"], "entity"].tolist())

    # ------------------------------------------------------------ Fig 9.1
    m = chi.merge(dist[["entity", "cosine_similarity"]], on="entity")
    m["neglogp"] = -np.log10(np.clip(m["p_value"], 1e-300, 1))
    fig, ax = plt.subplots(figsize=(10, 7))
    sizes = 20 + 160 * (m["transitions"] - m["transitions"].min()) / max(1, np.ptp(m["transitions"]))
    ax.scatter(m["cosine_similarity"], m["neglogp"], s=sizes, alpha=0.7)
    thr = -np.log10(PARAMS["chisq_alpha"])
    ax.axhline(thr, ls="--", lw=1)
    ax.text(m["cosine_similarity"].min(), thr + 0.05, "p = 0.05", fontsize=9)
    for r in m.sort_values("p_value").head(PARAMS["n_label"]).itertuples():
        ax.annotate(r.entity, (r.cosine_similarity, r.neglogp), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Cosine similarity to baseline π  (higher = more similar)")
    ax.set_ylabel("−log$_{10}$($p$-value)  (Chi-square GOF)")
    ax.set_title("Anomaly Map: Cosine Similarity vs Significance\n(size ~ #transitions)")
    fig.tight_layout(); fig.savefig(os.path.join(PARAMS["outdir"], "fig_anomaly_map.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------ Fig 9.2
    top = dist.head(PARAMS["top_k"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top["entity"], top["distance"])
    ax.set_ylabel("Distance to baseline π (cosine)"); ax.set_xlabel("Entity")
    ax.set_title("Entities Most Different from Baseline Steady State (SVD-smoothed)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(PARAMS["outdir"], "fig_top_suspects.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        PARAMS["csv_path"] = sys.argv[1]
    main()
