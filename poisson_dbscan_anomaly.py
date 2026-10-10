# -*- coding: utf-8 -*-
"""
poisson_dbscan_anomaly (practice).py   (RECREATED from Final_Report_TT.docx, sections 8.5, 9.3)

Failed-login anomaly detection:
  1. Split failed-login timeline into BASELINE and RECENT windows
  2. Per user: Poisson rate-shift likelihood-ratio test (H0: lambda_base = lambda_recent)
  3. Per user: burstiness = max failed logins in any BIN_MINUTES bin of the recent window
  4. DBSCAN on standardised [log10 recent rate, log10 rate ratio, LLR, burstiness]
     (label -1 = outlier / anomaly)

Outputs (PLOTS_DIR / cwd): poisson_failed_login_users.csv, dbscan_failed_login_labels.csv,
  failed_login_anomaly_users.csv, failed_login_outliers_only.csv, and plots
  fig_top_recent_rates.png, fig_dbscan_pca.png, fig_dbscan_feature_space.png, fig_poisson_volcano.png
"""
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PARAMS: Dict[str, object] = {
    # === Data source ===
    "CSV_PATH": "user_behavior_analytics_ttt.csv",
    # === Column alias sets (auto-detect) ===
    "time_candidates":   {"timestamp", "time", "datetime", "date", "event_time", "log_time"},
    "user_candidates":   {"user", "username", "account", "actor", "principal"},
    "act_candidates":    {"activity", "event", "action", "type", "operation", "message"},
    "status_candidates": {"result", "status", "outcome", "success", "reason"},  # optional

    # === Time-window split === # A split of recent (actual) and baseline (normal).
    "SPLIT_STRATEGY": "fraction_split",  # 'fraction_split': recent starts at the frac-quantile of the FAILED-LOGIN timestamps
    "RECENT_DAYS_PREF": None,
    "RECENT_FRACTION": 0.8,              # (not visible in the report screenshots) -> recent ~ last 20 % of failed-login events

    # === Minimum events per user to include ===
    "MIN_BASE_FAILS": 3,                 # min failed logins per user so that user is included in baseline
    "MIN_RECENT_FAILS": 1,               # min failed logins recently; set to 0 if you want to include zeros

    # === Poisson test settings ===
    "ALPHA": 0.05,                       # significance line for the LRT -- H0: lambda_base = lambda_recent
    "SMOOTH": 1,                         # added to counts to stabilise rate ratio when zeros occur

    # === Burstiness feature ===
    "BIN_MINUTES": 15,                   # size of time bin for max-burst feature (recent window)

    # === DBSCAN (feature-space clustering) === labels dense regions with cluster IDs and outliers as -1
    "DBSCAN_EPS": 1.1,
    "DBSCAN_MIN_SAMPLES": 6,             # min points to form a dense region

    # === Plots ===
    "MAKE_PLOTS": True,
    "PLOTS_DIR": "plots",                # folder to save all figures
    "SHOW_PLOTS": False,

    # === Plot annotations ===
    "ANNOTATE_OUTLIERS": True,           # write labels next to outlier points
    "MAX_LABELS": 40,                    # guard against clutter: cap number of labels per plot
}


def find_col(df, cands, required=True):
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in low:
            return low[c]
    if required:
        raise KeyError(f"None of {sorted(cands)} found in {list(df.columns)}")
    return None


FAIL_WORDS = ("fail", "denied", "invalid", "incorrect", "unsuccess", "false", "error", "reject")


def load_failed_logins(path):
    df = pd.read_csv(path)
    tc = find_col(df, PARAMS["time_candidates"])
    uc = find_col(df, PARAMS["user_candidates"])
    ac = find_col(df, PARAMS["act_candidates"])
    sc = find_col(df, PARAMS["status_candidates"], required=False)

    d = pd.DataFrame({"ts": pd.to_datetime(df[tc], errors="coerce"), "user": df[uc],
                      "act": df[ac].astype(str).str.lower()})
    d["status"] = df[sc].astype(str).str.lower() if sc else ""
    d = d.dropna(subset=["ts", "user"])
    text = d["act"] + " " + d["status"]
    is_login = text.str.contains("login|logon|auth|sign")
    is_fail = text.apply(lambda s: any(w in s for w in FAIL_WORDS))
    return d[is_login & is_fail].sort_values("ts").reset_index(drop=True)


def split_windows(f):
    t0, t1 = f["ts"].min(), f["ts"].max()
    if PARAMS["SPLIT_STRATEGY"] == "fraction_split" or PARAMS["RECENT_DAYS_PREF"] is None:
        cut = f["ts"].quantile(PARAMS["RECENT_FRACTION"])
    else:
        cut = t1 - pd.Timedelta(days=PARAMS["RECENT_DAYS_PREF"])
    return t0, cut, t1


def poisson_lrt(kb, Tb, kr, Tr):
    """Two-sample Poisson likelihood-ratio statistic (1 d.o.f.)."""
    lam = (kb + kr) / (Tb + Tr)
    def term(k, T):
        return k * np.log(k / (lam * T)) if k > 0 else 0.0
    G = 2.0 * (term(kb, Tb) + term(kr, Tr))
    return G, float(stats.chi2.sf(G, 1))


def per_user_table(f, t0, cut, t1):
    Tb = (cut - t0).total_seconds()
    Tr = (t1 - cut).total_seconds()
    s = PARAMS["SMOOTH"]
    rows = []
    base = f[f["ts"] < cut]
    rec = f[f["ts"] >= cut]
    users = set(base["user"]) | set(rec["user"])
    for u in sorted(users):
        kb = int((base["user"] == u).sum())
        kr = int((rec["user"] == u).sum())
        if kb < PARAMS["MIN_BASE_FAILS"] or kr < PARAMS["MIN_RECENT_FAILS"]:
            continue
        rate_b = (kb + s) / (Tb / 3600.0)
        rate_r = (kr + s) / (Tr / 3600.0)
        # raw per-hour rates for reporting; smoothed ones feed the ratio
        raw_b, raw_r = kb / (Tb / 3600.0), kr / (Tr / 3600.0)
        ratio = rate_r / rate_b
        G, p = poisson_lrt(kb, Tb, kr, Tr)
        llr = G / 2.0
        r_ts = rec.loc[rec["user"] == u, "ts"]
        bins = r_ts.dt.floor(f"{PARAMS['BIN_MINUTES']}min").value_counts()
        burst = int(bins.max()) if len(bins) else 0
        rows.append({"user": u, "k_base": kb, "k_recent": kr, "T_base_sec": Tb, "T_recent_sec": Tr,
                     "rate_base_per_hr": raw_b, "rate_recent_per_hr": raw_r,
                     "rate_ratio": ratio, "llr": llr, "G_stat": G, "p_value": p,
                     "poisson_significant": p < PARAMS["ALPHA"],
                     "burst_recent_max_per_bin": burst})
    return pd.DataFrame(rows)


def dbscan_stage(tbl):
    X = pd.DataFrame({
        "log_rate_recent": np.log10(tbl["rate_recent_per_hr"].clip(lower=1e-9)),
        "log_rate_ratio": np.log10(tbl["rate_ratio"].clip(lower=1e-9)),
        "llr": tbl["llr"],
        "burst": tbl["burst_recent_max_per_bin"],
    }).fillna(0.0)                                        # null-derived features -> neutral baseline value
    Z = StandardScaler().fit_transform(X)
    labels = DBSCAN(eps=PARAMS["DBSCAN_EPS"], min_samples=PARAMS["DBSCAN_MIN_SAMPLES"]).fit_predict(Z)
    out = tbl.copy()
    out["dbscan_label"] = labels
    out["dbscan_is_anomaly"] = labels == -1
    return out, X, Z


def make_plots(res, X, Z):
    d = PARAMS["PLOTS_DIR"]
    os.makedirs(d, exist_ok=True)
    out = res[res["dbscan_is_anomaly"]]
    colours = np.where(res["dbscan_is_anomaly"], "#e0555a", "grey")
    lab_budget = PARAMS["MAX_LABELS"]

    # Fig 9.8 - top users by recent failed-login rate (flagged users; falls back to top 10)
    top = (out if len(out) else res).sort_values("rate_recent_per_hr", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(top["user"], top["rate_recent_per_hr"])
    for b, v in zip(bars, top["rate_recent_per_hr"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Recent failed-login rate (per hour)"); ax.set_title("Top users by recent failed-login rate")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(d, "fig_top_recent_rates.png"), dpi=200); plt.close(fig)

    # Fig 9.9 - PCA projection
    n_comp = min(2, Z.shape[1], Z.shape[0])
    if n_comp == 2:
        P = PCA(n_components=2).fit_transform(Z)
        fig, ax = plt.subplots(figsize=(8.8, 6.6))
        ax.scatter(P[:, 0], P[:, 1], c=colours, s=14)
        if PARAMS["ANNOTATE_OUTLIERS"]:
            for i in np.where(res["dbscan_is_anomaly"].values)[0][:lab_budget]:
                ax.annotate(res["user"].iloc[i], (P[i, 0], P[i, 1]), xytext=(4, 4),
                            textcoords="offset points", fontsize=8)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("DBSCAN in PCA space (outliers in red)")
        fig.tight_layout(); fig.savefig(os.path.join(d, "fig_dbscan_pca.png"), dpi=200); plt.close(fig)

    # Fig 9.10 - interpretable feature space
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    ax.scatter(X["log_rate_recent"], X["log_rate_ratio"], c=colours, s=14)
    ax.axhline(0, ls="--", lw=1); ax.axvline(0, ls="--", lw=1)
    if PARAMS["ANNOTATE_OUTLIERS"]:
        for i in np.where(res["dbscan_is_anomaly"].values)[0][:lab_budget]:
            ax.annotate(res["user"].iloc[i], (X["log_rate_recent"].iloc[i], X["log_rate_ratio"].iloc[i]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("log10(Recent fail-rate per hour)"); ax.set_ylabel("log10(Rate Ratio: recent/base)")
    ax.set_title("Failed-Login Feature Space — DBSCAN outliers in red (label -1)")
    fig.tight_layout(); fig.savefig(os.path.join(d, "fig_dbscan_feature_space.png"), dpi=200); plt.close(fig)

    # Fig 9.11 - Poisson volcano (flagged users)
    v = out if len(out) else res
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    ax.scatter(np.log10(v["rate_ratio"]), -np.log10(v["p_value"].clip(lower=1e-300)), s=22, alpha=0.7)
    thr = -np.log10(PARAMS["ALPHA"])
    ax.axhline(thr, ls="--", lw=1)
    ax.text(ax.get_xlim()[0], thr + 0.02, "p = 0.05", fontsize=9)
    ax.set_xlabel("log10(Rate Ratio: recent/base)"); ax.set_ylabel("-log10(p-value) (Poisson LRT)")
    ax.set_title("Volcano Plot: Rate Shift vs Significance")
    fig.tight_layout(); fig.savefig(os.path.join(d, "fig_poisson_volcano.png"), dpi=200); plt.close(fig)
    if PARAMS["SHOW_PLOTS"]:
        plt.show()


def main():
    f = load_failed_logins(PARAMS["CSV_PATH"])
    print(f"{len(f)} failed-login events, {f['user'].nunique()} users")
    t0, cut, t1 = split_windows(f)
    print(f"baseline {(cut - t0).total_seconds() / 3600:.1f} h | recent {(t1 - cut).total_seconds() / 3600:.1f} h")
    tbl = per_user_table(f, t0, cut, t1)
    tbl.to_csv("poisson_failed_login_users.csv", index=False)

    res, X, Z = dbscan_stage(tbl)
    res[["user", "dbscan_label", "dbscan_is_anomaly"]].to_csv("dbscan_failed_login_labels.csv", index=False)
    res.to_csv("failed_login_anomaly_users.csv", index=False)
    res[res["dbscan_is_anomaly"]].to_csv("failed_login_outliers_only.csv", index=False)
    print("DBSCAN outliers:", res.loc[res["dbscan_is_anomaly"], "user"].tolist())
    if PARAMS["MAKE_PLOTS"]:
        make_plots(res, X, Z)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        PARAMS["CSV_PATH"] = sys.argv[1]
    main()
