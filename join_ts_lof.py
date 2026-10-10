# -*- coding: utf-8 -*-
"""
join_ts_lof (practice).py   (RECREATED from Final_Report_TT.docx, sections 8.4, 9.4, 9.5)

Two-part IDS model:
  A) Time-series burst detector: 1-min event counts, rolling mean/std (ROLL bins),
     z-score, anomaly where |z| >= Z_THR.
  B) Local Outlier Factor over per-IP features (event count, severity mean/variance,
     burstiness), joined with the time-series information.

Outputs (in OUTDIR): timeseries_counts.csv, lof_ip_scores.csv and the EDA figures
  fig_event_counts_boxplot.png, fig_event_counts_hist_ecdf.png, fig_event_counts_qq.png,
  fig_ts_zscore_hist_ecdf.png, fig_ts_rolling_zscore.png, fig_ts_anomalies.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

PARAMS = {
    "CSV_PATH": "ids_logs.csv",
    "OUTDIR":   "outputs",

    # -- Column overrides (None = auto-detect) --------------
    "TIME_COL": "timestamp",
    "IP_COL":   "source_ip",
    "SEV_COL":  "severity",

    # -- Time-series settings (affect the spike detector) ----------
    #        - "1min": very detailed; good for high-volume logs.
    "BIN":  "1min",

    # ROLL:  Rolling window length (in number of bins) used for rolling mean/std.
    "ROLL": 10,

    # Z_THR: Absolute z-score threshold for flagging spikes.
    #        - 3.0 -> balanced default
    "Z_THR": 3.0,

    # -- LOF settings (peer-based outlier detection per IP) ----------
    # LOF_NEIGHBORS: Number of neighbors for Local Outlier Factor.
    "LOF_NEIGHBORS": 15,

    # LOF_CONTAM: Expected fraction of outliers
    "LOF_CONTAM": 0.01,

    # -- Plotting --
    "SHOW_PLOTS": True,
}


def o(name):
    os.makedirs(PARAMS["OUTDIR"], exist_ok=True)
    return os.path.join(PARAMS["OUTDIR"], name)


def ecdf(x):
    x = np.sort(np.asarray(x, float))
    return x, np.arange(1, len(x) + 1) / len(x)


def load():
    df = pd.read_csv(PARAMS["CSV_PATH"])
    df = df.rename(columns={PARAMS["TIME_COL"]: "ts", PARAMS["IP_COL"]: "ip", PARAMS["SEV_COL"]: "sev"})
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "ip"])
    df["sev"] = pd.to_numeric(df["sev"], errors="coerce")
    df.loc[df["sev"] < 0, "sev"] = np.nan
    return df


# ---------------------------------------------------------------- A) time series
def time_series(df):
    ts = (df.set_index("ts").resample(PARAMS["BIN"]).size().rename("count").to_frame())
    r = ts["count"].rolling(PARAMS["ROLL"])             # window includes the current bin
    ts["roll_mean"] = r.mean()
    ts["roll_std"] = r.std()
    ts["z"] = (ts["count"] - ts["roll_mean"]) / ts["roll_std"].replace(0, np.nan)
    ts["anomaly_ts"] = ts["z"].abs() >= PARAMS["Z_THR"]
    return ts


# ---------------------------------------------------------------- B) LOF
def lof_table(df):
    df = df.copy()
    df["minute"] = df["ts"].dt.floor("1min")
    g = df.groupby("ip")
    feat = pd.DataFrame({
        "event_count": g.size(),
        "sev_mean": g["sev"].mean(),
        "sev_var": g["sev"].var(),
    })
    # per-IP burstiness: max of the rolling sum of events over ROLL minutes
    per_min = df.groupby(["ip", "minute"]).size().rename("n").reset_index()
    burst = {}
    for ip, sub in per_min.groupby("ip"):
        s = sub.set_index("minute")["n"].asfreq("1min", fill_value=0)
        burst[ip] = s.rolling(PARAMS["ROLL"], min_periods=1).max().max()
    feat["burst_rollmax"] = pd.Series(burst)

    # null handling (report 6.6 D): activity features -> 0, severity features -> median
    feat["event_count"] = feat["event_count"].fillna(0)
    feat["burst_rollmax"] = feat["burst_rollmax"].fillna(0)
    for c in ("sev_mean", "sev_var"):
        feat[c] = feat[c].fillna(feat[c].median())

    X = StandardScaler().fit_transform(feat[["event_count", "sev_mean", "sev_var", "burst_rollmax"]])
    k = min(PARAMS["LOF_NEIGHBORS"], len(feat) - 1)
    lof = LocalOutlierFactor(n_neighbors=k, contamination=PARAMS["LOF_CONTAM"])
    feat["label"] = lof.fit_predict(X)                       # -1 = outlier
    feat["lof_score"] = -lof.negative_outlier_factor_
    return feat.reset_index().rename(columns={"index": "ip"}).sort_values("lof_score", ascending=False)


# ---------------------------------------------------------------- plots
def make_plots(ts, feat):
    # 9.12 boxplot
    fig, ax = plt.subplots(figsize=(9.1, 5))
    try:
        ax.boxplot(feat["event_count"], orientation="horizontal")
    except TypeError:                      # older matplotlib
        ax.boxplot(feat["event_count"], vert=False)
    ax.set_title("Event Counts per IP — Boxplot", fontweight="bold")
    ax.set_xlabel("event_count", fontsize=12); ax.grid(ls=":", alpha=0.6)
    fig.tight_layout(); fig.savefig(o("fig_event_counts_boxplot.png"), dpi=150); plt.close(fig)

    # 9.13 histogram + ECDF
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5))
    ax[0].hist(feat["event_count"], bins=30, density=True)
    ax[0].set_title("Event Counts per IP — Histogram", fontweight="bold")
    ax[0].set_xlabel("event_count", fontsize=12); ax[0].set_ylabel("density", fontsize=12)
    x, y = ecdf(feat["event_count"])
    ax[1].step(x, y, where="post", lw=2)
    ax[1].set_title("Event Counts per IP — ECDF", fontweight="bold")
    ax[1].set_xlabel("event_count", fontsize=12); ax[1].set_ylabel("F(x)", fontsize=12)
    for a in ax: a.grid(ls=":", alpha=0.6)
    fig.tight_layout(); fig.savefig(o("fig_event_counts_hist_ecdf.png"), dpi=150); plt.close(fig)

    # 9.14 QQ plot
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    stats.probplot(feat["event_count"], dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor("blue"); ax.get_lines()[0].set_color("blue")
    ax.get_lines()[1].set_color("red")
    ax.set_title("QQ Plot: Event Counts per IP vs Normal", fontweight="bold")
    ax.set_xlabel("Theoretical quantiles", fontsize=12); ax.set_ylabel("Ordered Values", fontsize=12)
    ax.grid(ls=":", alpha=0.6)
    fig.tight_layout(); fig.savefig(o("fig_event_counts_qq.png"), dpi=150); plt.close(fig)

    z = ts["z"].dropna()
    # 9.15 z histogram + ECDF
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5))
    ax[0].hist(z, bins=30, density=True)
    ax[0].set_title("Time-series z-scores — Histogram", fontweight="bold")
    ax[0].set_xlabel("z", fontsize=12); ax[0].set_ylabel("density", fontsize=12)
    x, y = ecdf(z)
    ax[1].step(x, y, where="post", lw=2)
    ax[1].set_title("Time-series z-scores — ECDF", fontweight="bold")
    ax[1].set_xlabel("z", fontsize=12); ax[1].set_ylabel("F(x)", fontsize=12)
    for a in ax: a.grid(ls=":", alpha=0.6)
    fig.tight_layout(); fig.savefig(o("fig_ts_zscore_hist_ecdf.png"), dpi=150); plt.close(fig)

    # 9.16 counts + rolling stats, z-score panel
    anom = ts[ts["anomaly_ts"]]
    fig, ax = plt.subplots(2, 1, figsize=(12.8, 8), sharex=True)
    ax[0].plot(ts.index, ts["count"], label="count")
    ax[0].plot(ts.index, ts["roll_mean"], "--", color="tab:orange", label=f"rolling mean ({PARAMS['ROLL']})")
    ax[0].fill_between(ts.index, ts["roll_mean"] - ts["roll_std"], ts["roll_mean"] + ts["roll_std"],
                       alpha=0.2, label="±1 std")
    ax[0].scatter(anom.index, anom["count"], color="tab:orange", zorder=5, label="anomaly")
    ax[0].set_ylabel("events", fontsize=12)
    ax[0].set_title(f"Event Counts ({PARAMS['BIN']}) with Rolling Stats", fontweight="bold")
    ax[0].legend(loc="upper left")
    ax[1].plot(ts.index, ts["z"])
    ax[1].axhline(PARAMS["Z_THR"], ls="--", lw=2); ax[1].axhline(-PARAMS["Z_THR"], ls="--", lw=2)
    ax[1].set_ylabel("z"); ax[1].set_xlabel("time", fontsize=12)
    ax[1].set_title(f"Z-scores (rolling={PARAMS['ROLL']}, threshold=±{PARAMS['Z_THR']})", fontweight="bold")
    ax[1].xaxis.set_major_locator(mdates.DayLocator())
    ax[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax[1].xaxis.get_major_locator()))
    for a in ax: a.grid(ls=":", alpha=0.6)
    fig.tight_layout(); fig.savefig(o("fig_ts_rolling_zscore.png"), dpi=150); plt.close(fig)

    # 9.17 simple anomaly overlay
    fig, ax = plt.subplots(figsize=(12.8, 5))
    ax.plot(ts.index, ts["count"], label="count")
    ax.scatter(anom.index, anom["count"], zorder=5, label="anomaly")
    ax.set_title(f"Time Series Anomalies ({PARAMS['BIN']})", fontweight="bold")
    ax.set_xlabel("time", fontsize=12); ax.set_ylabel("events", fontsize=12)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(ls=":", alpha=0.6); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(o("fig_ts_anomalies.png"), dpi=150); plt.close(fig)


def main():
    df = load()
    ts = time_series(df)
    ts.reset_index().to_csv(o("timeseries_counts.csv"), index=False)
    print(f"{len(ts)} bins from {ts.index.min()} to {ts.index.max()};"
          f" max |z| = {ts['z'].abs().max():.2f}; anomalies = {int(ts['anomaly_ts'].sum())}")

    feat = lof_table(df)
    feat.to_csv(o("lof_ip_scores.csv"), index=False)
    print("LOF outliers:", feat.loc[feat["label"] == -1, "ip"].tolist())
    if PARAMS["SHOW_PLOTS"]:
        make_plots(ts, feat)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        PARAMS["CSV_PATH"] = sys.argv[1]
    main()
