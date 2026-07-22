import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


LONG_PATH = Path("outputs/mimic_cs_lactate_24h_long.csv")
ANALYSIS_PATH = Path("outputs/mimic_cs_analysis_dataset.csv")
OUT_DIR = Path("outputs")


BIN_LABELS = ["lact_0_6h", "lact_6_12h", "lact_12_18h", "lact_18_24h"]
BIN_MIDPOINTS = np.array([3.0, 9.0, 15.0, 21.0])


def make_feature_matrix(long_df):
    df = (
        long_df.groupby(["stay_id", "lactate_hour"], as_index=False)["lactate"]
        .median()
        .copy()
    )
    df["time_bin"] = pd.cut(
        df["lactate_hour"],
        bins=[0, 6, 12, 18, 24],
        labels=BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    # Median is robust to multiple draws in one time bin.
    wide = (
        df.dropna(subset=["time_bin"])
        .groupby(["stay_id", "time_bin"], observed=True)["lactate"]
        .median()
        .unstack()
        .reindex(columns=BIN_LABELS)
    )

    # Interpolate within patient across time bins. Remaining edge missingness is
    # filled by column medians to preserve all eligible stays.
    interpolated = wide.apply(
        lambda row: row.astype(float).interpolate(limit_direction="both"), axis=1
    )
    for col in BIN_LABELS:
        interpolated[col] = interpolated[col].fillna(interpolated[col].median())

    # Log transform reduces dominance of extreme lactate values.
    log_features = np.log1p(interpolated[BIN_LABELS])
    return wide, interpolated, log_features


def choose_clusters(features):
    scaled = StandardScaler().fit_transform(features)
    rows = []
    labels_by_k = {}
    for k in range(2, 6):
        model = KMeans(n_clusters=k, random_state=2026, n_init=50)
        labels = model.fit_predict(scaled)
        sil = silhouette_score(scaled, labels)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(sil),
                "cluster_sizes": {
                    str(i): int((labels == i).sum()) for i in range(k)
                },
            }
        )
        labels_by_k[k] = labels
    return rows, labels_by_k


def order_and_name_clusters(features_raw, labels):
    tmp = features_raw.copy()
    tmp["cluster_raw"] = labels
    means = tmp.groupby("cluster_raw")[BIN_LABELS].mean()
    means["overall_mean"] = means[BIN_LABELS].mean(axis=1)
    means["last"] = means["lact_18_24h"]
    means["slope"] = means["lact_18_24h"] - means["lact_0_6h"]
    ordered = means.sort_values(["overall_mean", "last"]).index.tolist()
    mapping = {raw: rank + 1 for rank, raw in enumerate(ordered)}
    named = pd.Series(labels, index=features_raw.index).map(mapping).astype(int)

    # Clinical names based on ordered severity and slope.
    summary = []
    for group in sorted(named.unique()):
        raw = [r for r, g in mapping.items() if g == group][0]
        row = means.loc[raw]
        if group == 1:
            name = "low_stable"
        elif row["slope"] <= -1:
            name = f"group_{group}_decreasing"
        elif row["last"] >= 4:
            name = f"group_{group}_persistent_high"
        else:
            name = f"group_{group}_moderate"
        summary.append(
            {
                "trajectory_group": int(group),
                "trajectory_name": name,
                "n": int((named == group).sum()),
                "mean_0_6h": float(row["lact_0_6h"]),
                "mean_6_12h": float(row["lact_6_12h"]),
                "mean_12_18h": float(row["lact_12_18h"]),
                "mean_18_24h": float(row["lact_18_24h"]),
                "slope_0_24h": float(row["slope"]),
            }
        )
    return named, summary


def mortality_by_group(analysis_df, assignments):
    df = analysis_df.merge(
        assignments[["stay_id", "trajectory_group", "trajectory_name"]],
        on="stay_id",
        how="inner",
    )
    out = []
    for (group, name), sub in df.groupby(["trajectory_group", "trajectory_name"]):
        deaths = int(sub["hospital_expire_flag"].sum())
        out.append(
            {
                "trajectory_group": int(group),
                "trajectory_name": name,
                "n": int(len(sub)),
                "deaths": deaths,
                "mortality_pct": deaths / len(sub) * 100,
                "initial_lactate_median": float(sub["initial_lactate_24h"].median()),
                "last_lactate_median": float(sub["last_lactate_24h"].median()),
                "peak_lactate_median": float(sub["peak_lactate_24h"].median()),
            }
        )
    return df, out


def logistic_trajectory(df):
    covariates = [
        "age",
        "sofa",
        "charlson_comorbidity_index",
        "mbp_mean",
        "creatinine_max",
        "mechvent_24h",
        "vasoactive_24h",
    ]
    model_df = df[["hospital_expire_flag", "trajectory_group"] + covariates].copy()
    model_df = model_df.dropna()
    dummies = pd.get_dummies(
        model_df["trajectory_group"].astype(int),
        prefix="traj",
        drop_first=True,
        dtype=float,
    )
    x = pd.concat([dummies, model_df[covariates].astype(float)], axis=1)
    x = sm.add_constant(x, has_constant="add")
    y = model_df["hospital_expire_flag"].astype(float)
    fit = sm.Logit(y, x).fit(disp=False)
    conf = fit.conf_int()
    terms = {}
    for term in fit.params.index:
        terms[term] = {
            "coef": float(fit.params[term]),
            "or": float(math.exp(fit.params[term])),
            "ci95_low": float(math.exp(conf.loc[term, 0])),
            "ci95_high": float(math.exp(conf.loc[term, 1])),
            "p": float(fit.pvalues[term]),
        }
    return {"n_complete_cases": int(len(model_df)), "aic": float(fit.aic), "terms": terms}


def plot_trajectories(summary, path):
    plt.figure(figsize=(7, 4.5), dpi=160)
    for row in summary:
        y = [
            row["mean_0_6h"],
            row["mean_6_12h"],
            row["mean_12_18h"],
            row["mean_18_24h"],
        ]
        label = f"G{row['trajectory_group']} {row['trajectory_name']} (n={row['n']})"
        plt.plot(BIN_MIDPOINTS, y, marker="o", linewidth=2, label=label)
    plt.xlabel("Hours after ICU admission")
    plt.ylabel("Mean lactate (mmol/L)")
    plt.title("Early lactate trajectory groups")
    plt.xticks(BIN_MIDPOINTS, ["0-6", "6-12", "12-18", "18-24"])
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    long_df = pd.read_csv(LONG_PATH)
    analysis_df = pd.read_csv(ANALYSIS_PATH)

    wide_observed, wide_filled, log_features = make_feature_matrix(long_df)
    metrics, labels_by_k = choose_clusters(log_features)

    # Main analysis: k=4 gives clinically interpretable low, moderate, high-decreasing,
    # and persistent/rising high trajectories. Metrics for 2-5 are still exported.
    main_k = 4
    labels = labels_by_k[main_k]
    ordered_labels, trajectory_summary = order_and_name_clusters(wide_filled, labels)

    assignments = pd.DataFrame(
        {
            "stay_id": wide_filled.index.astype(int),
            "trajectory_group": ordered_labels.values,
        }
    )
    name_map = {
        row["trajectory_group"]: row["trajectory_name"] for row in trajectory_summary
    }
    assignments["trajectory_name"] = assignments["trajectory_group"].map(name_map)
    for col in BIN_LABELS:
        assignments[col] = wide_filled[col].values

    merged, mortality_summary = mortality_by_group(analysis_df, assignments)
    regression = logistic_trajectory(merged)

    assignments.to_csv(OUT_DIR / "mimic_lactate_trajectory_assignments.csv", index=False)
    merged.to_csv(OUT_DIR / "mimic_analysis_dataset_with_trajectory.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT_DIR / "mimic_lactate_cluster_metrics.csv", index=False)
    pd.DataFrame(trajectory_summary).to_csv(
        OUT_DIR / "mimic_lactate_trajectory_summary.csv", index=False
    )
    pd.DataFrame(mortality_summary).to_csv(
        OUT_DIR / "mimic_lactate_trajectory_mortality.csv", index=False
    )
    plot_trajectories(trajectory_summary, OUT_DIR / "mimic_lactate_trajectory_plot.png")

    result = {
        "main_k": main_k,
        "cluster_metrics": metrics,
        "trajectory_summary": trajectory_summary,
        "mortality_summary": mortality_summary,
        "adjusted_logistic_trajectory": regression,
        "outputs": {
            "assignments": "outputs/mimic_lactate_trajectory_assignments.csv",
            "analysis_with_trajectory": "outputs/mimic_analysis_dataset_with_trajectory.csv",
            "cluster_metrics": "outputs/mimic_lactate_cluster_metrics.csv",
            "trajectory_summary": "outputs/mimic_lactate_trajectory_summary.csv",
            "mortality": "outputs/mimic_lactate_trajectory_mortality.csv",
            "plot": "outputs/mimic_lactate_trajectory_plot.png",
        },
    }
    with open(OUT_DIR / "mimic_lactate_trajectory_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
