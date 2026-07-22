import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


LONG_PATH = Path("outputs/mimic_cs_lactate_24h_long.csv")
ANALYSIS_PATH = Path("outputs/mimic_cs_analysis_dataset.csv")
OUT_DIR = Path("outputs")

BIN_LABELS = ["lact_0_6h", "lact_6_12h", "lact_12_18h", "lact_18_24h"]


def build_features(long_df, min_lactate_count=2):
    counts = long_df.groupby("stay_id").size()
    keep = counts[counts >= min_lactate_count].index
    df = (
        long_df[long_df["stay_id"].isin(keep)]
        .groupby(["stay_id", "lactate_hour"], as_index=False)["lactate"]
        .median()
    )
    df["time_bin"] = pd.cut(
        df["lactate_hour"],
        bins=[0, 6, 12, 18, 24],
        labels=BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    wide = (
        df.dropna(subset=["time_bin"])
        .groupby(["stay_id", "time_bin"], observed=True)["lactate"]
        .median()
        .unstack()
        .reindex(columns=BIN_LABELS)
    )
    filled = wide.apply(lambda row: row.astype(float).interpolate(limit_direction="both"), axis=1)
    for col in BIN_LABELS:
        filled[col] = filled[col].fillna(filled[col].median())
    return filled


def assign_ordered_clusters(features, k):
    x = StandardScaler().fit_transform(np.log1p(features[BIN_LABELS]))
    labels = KMeans(n_clusters=k, random_state=2026, n_init=50).fit_predict(x)
    tmp = features.copy()
    tmp["raw"] = labels
    means = tmp.groupby("raw")[BIN_LABELS].mean()
    means["overall"] = means[BIN_LABELS].mean(axis=1)
    order = means.sort_values("overall").index.tolist()
    mapping = {raw: i + 1 for i, raw in enumerate(order)}
    out = pd.DataFrame({"stay_id": features.index.astype(int), "trajectory_group": pd.Series(labels, index=features.index).map(mapping).astype(int).values})
    return out


def fit_logistic(df):
    covariates = [
        "age",
        "sofa",
        "charlson_comorbidity_index",
        "mbp_mean",
        "creatinine_max",
        "mechvent_24h",
        "vasoactive_24h",
    ]
    model_df = df[["hospital_expire_flag", "trajectory_group"] + covariates].dropna().copy()
    dummies = pd.get_dummies(model_df["trajectory_group"].astype(int), prefix="traj", drop_first=True, dtype=float)
    x = pd.concat([dummies, model_df[covariates].astype(float)], axis=1)
    x = sm.add_constant(x, has_constant="add")
    y = model_df["hospital_expire_flag"].astype(float)
    fit = sm.Logit(y, x).fit(disp=False)
    conf = fit.conf_int()
    terms = {}
    for term in fit.params.index:
        if not term.startswith("traj_"):
            continue
        terms[term] = {
            "or": float(math.exp(fit.params[term])),
            "ci95_low": float(math.exp(conf.loc[term, 0])),
            "ci95_high": float(math.exp(conf.loc[term, 1])),
            "p": float(fit.pvalues[term]),
        }
    return {"n_complete_cases": int(len(model_df)), "aic": float(fit.aic), "trajectory_terms": terms}


def summarize(df):
    rows = []
    for group, sub in df.groupby("trajectory_group"):
        deaths = int(sub["hospital_expire_flag"].sum())
        rows.append(
            {
                "trajectory_group": int(group),
                "n": int(len(sub)),
                "deaths": deaths,
                "mortality_pct": deaths / len(sub) * 100,
                "initial_lactate_median": float(sub["initial_lactate_24h"].median()),
                "last_lactate_median": float(sub["last_lactate_24h"].median()),
                "peak_lactate_median": float(sub["peak_lactate_24h"].median()),
            }
        )
    return rows


def run_scenario(long_df, analysis_df, k, min_count):
    features = build_features(long_df, min_lactate_count=min_count)
    assignments = assign_ordered_clusters(features, k)
    merged = analysis_df.merge(assignments, on="stay_id", how="inner")
    scenario_name = f"k{k}_min{min_count}"
    merged.to_csv(OUT_DIR / f"mimic_sensitivity_{scenario_name}.csv", index=False)
    return {
        "scenario": scenario_name,
        "k": k,
        "min_lactate_count": min_count,
        "n": int(len(merged)),
        "mortality": summarize(merged),
        "adjusted_logistic": fit_logistic(merged),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    long_df = pd.read_csv(LONG_PATH)
    analysis_df = pd.read_csv(ANALYSIS_PATH)
    scenarios = []
    for k in [2, 3, 5]:
        scenarios.append(run_scenario(long_df, analysis_df, k=k, min_count=2))
    scenarios.append(run_scenario(long_df, analysis_df, k=4, min_count=3))
    with open(OUT_DIR / "mimic_trajectory_sensitivity_results.json", "w", encoding="utf-8") as handle:
        json.dump({"scenarios": scenarios}, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"scenarios": scenarios}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
