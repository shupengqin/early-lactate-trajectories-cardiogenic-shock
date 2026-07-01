import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


LONG_PATH = Path("outputs/eicu_cs_lactate_24h_long.csv")
ANALYSIS_PATH = Path("outputs/eicu_cs_analysis_dataset.csv")
OUT_DIR = Path("outputs")
BIN_LABELS = ["lact_0_6h", "lact_6_12h", "lact_12_18h", "lact_18_24h"]


def make_features(long_df):
    df = long_df.copy()
    df["time_bin"] = pd.cut(
        df["lactate_hour"],
        bins=[0, 6, 12, 18, 24],
        labels=BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    wide = (
        df.dropna(subset=["time_bin"])
        .groupby(["patientunitstayid", "time_bin"], observed=True)["lactate"]
        .median()
        .unstack()
        .reindex(columns=BIN_LABELS)
    )
    filled = wide.apply(lambda row: row.astype(float).interpolate(limit_direction="both"), axis=1)
    for col in BIN_LABELS:
        filled[col] = filled[col].fillna(filled[col].median())
    return filled


def assign_clusters(features, k=4):
    x = StandardScaler().fit_transform(np.log1p(features[BIN_LABELS]))
    labels = KMeans(n_clusters=k, random_state=2026, n_init=50).fit_predict(x)
    tmp = features.copy()
    tmp["raw"] = labels
    means = tmp.groupby("raw")[BIN_LABELS].mean()
    means["overall"] = means[BIN_LABELS].mean(axis=1)
    order = means.sort_values("overall").index.tolist()
    mapping = {raw: i + 1 for i, raw in enumerate(order)}
    groups = pd.Series(labels, index=features.index).map(mapping).astype(int)
    assignments = pd.DataFrame(
        {
            "patientunitstayid": features.index.astype(int),
            "trajectory_group": groups.values,
        }
    )
    for col in BIN_LABELS:
        assignments[col] = features[col].values

    summary = []
    for group in sorted(assignments["trajectory_group"].unique()):
        sub = assignments[assignments["trajectory_group"] == group]
        means_group = sub[BIN_LABELS].mean()
        summary.append(
            {
                "trajectory_group": int(group),
                "n": int(len(sub)),
                "mean_0_6h": float(means_group["lact_0_6h"]),
                "mean_6_12h": float(means_group["lact_6_12h"]),
                "mean_12_18h": float(means_group["lact_12_18h"]),
                "mean_18_24h": float(means_group["lact_18_24h"]),
            }
        )
    return assignments, summary


def fit_logit(df, exposure_cols, covariates):
    cols = ["hospital_expire_flag"] + exposure_cols + covariates
    model_df = df[cols].dropna().copy()
    x = model_df[exposure_cols + covariates].astype(float)
    x = sm.add_constant(x, has_constant="add")
    y = model_df["hospital_expire_flag"].astype(float)
    fit = sm.Logit(y, x).fit(disp=False)
    conf = fit.conf_int()
    terms = {}
    for term in fit.params.index:
        terms[term] = {
            "or": float(math.exp(fit.params[term])),
            "ci95_low": float(math.exp(conf.loc[term, 0])),
            "ci95_high": float(math.exp(conf.loc[term, 1])),
            "p": float(fit.pvalues[term]),
        }
    return {"n_complete_cases": int(len(model_df)), "aic": float(fit.aic), "terms": terms}


def mortality_by_group(df, group_col):
    out = []
    for group, sub in df.groupby(group_col):
        deaths = int(sub["hospital_expire_flag"].sum())
        out.append(
            {
                group_col: int(group),
                "n": int(len(sub)),
                "deaths": deaths,
                "mortality_pct": float(deaths / len(sub) * 100),
                "initial_lactate_median": float(sub["initial_lactate_24h"].median()),
                "last_lactate_median": float(sub["last_lactate_24h"].median()),
                "peak_lactate_median": float(sub["peak_lactate_24h"].median()),
            }
        )
    return out


def main():
    OUT_DIR.mkdir(exist_ok=True)
    long_df = pd.read_csv(LONG_PATH)
    analysis_df = pd.read_csv(ANALYSIS_PATH)
    features = make_features(long_df)
    assignments, traj_summary = assign_clusters(features, k=4)
    merged = analysis_df.merge(assignments, on="patientunitstayid", how="inner")

    dummies = pd.get_dummies(
        merged["trajectory_group"].astype(int), prefix="traj", drop_first=True, dtype=float
    )
    merged_model = pd.concat([merged, dummies], axis=1)

    covariates = [
        "age",
        "apachescore",
        "acutephysiologyscore",
        "meanbp",
        "creatinine",
        "vent",
        "vasoactive_24h",
    ]
    # eICU has more missingness in some APACHE variables; keep a simpler fallback too.
    covariates_simple = ["age", "meanbp", "creatinine", "vent", "vasoactive_24h"]

    traj_terms = [c for c in merged_model.columns if c.startswith("traj_")]

    result = {
        "n": int(len(merged)),
        "deaths": int(merged["hospital_expire_flag"].sum()),
        "mortality_pct": float(merged["hospital_expire_flag"].mean() * 100),
        "trajectory_summary": traj_summary,
        "trajectory_mortality": mortality_by_group(merged, "trajectory_group"),
        "persistent_high_mortality": mortality_by_group(merged, "persistent_high_lactate_24h"),
        "adjusted_logistic_persistent_high_simple": fit_logit(
            merged, ["persistent_high_lactate_24h"], covariates_simple
        ),
        "adjusted_logistic_trajectory_simple": fit_logit(
            merged_model, traj_terms, covariates_simple
        ),
        "adjusted_logistic_persistent_high_apache": fit_logit(
            merged, ["persistent_high_lactate_24h"], covariates
        ),
        "adjusted_logistic_trajectory_apache": fit_logit(
            merged_model, traj_terms, covariates
        ),
        "outputs": {
            "assignments": "outputs/eicu_lactate_trajectory_assignments.csv",
            "analysis_with_trajectory": "outputs/eicu_analysis_dataset_with_trajectory.csv",
            "json": "outputs/eicu_external_validation_results.json",
        },
    }

    assignments.to_csv(OUT_DIR / "eicu_lactate_trajectory_assignments.csv", index=False)
    merged.to_csv(OUT_DIR / "eicu_analysis_dataset_with_trajectory.csv", index=False)
    with open(OUT_DIR / "eicu_external_validation_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
