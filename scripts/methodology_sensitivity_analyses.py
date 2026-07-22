from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prediction_utils import restrict_to_24h_survivors


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TABLES = ROOT / "manuscript_tables"

BIN_LABELS = ["lact_0_6h", "lact_6_12h", "lact_12_18h", "lact_18_24h"]
BASE_COVARS = [
    "age",
    "sofa",
    "charlson_comorbidity_index",
    "mbp_mean",
    "creatinine_max",
    "mechvent_24h",
    "vasoactive_24h",
]
BASE_COVARS_NO_TREAT = [
    "age",
    "sofa",
    "charlson_comorbidity_index",
    "mbp_mean",
    "creatinine_max",
]


def time_binned_wide(long_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = (
        long_df.groupby([id_col, "lactate_hour"], as_index=False)["lactate"]
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
    return (
        df.dropna(subset=["time_bin"])
        .groupby([id_col, "time_bin"], observed=True)["lactate"]
        .median()
        .unstack()
        .reindex(columns=BIN_LABELS)
    )


def fill_lactate_windows(wide: pd.DataFrame, fill_values: pd.Series | None = None) -> pd.DataFrame:
    filled = wide.apply(lambda row: row.astype(float).interpolate(limit_direction="both"), axis=1)
    if fill_values is None:
        fill_values = filled[BIN_LABELS].median()
    for col in BIN_LABELS:
        filled[col] = filled[col].fillna(float(fill_values[col]))
    return filled


def fill_status(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stay_id, row in wide.iterrows():
        observed = row.notna().to_numpy()
        for idx, col in enumerate(BIN_LABELS):
            if observed[idx]:
                status = "observed"
            else:
                left = observed[:idx].any()
                right = observed[idx + 1 :].any()
                if left and right:
                    status = "within_patient_linear_interpolation"
                elif left or right:
                    status = "within_patient_edge_extrapolation"
                else:
                    status = "cohort_median_fill"
            rows.append({"stay_id": int(stay_id), "time_window": col, "fill_status": status})
    return pd.DataFrame(rows)


def fit_ordered_kmeans(features: pd.DataFrame, k: int = 4, seed: int = 2026):
    scaler = StandardScaler()
    x = scaler.fit_transform(np.log1p(features[BIN_LABELS]))
    model = KMeans(n_clusters=k, random_state=seed, n_init=50).fit(x)
    tmp = features.copy()
    tmp["raw"] = model.labels_
    means = tmp.groupby("raw")[BIN_LABELS].mean()
    means["overall"] = means[BIN_LABELS].mean(axis=1)
    means["last"] = means["lact_18_24h"]
    ordered_raw = means.sort_values(["overall", "last"]).index.tolist()
    raw_to_ordered = {raw: rank + 1 for rank, raw in enumerate(ordered_raw)}
    ordered = pd.Series(model.labels_, index=features.index).map(raw_to_ordered).astype(int)
    return model, scaler, raw_to_ordered, ordered, x


def assign_nearest(features: pd.DataFrame, model: KMeans, scaler: StandardScaler, raw_to_ordered: dict[int, int]):
    x = scaler.transform(np.log1p(features[BIN_LABELS]))
    dist = ((x[:, None, :] - model.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
    raw = dist.argmin(axis=1)
    return pd.Series(raw, index=features.index).map(raw_to_ordered).astype(int)


def fit_logit_terms(
    df: pd.DataFrame,
    group_col: str,
    covars: list[str],
    outcome: str = "hospital_expire_flag",
    cluster_col: str | None = None,
):
    cols = [outcome, group_col] + covars
    if cluster_col is not None:
        cols.append(cluster_col)
    model_df = df[cols].dropna().copy()
    dummies = pd.get_dummies(model_df[group_col].astype(int), prefix="traj", drop_first=True, dtype=float)
    x = pd.concat([dummies, model_df[covars].astype(float)], axis=1)
    x = sm.add_constant(x, has_constant="add")
    y = model_df[outcome].astype(float)
    fit_kwargs = {"disp": False, "maxiter": 300}
    if cluster_col is not None:
        fit_kwargs.update(
            cov_type="cluster",
            cov_kwds={"groups": model_df[cluster_col]},
        )
    fit = sm.Logit(y, x).fit(**fit_kwargs)
    conf = fit.conf_int()
    rows = []
    for term in dummies.columns:
        rows.append(
            {
                "term": term,
                "n_complete": int(len(model_df)),
                "or": math.exp(fit.params[term]),
                "ci95_low": math.exp(conf.loc[term, 0]),
                "ci95_high": math.exp(conf.loc[term, 1]),
                "p_value": fit.pvalues[term],
            }
        )
    return rows


def mortality_by_group(df: pd.DataFrame, group_col: str = "trajectory_group") -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col):
        deaths = int(sub["hospital_expire_flag"].sum())
        rows.append(
            {
                "trajectory_group": int(group),
                "n": int(len(sub)),
                "deaths": deaths,
                "mortality_pct": deaths / len(sub) * 100,
                "initial_lactate_median": sub["initial_lactate_24h"].median(),
                "last_lactate_median": sub["last_lactate_24h"].median(),
                "peak_lactate_median": sub["peak_lactate_24h"].median(),
            }
        )
    return pd.DataFrame(rows)


def association_table(
    cohort_name: str,
    df: pd.DataFrame,
    covars: list[str],
    cluster_col: str | None = None,
) -> pd.DataFrame:
    mortality = mortality_by_group(df).set_index("trajectory_group")
    estimates = {
        row["term"]: row
        for row in fit_logit_terms(df, "trajectory_group", covars, cluster_col=cluster_col)
    }
    model_n = next(iter(estimates.values()))["n_complete"]
    rows = []
    for group in range(1, 5):
        mort = mortality.loc[group]
        estimate = estimates.get(f"traj_{group}")
        rows.append(
            {
                "cohort": cohort_name,
                "trajectory_group": group,
                "n": int(mort["n"]),
                "deaths": int(mort["deaths"]),
                "mortality_pct": mort["mortality_pct"],
                "adjusted_or": estimate["or"] if estimate else 1.0,
                "ci95_low": estimate["ci95_low"] if estimate else np.nan,
                "ci95_high": estimate["ci95_high"] if estimate else np.nan,
                "p_value": estimate["p_value"] if estimate else np.nan,
                "model_n": model_n,
            }
        )
    return pd.DataFrame(rows)


def primary_and_full_cohort_associations(
    mimic: pd.DataFrame,
    cohort: pd.DataFrame,
    eicu: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mimic_landmark = restrict_to_24h_survivors(mimic, OUT / "mimic_cs_lactate_24h_cohort.csv")
    eicu_landmark = eicu[eicu["hospitaldischargeoffset"].gt(24 * 60)].copy()
    eicu_covars = [
        "age",
        "apachescore",
        "meanbp",
        "creatinine",
        "vent",
        "vasoactive_24h",
    ]
    primary = pd.concat(
        [
            association_table("MIMIC-IV", mimic_landmark, BASE_COVARS),
            association_table("eICU-CRD", eicu_landmark, eicu_covars, "hospitalid"),
        ],
        ignore_index=True,
    )
    full = pd.concat(
        [
            association_table("MIMIC-IV", mimic, BASE_COVARS),
            association_table("eICU-CRD", eicu, eicu_covars, "hospitalid"),
        ],
        ignore_index=True,
    )
    return primary, full


def landmark_and_early_death_tables(mimic: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    base = cohort.copy()
    base["intime_dt"] = pd.to_datetime(base["intime"])
    base["deathtime_dt"] = pd.to_datetime(base["deathtime"], errors="coerce")
    base["dischtime_dt"] = pd.to_datetime(base["dischtime"], errors="coerce")
    base["death_hours_after_icu"] = (
        base["deathtime_dt"] - base["intime_dt"]
    ).dt.total_seconds() / 3600
    base["discharge_hours_after_icu"] = (
        base["dischtime_dt"] - base["intime_dt"]
    ).dt.total_seconds() / 3600
    base["death_within_24h"] = (
        (base["hospital_expire_flag"] == 1)
        & base["death_hours_after_icu"].gt(0)
        & base["death_hours_after_icu"].le(24)
    )
    base["at_risk_at_24h"] = base["discharge_hours_after_icu"].gt(24)
    base["lactate_category"] = np.select(
        [
            base["lactate_n_24h"].fillna(0).eq(0),
            base["lactate_n_24h"].fillna(0).eq(1),
            base["lactate_n_24h"].fillna(0).ge(2),
        ],
        ["0 lactates", "1 lactate", ">=2 lactates"],
        default="unknown",
    )

    severity = pd.read_csv(OUT / "mimic_cs_analysis_dataset.csv")[
        ["stay_id", "sofa", "sapsii", "oasis", "mbp_mean", "creatinine_max"]
    ]
    base = base.merge(severity, on="stay_id", how="left")

    rows = []
    for cat, sub in base.groupby("lactate_category"):
        rows.append(
            {
                "analysis": "lactate_measurement_availability",
                "group": cat,
                "n": int(len(sub)),
                "hospital_deaths": int(sub["hospital_expire_flag"].sum()),
                "hospital_mortality_pct": sub["hospital_expire_flag"].mean() * 100,
                "death_within_24h": int(sub["death_within_24h"].sum()),
                "death_within_24h_pct": sub["death_within_24h"].mean() * 100,
                "sofa_median": sub["sofa"].median(),
                "sapsii_median": sub["sapsii"].median(),
                "oasis_median": sub["oasis"].median(),
            }
        )

    trajectory_eligible = base[base["lactate_n_24h"].fillna(0).ge(2)].copy()
    for label, sub in [
        (
            "not_hospitalized_at_24h",
            trajectory_eligible[~trajectory_eligible["at_risk_at_24h"]],
        ),
        (
            "alive_and_hospitalized_at_24h",
            trajectory_eligible[trajectory_eligible["at_risk_at_24h"]],
        ),
    ]:
        rows.append(
            {
                "analysis": "landmark_risk_set_accounting",
                "group": label,
                "n": int(len(sub)),
                "hospital_deaths": int(sub["hospital_expire_flag"].sum()),
                "hospital_mortality_pct": sub["hospital_expire_flag"].mean() * 100,
                "death_within_24h": int(sub["death_within_24h"].sum()),
                "death_within_24h_pct": sub["death_within_24h"].mean() * 100,
            }
        )

    analysis = mimic.merge(
        base[["stay_id", "death_within_24h", "at_risk_at_24h"]],
        on="stay_id",
        how="left",
    )
    landmark = analysis[analysis["at_risk_at_24h"].eq(True)].copy()
    for _, row in mortality_by_group(landmark).iterrows():
        rows.append(
            {
                "analysis": "24h_landmark_alive_and_hospitalized",
                "group": f"trajectory_group_{int(row['trajectory_group'])}",
                "n": int(row["n"]),
                "hospital_deaths": int(row["deaths"]),
                "hospital_mortality_pct": row["mortality_pct"],
                "death_within_24h": 0,
                "death_within_24h_pct": 0,
                "sofa_median": np.nan,
                "sapsii_median": np.nan,
                "oasis_median": np.nan,
            }
        )
    for term in fit_logit_terms(landmark, "trajectory_group", BASE_COVARS):
        rows.append(
            {
                "analysis": "24h_landmark_adjusted_or",
                "group": term["term"],
                "n": term["n_complete"],
                "hospital_deaths": np.nan,
                "hospital_mortality_pct": np.nan,
                "death_within_24h": np.nan,
                "death_within_24h_pct": np.nan,
                "sofa_median": np.nan,
                "sapsii_median": np.nan,
                "oasis_median": np.nan,
                "adjusted_or": term["or"],
                "ci95_low": term["ci95_low"],
                "ci95_high": term["ci95_high"],
                "p_value": term["p_value"],
            }
        )

    eligible = mimic.copy()
    eligible["lactate_frequency_group"] = pd.cut(
        eligible["lactate_n_24h"],
        bins=[1, 2, 4, np.inf],
        labels=["2 lactates", "3-4 lactates", ">=5 lactates"],
        right=True,
    )
    for cat, sub in eligible.groupby("lactate_frequency_group", observed=True):
        rows.append(
            {
                "analysis": "eligible_lactate_frequency_and_severity",
                "group": str(cat),
                "n": int(len(sub)),
                "hospital_deaths": int(sub["hospital_expire_flag"].sum()),
                "hospital_mortality_pct": sub["hospital_expire_flag"].mean() * 100,
                "death_within_24h": np.nan,
                "death_within_24h_pct": np.nan,
                "sofa_median": sub["sofa"].median(),
                "sapsii_median": sub["sapsii"].median(),
                "oasis_median": sub["oasis"].median(),
            }
        )
    return pd.DataFrame(rows)


def imputation_audit(mimic: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    status = fill_status(wide)
    status = status.merge(mimic[["stay_id", "trajectory_group"]], on="stay_id", how="left")
    total = len(wide)
    rows = []
    for (window, fill), sub in status.groupby(["time_window", "fill_status"]):
        rows.append(
            {
                "cohort_or_group": "MIMIC-IV overall",
                "time_window": window,
                "fill_status": fill,
                "n_cells": int(len(sub)),
                "pct_of_patients": len(sub) / total * 100,
            }
        )
    for (group, fill), sub in status.groupby(["trajectory_group", "fill_status"]):
        denom = int((mimic["trajectory_group"] == group).sum()) * len(BIN_LABELS)
        rows.append(
            {
                "cohort_or_group": f"trajectory_group_{int(group)}",
                "time_window": "all_windows",
                "fill_status": fill,
                "n_cells": int(len(sub)),
                "pct_of_cells": len(sub) / denom * 100 if denom else np.nan,
            }
        )
    observed_windows = wide.notna().sum(axis=1).rename("observed_windows").reset_index()
    observed_windows = observed_windows.merge(mimic[["stay_id", "trajectory_group"]], on="stay_id")
    for (group, obs), sub in observed_windows.groupby(["trajectory_group", "observed_windows"]):
        denom = int((observed_windows["trajectory_group"] == group).sum())
        rows.append(
            {
                "cohort_or_group": f"trajectory_group_{int(group)}",
                "time_window": "observed_window_count",
                "fill_status": f"{int(obs)} observed windows",
                "n_cells": int(len(sub)),
                "pct_of_cells": len(sub) / denom * 100 if denom else np.nan,
            }
        )
    return pd.DataFrame(rows)


def cluster_stability_and_observed_window_sensitivity(mimic: pd.DataFrame, wide: pd.DataFrame):
    filled = fill_lactate_windows(wide)
    _, _, _, base_labels, x = fit_ordered_kmeans(filled, 4, seed=2026)
    rows = []
    rows.append(
        {
            "analysis": "primary_k4",
            "metric": "silhouette",
            "value": silhouette_score(x, base_labels),
            "n": len(filled),
        }
    )
    for k in range(2, 6):
        _, _, _, candidate_labels, candidate_x = fit_ordered_kmeans(filled, k, seed=2026)
        rows.append(
            {
                "analysis": "candidate_k",
                "metric": f"k_{k}_silhouette",
                "value": silhouette_score(candidate_x, candidate_labels),
                "n": len(filled),
            }
        )
    for seed in range(100):
        _, _, _, labels, _ = fit_ordered_kmeans(filled, 4, seed=seed)
        rows.append(
            {
                "analysis": "kmeans_seed_stability",
                "metric": "adjusted_rand_index_vs_primary",
                "value": adjusted_rand_score(base_labels, labels),
                "n": len(filled),
            }
        )

    rng = np.random.default_rng(2026)
    for _ in range(100):
        sample_idx = rng.integers(0, len(filled), len(filled))
        bootstrap_features = filled.iloc[sample_idx]
        model, scaler, mapping, _, _ = fit_ordered_kmeans(
            bootstrap_features, 4, seed=2026
        )
        bootstrap_labels = assign_nearest(filled, model, scaler, mapping)
        rows.append(
            {
                "analysis": "patient_bootstrap_stability",
                "metric": "adjusted_rand_index_vs_primary",
                "value": adjusted_rand_score(base_labels, bootstrap_labels),
                "n": len(filled),
            }
        )

    observed_n = wide.notna().sum(axis=1)
    subset_ids = observed_n[observed_n >= 3].index
    subset_features = filled.loc[subset_ids]
    _, _, _, subset_labels, subset_x = fit_ordered_kmeans(subset_features, 4, seed=2026)
    assign = pd.DataFrame(
        {"stay_id": subset_features.index.astype(int), "trajectory_group": subset_labels.values}
    )
    subset = mimic.drop(columns=["trajectory_group"]).merge(assign, on="stay_id", how="inner")
    for _, row in mortality_by_group(subset).iterrows():
        rows.append(
            {
                "analysis": "at_least_3_observed_windows",
                "metric": f"group_{int(row['trajectory_group'])}_mortality_pct",
                "value": row["mortality_pct"],
                "n": int(row["n"]),
            }
        )
    for term in fit_logit_terms(subset, "trajectory_group", BASE_COVARS):
        rows.append(
            {
                "analysis": "at_least_3_observed_windows_adjusted_or",
                "metric": term["term"],
                "value": term["or"],
                "ci95_low": term["ci95_low"],
                "ci95_high": term["ci95_high"],
                "p_value": term["p_value"],
                "n": term["n_complete"],
            }
        )
    rows.append(
        {
            "analysis": "at_least_3_observed_windows",
            "metric": "silhouette",
            "value": silhouette_score(subset_x, subset_labels),
            "n": len(subset_features),
        }
    )

    landmark = restrict_to_24h_survivors(mimic, OUT / "mimic_cs_lactate_24h_cohort.csv")
    landmark_features = filled.loc[landmark["stay_id"]]
    _, _, _, landmark_labels, landmark_x = fit_ordered_kmeans(
        landmark_features, 4, seed=2026
    )
    landmark_assign = pd.DataFrame(
        {
            "stay_id": landmark_features.index.astype(int),
            "trajectory_group": landmark_labels.values,
        }
    )
    landmark_refit = landmark.drop(columns=["trajectory_group"]).merge(
        landmark_assign, on="stay_id", how="inner"
    )
    rows.append(
        {
            "analysis": "landmark_only_k4_refit",
            "metric": "adjusted_rand_index_vs_full_cohort_labels",
            "value": adjusted_rand_score(
                mimic.set_index("stay_id").loc[landmark_features.index, "trajectory_group"],
                landmark_labels,
            ),
            "n": len(landmark_refit),
        }
    )
    rows.append(
        {
            "analysis": "landmark_only_k4_refit",
            "metric": "silhouette",
            "value": silhouette_score(landmark_x, landmark_labels),
            "n": len(landmark_refit),
        }
    )
    for _, row in mortality_by_group(landmark_refit).iterrows():
        rows.append(
            {
                "analysis": "landmark_only_k4_refit",
                "metric": f"group_{int(row['trajectory_group'])}_mortality_pct",
                "value": row["mortality_pct"],
                "n": int(row["n"]),
            }
        )
    for term in fit_logit_terms(landmark_refit, "trajectory_group", BASE_COVARS):
        rows.append(
            {
                "analysis": "landmark_only_k4_refit_adjusted_or",
                "metric": term["term"],
                "value": term["or"],
                "ci95_low": term["ci95_low"],
                "ci95_high": term["ci95_high"],
                "p_value": term["p_value"],
                "n": term["n_complete"],
            }
        )
    return pd.DataFrame(rows)


def preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [c for c in features if c in ["trajectory_group", "fold_trajectory_group", "gender"]]
    numeric = [c for c in features if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def foldwise_prediction(mimic: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    df = mimic.copy().reset_index(drop=True)
    y = df["hospital_expire_flag"].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    pred = pd.DataFrame({"stay_id": df["stay_id"], "hospital_expire_flag": y})
    pred["clinical_base"] = np.nan
    pred["base_plus_fold_trajectory"] = np.nan
    pred["base_plus_full_lactate_fold_trajectory"] = np.nan
    pred["fold_trajectory_group"] = np.nan

    id_to_row = {sid: i for i, sid in enumerate(df["stay_id"].to_numpy())}
    for train_idx, test_idx in cv.split(df, y):
        train_ids = df.iloc[train_idx]["stay_id"].to_numpy()
        test_ids = df.iloc[test_idx]["stay_id"].to_numpy()
        train_wide = wide.loc[train_ids]
        test_wide = wide.loc[test_ids]
        train_filled = fill_lactate_windows(train_wide)
        train_fill_values = train_filled[BIN_LABELS].median()
        test_filled = fill_lactate_windows(test_wide, train_fill_values)
        model, scaler, mapping, train_labels, _ = fit_ordered_kmeans(train_filled, 4, seed=2026)
        test_labels = assign_nearest(test_filled, model, scaler, mapping)

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        train_df["fold_trajectory_group"] = train_df["stay_id"].map(train_labels.to_dict()).astype(int)
        test_df["fold_trajectory_group"] = test_df["stay_id"].map(test_labels.to_dict()).astype(int)
        pred.loc[test_idx, "fold_trajectory_group"] = test_df["fold_trajectory_group"].to_numpy()

        specs = {
            "clinical_base": BASE_COVARS,
            "base_plus_fold_trajectory": BASE_COVARS + ["fold_trajectory_group"],
            "base_plus_full_lactate_fold_trajectory": BASE_COVARS
            + [
                "initial_lactate_24h",
                "last_lactate_24h",
                "peak_lactate_24h",
                "lactate_slope_24h",
                "lactate_clearance_24h",
                "fold_trajectory_group",
            ],
        }
        for name, features in specs.items():
            pipe = Pipeline(
                [
                    ("prep", preprocessor(features)),
                    ("model", LogisticRegression(max_iter=3000)),
                ]
            )
            pipe.fit(train_df[features], train_df["hospital_expire_flag"].astype(int))
            pred.loc[test_idx, name] = pipe.predict_proba(test_df[features])[:, 1]

    rows = []
    for model in ["clinical_base", "base_plus_fold_trajectory", "base_plus_full_lactate_fold_trajectory"]:
        rows.append(
            {
                "model": model,
                "n": int(len(pred)),
                "events": int(pred["hospital_expire_flag"].sum()),
                "auroc": roc_auc_score(y, pred[model]),
                "auprc": average_precision_score(y, pred[model]),
                "brier": brier_score_loss(y, pred[model]),
                "delta_auroc_vs_clinical_base": roc_auc_score(y, pred[model])
                - roc_auc_score(y, pred["clinical_base"]),
                "delta_auprc_vs_clinical_base": average_precision_score(y, pred[model])
                - average_precision_score(y, pred["clinical_base"]),
            }
        )
    pred.to_csv(OUT / "mimic_foldwise_trajectory_predictions.csv", index=False)
    return pd.DataFrame(rows)


def adjustment_models(mimic: pd.DataFrame, eicu: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mimic = restrict_to_24h_survivors(mimic, OUT / "mimic_cs_lactate_24h_cohort.csv")
    eicu = eicu[eicu["hospitaldischargeoffset"].gt(24 * 60)].copy()
    eicu_primary_covars = [
        "age", "apachescore", "meanbp", "creatinine", "vent", "vasoactive_24h"
    ]
    eicu_imputed = eicu.copy()
    imputation_indicators = []
    for variable in ["apachescore", "meanbp", "creatinine", "vent"]:
        if eicu_imputed[variable].isna().any():
            if variable in ["apachescore", "meanbp"]:
                indicator = f"{variable}_missing"
                eicu_imputed[indicator] = eicu_imputed[variable].isna().astype(int)
                imputation_indicators.append(indicator)
            eicu_imputed[variable] = eicu_imputed[variable].fillna(
                eicu_imputed[variable].median()
            )
    specs = [
        ("MIMIC-IV", mimic, "primary", BASE_COVARS),
        ("MIMIC-IV", mimic, "without_treatment_variables", BASE_COVARS_NO_TREAT),
        ("MIMIC-IV", mimic, "severity_only", ["age", "sofa", "charlson_comorbidity_index"]),
        (
            "eICU-CRD",
            eicu,
            "primary_apache_without_acute_physiology",
            eicu_primary_covars,
        ),
        ("eICU-CRD", eicu, "acute_physiology_without_apache", ["age", "acutephysiologyscore", "meanbp", "creatinine", "vent", "vasoactive_24h"]),
        ("eICU-CRD", eicu, "apache_plus_acute_physiology_sensitivity", ["age", "apachescore", "acutephysiologyscore", "meanbp", "creatinine", "vent", "vasoactive_24h"]),
        ("eICU-CRD", eicu, "without_treatment_variables", ["age", "apachescore", "meanbp", "creatinine"]),
        (
            "eICU-CRD",
            eicu_imputed,
            "median_imputation_with_missing_indicators",
            eicu_primary_covars + imputation_indicators,
        ),
    ]
    for cohort, df, model_name, covars in specs:
        cluster_col = "hospitalid" if cohort == "eICU-CRD" else None
        for term in fit_logit_terms(
            df, "trajectory_group", covars, cluster_col=cluster_col
        ):
            rows.append(
                {
                    "cohort": cohort,
                    "model": model_name,
                    "covariates": "; ".join(covars),
                    **term,
                }
            )
    return pd.DataFrame(rows)


def model_missingness_table(mimic: pd.DataFrame, eicu: pd.DataFrame) -> pd.DataFrame:
    mimic = restrict_to_24h_survivors(mimic, OUT / "mimic_cs_lactate_24h_cohort.csv")
    eicu = eicu[eicu["hospitaldischargeoffset"].gt(24 * 60)].copy()
    specs = [
        ("MIMIC-IV primary adjusted model", mimic, ["hospital_expire_flag", "trajectory_group"] + BASE_COVARS),
        ("MIMIC-IV no-treatment model", mimic, ["hospital_expire_flag", "trajectory_group"] + BASE_COVARS_NO_TREAT),
        (
            "eICU-CRD primary adjusted model",
            eicu,
            [
                "hospital_expire_flag",
                "trajectory_group",
                "age",
                "apachescore",
                "meanbp",
                "creatinine",
                "vent",
                "vasoactive_24h",
            ],
        ),
    ]
    rows = []
    for model_name, df, vars_ in specs:
        complete = df[vars_].dropna()
        rows.append(
            {
                "model": model_name,
                "variable": "complete_case_model_n",
                "n_total": len(df),
                "n_missing": len(df) - len(complete),
                "n_complete": len(complete),
                "missing_pct": (len(df) - len(complete)) / len(df) * 100,
            }
        )
        for var in vars_:
            rows.append(
                {
                    "model": model_name,
                    "variable": var,
                    "n_total": len(df),
                    "n_missing": int(df[var].isna().sum()),
                    "n_complete": int(df[var].notna().sum()),
                    "missing_pct": df[var].isna().mean() * 100,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    mimic = pd.read_csv(OUT / "mimic_analysis_dataset_with_trajectory.csv")
    eicu = pd.read_csv(OUT / "eicu_analysis_dataset_with_trajectory.csv")
    mimic_long = pd.read_csv(OUT / "mimic_cs_lactate_24h_long.csv")
    cohort = pd.read_csv(OUT / "mimic_cs_lactate_24h_cohort.csv")
    wide = time_binned_wide(mimic_long, "stay_id").loc[mimic["stay_id"]]

    primary_assoc, full_assoc = primary_and_full_cohort_associations(mimic, cohort, eicu)
    primary_assoc.to_csv(TABLES / "table_primary_24h_landmark_associations.csv", index=False)
    full_assoc.to_csv(TABLES / "table_supplementary_full_cohort_associations.csv", index=False)
    landmark_and_early_death_tables(mimic, cohort).to_csv(
        TABLES / "table_supplementary_24h_landmark_early_death.csv", index=False
    )
    imputation_audit(mimic, wide).to_csv(
        TABLES / "table_supplementary_lactate_imputation_audit.csv", index=False
    )
    cluster_stability_and_observed_window_sensitivity(mimic, wide).to_csv(
        TABLES / "table_supplementary_cluster_stability.csv", index=False
    )
    prediction_mimic = restrict_to_24h_survivors(
        mimic, OUT / "mimic_cs_lactate_24h_cohort.csv"
    )
    prediction_wide = wide.loc[prediction_mimic["stay_id"]]
    foldwise_prediction(prediction_mimic, prediction_wide).to_csv(
        TABLES / "table_supplementary_foldwise_trajectory_prediction.csv", index=False
    )
    adjustment_models(mimic, eicu).to_csv(
        TABLES / "table_supplementary_adjustment_models.csv", index=False
    )
    model_missingness_table(mimic, eicu).to_csv(
        TABLES / "table_supplementary_model_missingness_complete_cases.csv", index=False
    )
    print("methodology sensitivity tables written")


if __name__ == "__main__":
    main()
