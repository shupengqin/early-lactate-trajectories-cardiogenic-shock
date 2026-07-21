from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from methodology_sensitivity_analyses import (
    BASE_COVARS,
    BIN_LABELS,
    assign_nearest,
    fill_lactate_windows,
    fill_status,
    fit_logit_terms,
    fit_ordered_kmeans,
    mortality_by_group,
    preprocessor,
    time_binned_wide,
)
from prediction_utils import logistic_calibration_metrics, restrict_to_24h_survivors


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TABLES = ROOT / "manuscript_tables"
RANDOM_SEED = 2026


def _ordered_centroids(model, raw_to_ordered: dict[int, int]) -> np.ndarray:
    ordered = np.empty_like(model.cluster_centers_)
    for raw, group in raw_to_ordered.items():
        ordered[group - 1] = model.cluster_centers_[raw]
    return ordered


def _distance_profile(features, model, scaler, raw_to_ordered: dict[int, int]) -> pd.DataFrame:
    x = scaler.transform(np.log1p(features[BIN_LABELS]))
    distances_raw = np.sqrt(
        ((x[:, None, :] - model.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
    )
    raw_nearest = distances_raw.argmin(axis=1)
    sorted_distances = np.sort(distances_raw, axis=1)
    nearest = sorted_distances[:, 0]
    second = sorted_distances[:, 1]
    return pd.DataFrame(
        {
            "patient_id": features.index,
            "trajectory_group": [raw_to_ordered[int(raw)] for raw in raw_nearest],
            "nearest_centroid_distance": nearest,
            "second_nearest_centroid_distance": second,
            "distance_margin": second - nearest,
            "relative_margin": (second - nearest) / np.maximum(second, 1e-12),
            "nearest_to_second_ratio": nearest / np.maximum(second, 1e-12),
        }
    )


def cluster_assignment_clarity(
    mimic: pd.DataFrame,
    eicu: pd.DataFrame,
    mimic_wide: pd.DataFrame,
    eicu_wide: pd.DataFrame,
) -> pd.DataFrame:
    mimic_filled = fill_lactate_windows(mimic_wide)
    fill_values = mimic_filled[BIN_LABELS].median()
    model, scaler, mapping, labels, x = fit_ordered_kmeans(
        mimic_filled, 4, seed=RANDOM_SEED
    )
    centroids = _ordered_centroids(model, mapping)
    centroid_distances = squareform(pdist(centroids, metric="euclidean"))

    rows: list[dict] = []
    for first in range(4):
        for second in range(first + 1, 4):
            rows.append(
                {
                    "section": "centroid_pairwise_distance",
                    "cohort": "MIMIC-IV derivation",
                    "trajectory_group": f"{first + 1} vs {second + 1}",
                    "metric": "euclidean_distance_in_standardized_log_lactate_space",
                    "value": centroid_distances[first, second],
                    "n": len(mimic_filled),
                }
            )

    mimic_profile = _distance_profile(mimic_filled, model, scaler, mapping)
    mimic_profile["cohort"] = "MIMIC-IV derivation"
    eicu_filled = fill_lactate_windows(eicu_wide, fill_values)
    eicu_profile = _distance_profile(eicu_filled, model, scaler, mapping)
    eicu_profile["cohort"] = "eICU-CRD fixed-centroid validation"
    threshold = float(mimic_profile["nearest_centroid_distance"].quantile(0.95))

    existing = mimic.set_index("stay_id").loc[mimic_wide.index, "trajectory_group"]
    rows.extend(
        [
            {
                "section": "model_summary",
                "cohort": "MIMIC-IV derivation",
                "trajectory_group": "all",
                "metric": "silhouette_score",
                "value": silhouette_score(x, labels),
                "n": len(mimic_filled),
            },
            {
                "section": "model_summary",
                "cohort": "MIMIC-IV derivation",
                "trajectory_group": "all",
                "metric": "adjusted_rand_index_vs_saved_primary_labels",
                "value": adjusted_rand_score(existing, labels),
                "n": len(mimic_filled),
            },
            {
                "section": "model_summary",
                "cohort": "MIMIC-IV derivation",
                "trajectory_group": "all",
                "metric": "derivation_95th_percentile_nearest_distance_threshold",
                "value": threshold,
                "n": len(mimic_filled),
            },
        ]
    )

    for profile in [mimic_profile, eicu_profile]:
        profile["outside_derivation_95pct_distance"] = (
            profile["nearest_centroid_distance"] > threshold
        )
        for group_name, subset in [("all", profile)]:
            rows.extend(_distance_summary_rows(subset, profile["cohort"].iloc[0], group_name))
        for group, subset in profile.groupby("trajectory_group"):
            rows.extend(
                _distance_summary_rows(
                    subset, profile["cohort"].iloc[0], str(int(group))
                )
            )
    return pd.DataFrame(rows)


def _distance_summary_rows(
    subset: pd.DataFrame, cohort: str, group: str
) -> list[dict]:
    metrics = {
        "nearest_distance_median": subset["nearest_centroid_distance"].median(),
        "nearest_distance_q25": subset["nearest_centroid_distance"].quantile(0.25),
        "nearest_distance_q75": subset["nearest_centroid_distance"].quantile(0.75),
        "nearest_distance_p95": subset["nearest_centroid_distance"].quantile(0.95),
        "distance_margin_median": subset["distance_margin"].median(),
        "relative_margin_median": subset["relative_margin"].median(),
        "nearest_to_second_ratio_median": subset["nearest_to_second_ratio"].median(),
        "outside_derivation_95pct_distance_pct": 100
        * subset["outside_derivation_95pct_distance"].mean(),
    }
    return [
        {
            "section": "assignment_distance_summary",
            "cohort": cohort,
            "trajectory_group": group,
            "metric": metric,
            "value": value,
            "n": len(subset),
        }
        for metric, value in metrics.items()
    ]


def detailed_window_status(mimic: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    status = fill_status(wide).merge(
        mimic[["stay_id", "trajectory_group"]], on="stay_id", how="left", validate="many_to_one"
    )
    rows: list[dict] = []
    for window, subset in status.groupby("time_window", sort=False):
        denominator = subset["stay_id"].nunique()
        for fill_type, cells in subset.groupby("fill_status"):
            rows.append(
                {
                    "cohort_or_group": "MIMIC-IV overall",
                    "trajectory_group": "all",
                    "time_window": window,
                    "fill_status": fill_type,
                    "n_cells": len(cells),
                    "denominator_cells": denominator,
                    "pct_of_group_window_cells": 100 * len(cells) / denominator,
                }
            )
    for (group, window), subset in status.groupby(
        ["trajectory_group", "time_window"], sort=False
    ):
        denominator = subset["stay_id"].nunique()
        for fill_type, cells in subset.groupby("fill_status"):
            rows.append(
                {
                    "cohort_or_group": f"MIMIC-IV group {int(group)}",
                    "trajectory_group": int(group),
                    "time_window": window,
                    "fill_status": fill_type,
                    "n_cells": len(cells),
                    "denominator_cells": denominator,
                    "pct_of_group_window_cells": 100 * len(cells) / denominator,
                }
            )
    return pd.DataFrame(rows)


def complete_observed_window_sensitivity(
    mimic: pd.DataFrame, wide: pd.DataFrame
) -> pd.DataFrame:
    complete_wide = wide.loc[wide.notna().all(axis=1), BIN_LABELS]
    _, _, _, labels, x = fit_ordered_kmeans(complete_wide, 4, seed=RANDOM_SEED)
    assignment = pd.DataFrame(
        {"stay_id": complete_wide.index.astype(int), "trajectory_group": labels.values}
    )
    subset = mimic.drop(columns=["trajectory_group"]).merge(
        assignment, on="stay_id", how="inner", validate="one_to_one"
    )
    landmark = restrict_to_24h_survivors(
        subset, OUT / "mimic_cs_lactate_24h_cohort.csv"
    )
    rows: list[dict] = [
        {
            "analysis_population": "all four windows directly observed",
            "metric": "silhouette_score",
            "trajectory_group": "all",
            "n": len(subset),
            "value": silhouette_score(x, labels),
        },
        {
            "analysis_population": "all four windows directly observed",
            "metric": "adjusted_rand_index_vs_primary_group",
            "trajectory_group": "all",
            "n": len(subset),
            "value": adjusted_rand_score(
                mimic.set_index("stay_id").loc[complete_wide.index, "trajectory_group"],
                labels,
            ),
        },
    ]
    for population_name, frame in [
        ("full trajectory cohort", subset),
        ("24-hour landmark cohort", landmark),
    ]:
        associations = {
            term["term"]: term
            for term in fit_logit_terms(frame, "trajectory_group", BASE_COVARS)
        }
        model_n = next(iter(associations.values()))["n_complete"]
        for _, summary in mortality_by_group(frame).iterrows():
            group = int(summary["trajectory_group"])
            estimate = associations.get(f"traj_{group}")
            rows.append(
                {
                    "analysis_population": population_name,
                    "metric": "group_result",
                    "trajectory_group": group,
                    "n": int(summary["n"]),
                    "deaths": int(summary["deaths"]),
                    "mortality_pct": summary["mortality_pct"],
                    "adjusted_or": estimate["or"] if estimate else 1.0,
                    "ci95_low": estimate["ci95_low"] if estimate else np.nan,
                    "ci95_high": estimate["ci95_high"] if estimate else np.nan,
                    "p_value": estimate["p_value"] if estimate else np.nan,
                    "model_n": model_n,
                }
            )
    return pd.DataFrame(rows)


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    values = np.clip(values, 1e-12, None)
    return (vectors * values) @ vectors.T


def adjusted_absolute_effects(
    cohort: str,
    df: pd.DataFrame,
    covars: list[str],
    cluster_col: str | None = None,
    n_draws: int = 5000,
) -> pd.DataFrame:
    columns = ["hospital_expire_flag", "trajectory_group"] + covars
    if cluster_col:
        columns.append(cluster_col)
    model_df = df[columns].dropna().copy()
    dummies = pd.get_dummies(
        model_df["trajectory_group"].astype(int), prefix="traj", drop_first=True, dtype=float
    )
    for group in [2, 3, 4]:
        column = f"traj_{group}"
        if column not in dummies:
            dummies[column] = 0.0
    dummies = dummies[["traj_2", "traj_3", "traj_4"]]
    x = pd.concat([dummies, model_df[covars].astype(float)], axis=1)
    x = sm.add_constant(x, has_constant="add")
    y = model_df["hospital_expire_flag"].astype(float)
    fit_kwargs: dict = {"disp": False, "maxiter": 300}
    if cluster_col:
        fit_kwargs.update(
            cov_type="cluster", cov_kwds={"groups": model_df[cluster_col]}
        )
    logit = sm.Logit(y, x).fit(**fit_kwargs)

    poisson_kwargs: dict = {"cov_type": "HC3"}
    if cluster_col:
        poisson_kwargs = {
            "cov_type": "cluster",
            "cov_kwds": {"groups": model_df[cluster_col]},
        }
    poisson = sm.GLM(y, x, family=sm.families.Poisson()).fit(**poisson_kwargs)

    rng = np.random.default_rng(RANDOM_SEED + (1 if cluster_col else 0))
    covariance = _nearest_psd(np.asarray(logit.cov_params()))
    beta_draws = rng.multivariate_normal(
        np.asarray(logit.params), covariance, size=n_draws
    )
    standardized_risk: dict[int, float] = {}
    standardized_risk_draws: dict[int, np.ndarray] = {}
    for group in [1, 2, 3, 4]:
        x_group = x.copy()
        x_group[["traj_2", "traj_3", "traj_4"]] = 0.0
        if group > 1:
            x_group[f"traj_{group}"] = 1.0
        matrix = x_group.to_numpy(dtype=float)
        standardized_risk[group] = float(expit(matrix @ logit.params.to_numpy()).mean())
        standardized_risk_draws[group] = expit(matrix @ beta_draws.T).mean(axis=0)

    observed = (
        model_df.groupby("trajectory_group")["hospital_expire_flag"]
        .agg(n="size", deaths="sum", mortality_pct="mean")
        .reset_index()
    )
    observed["mortality_pct"] = 100 * observed["mortality_pct"]
    rows = []
    reference_draws = standardized_risk_draws[1]
    poisson_ci = poisson.conf_int()
    logit_ci = logit.conf_int()
    for _, summary in observed.iterrows():
        group = int(summary["trajectory_group"])
        draws = standardized_risk_draws[group]
        risk_difference_draws = 100 * (draws - reference_draws)
        if group == 1:
            adjusted_rr = adjusted_or = 1.0
            rr_low = rr_high = or_low = or_high = np.nan
            p_rr = p_or = np.nan
        else:
            term = f"traj_{group}"
            adjusted_rr = math.exp(poisson.params[term])
            rr_low = math.exp(poisson_ci.loc[term, 0])
            rr_high = math.exp(poisson_ci.loc[term, 1])
            p_rr = poisson.pvalues[term]
            adjusted_or = math.exp(logit.params[term])
            or_low = math.exp(logit_ci.loc[term, 0])
            or_high = math.exp(logit_ci.loc[term, 1])
            p_or = logit.pvalues[term]
        rows.append(
            {
                "cohort": cohort,
                "trajectory_group": group,
                "n": int(summary["n"]),
                "deaths": int(summary["deaths"]),
                "observed_mortality_pct": summary["mortality_pct"],
                "adjusted_mortality_pct": 100 * standardized_risk[group],
                "adjusted_mortality_ci95_low": 100 * np.percentile(draws, 2.5),
                "adjusted_mortality_ci95_high": 100 * np.percentile(draws, 97.5),
                "adjusted_risk_difference_per_100": 100
                * (standardized_risk[group] - standardized_risk[1]),
                "risk_difference_ci95_low": np.percentile(risk_difference_draws, 2.5),
                "risk_difference_ci95_high": np.percentile(risk_difference_draws, 97.5),
                "modified_poisson_rr": adjusted_rr,
                "rr_ci95_low": rr_low,
                "rr_ci95_high": rr_high,
                "rr_p_value": p_rr,
                "adjusted_or": adjusted_or,
                "or_ci95_low": or_low,
                "or_ci95_high": or_high,
                "or_p_value": p_or,
                "model_n": len(model_df),
                "covariates": "; ".join(covars),
            }
        )
    return pd.DataFrame(rows)


def _binary_smd(exposed: pd.Series, comparison: pd.Series) -> float:
    p1 = exposed.mean()
    p0 = comparison.mean()
    pooled = math.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    return (p1 - p0) / pooled if pooled else np.nan


def _continuous_smd(exposed: pd.Series, comparison: pd.Series) -> float:
    pooled = math.sqrt((exposed.var(ddof=1) + comparison.var(ddof=1)) / 2)
    return (exposed.mean() - comparison.mean()) / pooled if pooled else np.nan


def selection_comparison() -> pd.DataFrame:
    mimic = pd.read_csv(OUT / "mimic_cs_lactate_24h_cohort.csv")
    eicu = pd.read_csv(OUT / "eicu_cs_lactate_24h_cohort.csv")
    mimic["repeated_lactate_testing"] = mimic["lactate_n_24h"].fillna(0).ge(2)
    mimic["female"] = mimic["gender"].eq("F").astype(float)
    mimic["early_exit_24h"] = (
        (
            pd.to_datetime(mimic["dischtime"], errors="coerce")
            - pd.to_datetime(mimic["intime"], errors="coerce")
        ).dt.total_seconds()
        / 3600
    ).le(24)
    mimic["early_death_24h"] = (
        (
            pd.to_datetime(mimic["deathtime"], errors="coerce")
            - pd.to_datetime(mimic["intime"], errors="coerce")
        ).dt.total_seconds()
        / 3600
    ).between(0, 24, inclusive="right")

    eicu["repeated_lactate_testing"] = eicu["lactate_n_24h"].fillna(0).ge(2)
    eicu["female"] = eicu["gender"].eq("Female").astype(float)
    eicu["early_exit_24h"] = eicu["hospitaldischargeoffset"].le(24 * 60)
    eicu["early_death_24h"] = (
        eicu["early_exit_24h"] & eicu["hospital_expire_flag"].eq(1)
    )
    rows: list[dict] = []
    for cohort, frame in [("MIMIC-IV", mimic), ("eICU-CRD", eicu)]:
        exposed = frame[frame["repeated_lactate_testing"]]
        comparison = frame[~frame["repeated_lactate_testing"]]
        for group_label, subset in [
            ("at least two lactate measurements", exposed),
            ("fewer than two lactate measurements", comparison),
        ]:
            rows.append(
                {
                    "cohort": cohort,
                    "testing_group": group_label,
                    "metric": "population_n",
                    "n_nonmissing": len(subset),
                    "value": len(subset),
                }
            )
        for variable, label, variable_type in [
            ("age", "age_years", "continuous"),
            ("female", "female", "binary"),
            ("hospital_expire_flag", "in_hospital_mortality", "binary"),
            ("early_exit_24h", "death_or_discharge_within_24h", "binary"),
            ("early_death_24h", "death_within_24h", "binary"),
        ]:
            exposed_values = pd.to_numeric(exposed[variable], errors="coerce").dropna()
            comparison_values = pd.to_numeric(comparison[variable], errors="coerce").dropna()
            smd = (
                _continuous_smd(exposed_values, comparison_values)
                if variable_type == "continuous"
                else _binary_smd(exposed_values, comparison_values)
            )
            for group_label, values in [
                ("at least two lactate measurements", exposed_values),
                ("fewer than two lactate measurements", comparison_values),
            ]:
                row = {
                    "cohort": cohort,
                    "testing_group": group_label,
                    "metric": label,
                    "n_nonmissing": len(values),
                    "standardized_mean_difference_repeated_vs_not": smd,
                }
                if variable_type == "continuous":
                    row.update(
                        value=values.mean(),
                        median=values.median(),
                        q25=values.quantile(0.25),
                        q75=values.quantile(0.75),
                    )
                else:
                    row.update(
                        events=int(values.sum()),
                        value=100 * values.mean(),
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _net_benefit(y: np.ndarray, probability: np.ndarray, threshold: float) -> float:
    positive = probability >= threshold
    tp = np.sum(positive & (y == 1))
    fp = np.sum(positive & (y == 0))
    return tp / len(y) - fp / len(y) * threshold / (1 - threshold)


def repeated_foldwise_prediction(
    mimic: pd.DataFrame,
    wide: pd.DataFrame,
    n_repeats: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = mimic.reset_index(drop=True).copy()
    y = df["hospital_expire_flag"].astype(int).to_numpy()
    model_features = {
        "clinical_base": BASE_COVARS,
        "base_plus_initial_and_clearance": BASE_COVARS
        + ["initial_lactate_24h", "lactate_clearance_24h"],
        "base_plus_trajectory": BASE_COVARS + ["fold_trajectory_group"],
        "base_plus_full_lactate_dynamics": BASE_COVARS
        + [
            "initial_lactate_24h",
            "last_lactate_24h",
            "peak_lactate_24h",
            "lactate_slope_24h",
            "lactate_clearance_24h",
            "persistent_high_lactate_24h",
            "fold_trajectory_group",
        ],
    }
    metric_rows: list[dict] = []
    decision_rows: list[dict] = []
    thresholds = np.arange(0.10, 0.701, 0.05)
    for repeat in range(n_repeats):
        split_seed = RANDOM_SEED + repeat
        predictions = {model: np.full(len(df), np.nan) for model in model_features}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
        for fold, (train_idx, test_idx) in enumerate(cv.split(df, y), start=1):
            train_ids = df.iloc[train_idx]["stay_id"].to_numpy()
            test_ids = df.iloc[test_idx]["stay_id"].to_numpy()
            train_filled = fill_lactate_windows(wide.loc[train_ids])
            fill_values = train_filled[BIN_LABELS].median()
            test_filled = fill_lactate_windows(wide.loc[test_ids], fill_values)
            trajectory_model, scaler, mapping, train_labels, _ = fit_ordered_kmeans(
                train_filled, 4, seed=split_seed * 10 + fold
            )
            test_labels = assign_nearest(
                test_filled, trajectory_model, scaler, mapping
            )
            train = df.iloc[train_idx].copy()
            test = df.iloc[test_idx].copy()
            train["fold_trajectory_group"] = (
                train["stay_id"].map(train_labels.to_dict()).astype(int)
            )
            test["fold_trajectory_group"] = (
                test["stay_id"].map(test_labels.to_dict()).astype(int)
            )
            for model_name, features in model_features.items():
                pipeline = Pipeline(
                    [
                        ("prep", preprocessor(features)),
                        ("model", LogisticRegression(max_iter=3000)),
                    ]
                )
                pipeline.fit(train[features], y[train_idx])
                predictions[model_name][test_idx] = pipeline.predict_proba(
                    test[features]
                )[:, 1]
        repeat_metrics: dict[str, dict[str, float]] = {}
        for model_name, probability in predictions.items():
            calibration = logistic_calibration_metrics(y, probability)
            metrics = {
                "auroc": roc_auc_score(y, probability),
                "auprc": average_precision_score(y, probability),
                "brier": brier_score_loss(y, probability),
                **calibration,
            }
            repeat_metrics[model_name] = metrics
            metric_rows.append(
                {
                    "repeat": repeat + 1,
                    "split_seed": split_seed,
                    "model": model_name,
                    "n": len(df),
                    "events": int(y.sum()),
                    **metrics,
                }
            )
            for threshold in thresholds:
                decision_rows.append(
                    {
                        "repeat": repeat + 1,
                        "model": model_name,
                        "threshold": threshold,
                        "net_benefit": _net_benefit(y, probability, threshold),
                    }
                )
        metric_rows.append(
            {
                "repeat": repeat + 1,
                "split_seed": split_seed,
                "model": "trajectory_minus_initial_and_clearance",
                "n": len(df),
                "events": int(y.sum()),
                "auroc": repeat_metrics["base_plus_trajectory"]["auroc"]
                - repeat_metrics["base_plus_initial_and_clearance"]["auroc"],
                "auprc": repeat_metrics["base_plus_trajectory"]["auprc"]
                - repeat_metrics["base_plus_initial_and_clearance"]["auprc"],
                "brier": repeat_metrics["base_plus_trajectory"]["brier"]
                - repeat_metrics["base_plus_initial_and_clearance"]["brier"],
                "calibration_intercept": repeat_metrics["base_plus_trajectory"][
                    "calibration_intercept"
                ]
                - repeat_metrics["base_plus_initial_and_clearance"][
                    "calibration_intercept"
                ],
                "calibration_slope": repeat_metrics["base_plus_trajectory"][
                    "calibration_slope"
                ]
                - repeat_metrics["base_plus_initial_and_clearance"][
                    "calibration_slope"
                ],
            }
        )

    metrics = pd.DataFrame(metric_rows)
    summary_rows = []
    for model, subset in metrics.groupby("model", sort=False):
        for metric in [
            "auroc",
            "auprc",
            "brier",
            "calibration_intercept",
            "calibration_slope",
        ]:
            values = subset[metric].dropna()
            summary_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_repeats": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "ci95_low_across_splits": values.quantile(0.025),
                    "ci95_high_across_splits": values.quantile(0.975),
                    "minimum": values.min(),
                    "maximum": values.max(),
                }
            )
    decisions = pd.DataFrame(decision_rows)
    decision_summary = (
        decisions.groupby(["model", "threshold"])["net_benefit"]
        .agg(
            mean="mean",
            sd="std",
            ci95_low_across_splits=lambda value: value.quantile(0.025),
            ci95_high_across_splits=lambda value: value.quantile(0.975),
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )
    return metrics, pd.DataFrame(summary_rows), decision_summary


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    mimic = pd.read_csv(OUT / "mimic_analysis_dataset_with_trajectory.csv")
    eicu = pd.read_csv(OUT / "eicu_analysis_dataset_with_trajectory.csv")
    mimic_long = pd.read_csv(OUT / "mimic_cs_lactate_24h_long.csv")
    eicu_long = pd.read_csv(OUT / "eicu_cs_lactate_24h_long.csv")
    mimic_wide = time_binned_wide(mimic_long, "stay_id").loc[mimic["stay_id"]]
    eicu_wide = time_binned_wide(eicu_long, "patientunitstayid").loc[
        eicu["patientunitstayid"]
    ]

    detailed_window_status(mimic, mimic_wide).to_csv(
        TABLES / "table_supplementary_fill_status_by_group_window.csv", index=False
    )
    cluster_assignment_clarity(mimic, eicu, mimic_wide, eicu_wide).to_csv(
        TABLES / "table_supplementary_cluster_assignment_clarity.csv", index=False
    )
    complete_observed_window_sensitivity(mimic, mimic_wide).to_csv(
        TABLES / "table_supplementary_complete_observed_windows.csv", index=False
    )
    selection_comparison().to_csv(
        TABLES / "table_supplementary_selection_comparison.csv", index=False
    )

    mimic_landmark = restrict_to_24h_survivors(
        mimic, OUT / "mimic_cs_lactate_24h_cohort.csv"
    )
    eicu_landmark = eicu[eicu["hospitaldischargeoffset"].gt(24 * 60)].copy()
    absolute_effects = pd.concat(
        [
            adjusted_absolute_effects("MIMIC-IV", mimic_landmark, BASE_COVARS),
            adjusted_absolute_effects(
                "eICU-CRD",
                eicu_landmark,
                [
                    "age",
                    "apachescore",
                    "meanbp",
                    "creatinine",
                    "vent",
                    "vasoactive_24h",
                ],
                cluster_col="hospitalid",
            ),
        ],
        ignore_index=True,
    )
    absolute_effects.to_csv(
        TABLES / "table_primary_adjusted_absolute_effects.csv", index=False
    )

    prediction_wide = mimic_wide.loc[mimic_landmark["stay_id"]]
    repeated, repeated_summary, decision_summary = repeated_foldwise_prediction(
        mimic_landmark, prediction_wide
    )
    repeated.to_csv(
        TABLES / "table_supplementary_repeated_cv_per_repeat.csv", index=False
    )
    repeated_summary.to_csv(
        TABLES / "table_supplementary_repeated_cv_summary.csv", index=False
    )
    decision_summary.to_csv(
        TABLES / "table_supplementary_repeated_cv_decision_curve.csv", index=False
    )
    print("Reviewer hardening analyses completed.")


if __name__ == "__main__":
    main()
