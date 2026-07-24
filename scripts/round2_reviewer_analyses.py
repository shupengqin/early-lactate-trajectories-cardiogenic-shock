from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    brier_score_loss,
    calinski_harabasz_score,
    davies_bouldin_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from methodology_sensitivity_analyses import (
    BASE_COVARS,
    BIN_LABELS,
    fill_lactate_windows,
    fit_logit_terms,
    fit_ordered_kmeans,
    preprocessor,
    time_binned_wide,
)
from prediction_utils import logistic_calibration_metrics, restrict_to_24h_survivors
from reviewer_hardening_analyses import adjusted_absolute_effects


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TABLES = ROOT / "manuscript_tables"
RANDOM_SEED = 2026

EICU_COVARS = [
    "age",
    "apachescore",
    "meanbp",
    "creatinine",
    "vent",
    "vasoactive_24h",
]


def median_impute_eicu_covariates(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    imputed = frame.copy()
    missing_indicators: list[str] = []
    for variable in ["apachescore", "meanbp", "creatinine", "vent"]:
        if not imputed[variable].isna().any():
            continue
        if variable in ["apachescore", "meanbp"]:
            indicator = f"{variable}_missing"
            imputed[indicator] = imputed[variable].isna().astype(int)
            missing_indicators.append(indicator)
        imputed[variable] = imputed[variable].fillna(imputed[variable].median())
    covariates = EICU_COVARS + missing_indicators
    if imputed[covariates].isna().any().any():
        raise RuntimeError("eICU median imputation left missing model covariates")
    return imputed, covariates


def deduplicate_long(frame: pd.DataFrame, id_col: str) -> pd.DataFrame:
    return (
        frame.groupby([id_col, "lactate_hour"], as_index=False)["lactate"]
        .median()
        .sort_values([id_col, "lactate_hour"])
    )


def time_weighted_mean(group: pd.DataFrame) -> float:
    ordered = group.sort_values("lactate_hour")
    times = ordered["lactate_hour"].to_numpy(dtype=float)
    values = ordered["lactate"].to_numpy(dtype=float)
    if times[0] > 0:
        times = np.insert(times, 0, 0.0)
        values = np.insert(values, 0, values[0])
    if times[-1] < 24:
        times = np.append(times, 24.0)
        values = np.append(values, values[-1])
    return float(np.trapezoid(values, times) / 24.0)


def derive_simple_lactate_features(long: pd.DataFrame) -> pd.DataFrame:
    long = deduplicate_long(long, "stay_id")
    grouped = long.groupby("stay_id")
    features = grouped["lactate"].mean().rename("observed_mean_lactate_24h").to_frame()
    features["time_weighted_mean_lactate_24h"] = grouped.apply(
        time_weighted_mean, include_groups=False
    )
    final_window = (
        long[long["lactate_hour"].between(18, 24, inclusive="both")]
        .groupby("stay_id")["lactate"]
        .median()
    )
    features["observed_lactate_18_24h"] = final_window
    features["last_measurement_gap_hours"] = 24 - grouped["lactate_hour"].max()
    return features.reset_index()


def net_benefit(y: np.ndarray, probability: np.ndarray, threshold: float) -> float:
    predicted = probability >= threshold
    tp = np.sum(predicted & (y == 1))
    fp = np.sum(predicted & (y == 0))
    return float(tp / len(y) - fp / len(y) * threshold / (1 - threshold))


def simple_marker_prediction() -> tuple[pd.DataFrame, pd.DataFrame]:
    mimic = pd.read_csv(OUTPUTS / "mimic_analysis_dataset_with_trajectory.csv")
    mimic = restrict_to_24h_survivors(
        mimic, OUTPUTS / "mimic_cs_lactate_24h_cohort.csv"
    ).reset_index(drop=True)
    long = pd.read_csv(OUTPUTS / "mimic_cs_lactate_24h_long.csv")
    mimic = mimic.merge(
        derive_simple_lactate_features(long), on="stay_id", how="left", validate="one_to_one"
    )
    existing = pd.read_csv(OUTPUTS / "mimic_foldwise_marker_comparison_predictions.csv")
    existing = existing[
        [
            "stay_id",
            "clinical_base",
            "base_plus_initial_lactate",
            "base_plus_initial_and_clearance",
            "base_plus_persistent_hyperlactatemia",
            "base_plus_trajectory",
            "base_plus_full_lactate_dynamics",
        ]
    ]
    predictions = mimic[["stay_id", "hospital_expire_flag"]].merge(
        existing, on="stay_id", how="left", validate="one_to_one"
    )
    model_features = {
        "base_plus_last_lactate": BASE_COVARS + ["last_lactate_24h"],
        "base_plus_initial_and_last": BASE_COVARS
        + ["initial_lactate_24h", "last_lactate_24h"],
        "base_plus_observed_18_24h": BASE_COVARS + ["observed_lactate_18_24h"],
        "base_plus_observed_mean": BASE_COVARS + ["observed_mean_lactate_24h"],
        "base_plus_time_weighted_mean": BASE_COVARS
        + ["time_weighted_mean_lactate_24h"],
        "base_plus_peak_lactate": BASE_COVARS + ["peak_lactate_24h"],
        "base_plus_minimum_lactate": BASE_COVARS + ["min_lactate_24h"],
        "base_plus_lactate_slope": BASE_COVARS + ["lactate_slope_24h"],
    }
    for model in model_features:
        predictions[model] = np.nan

    y = mimic["hospital_expire_flag"].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for train_index, test_index in cv.split(mimic, y):
        train = mimic.iloc[train_index]
        test = mimic.iloc[test_index]
        for model, features in model_features.items():
            pipeline = Pipeline(
                [
                    ("prep", preprocessor(features)),
                    ("model", LogisticRegression(max_iter=3000)),
                ]
            )
            pipeline.fit(train[features], train["hospital_expire_flag"].astype(int))
            predictions.loc[test_index, model] = pipeline.predict_proba(test[features])[:, 1]

    predictions.to_csv(OUTPUTS / "mimic_round2_simple_marker_predictions.csv", index=False)
    labels = {
        "clinical_base": "Clinical base model",
        "base_plus_initial_lactate": "Clinical base + initial lactate",
        "base_plus_last_lactate": "Clinical base + last lactate",
        "base_plus_initial_and_last": "Clinical base + initial and last lactate",
        "base_plus_initial_and_clearance": "Clinical base + initial lactate and clearance",
        "base_plus_observed_18_24h": "Clinical base + observed 18-24 h lactate",
        "base_plus_observed_mean": "Clinical base + mean observed lactate",
        "base_plus_time_weighted_mean": "Clinical base + time-weighted mean lactate",
        "base_plus_peak_lactate": "Clinical base + peak lactate",
        "base_plus_minimum_lactate": "Clinical base + minimum lactate",
        "base_plus_lactate_slope": "Clinical base + lactate slope",
        "base_plus_persistent_hyperlactatemia": "Clinical base + persistent hyperlactatemia",
        "base_plus_trajectory": "Clinical base + trajectory group",
        "base_plus_full_lactate_dynamics": "Clinical base + full lactate dynamics",
    }
    base_auroc = roc_auc_score(y, predictions["clinical_base"])
    base_auprc = average_precision_score(y, predictions["clinical_base"])
    performance_rows = []
    for model, label in labels.items():
        probability = predictions[model].to_numpy(dtype=float)
        calibration = logistic_calibration_metrics(y, probability)
        performance_rows.append(
            {
                "model": model,
                "label": label,
                "auroc": roc_auc_score(y, probability),
                "auprc": average_precision_score(y, probability),
                "brier": brier_score_loss(y, probability),
                **calibration,
                "delta_auroc_vs_clinical_base": roc_auc_score(y, probability)
                - base_auroc,
                "delta_auprc_vs_clinical_base": average_precision_score(y, probability)
                - base_auprc,
                "net_benefit_threshold_0_30": net_benefit(y, probability, 0.30),
                "net_benefit_threshold_0_40": net_benefit(y, probability, 0.40),
                "net_benefit_threshold_0_50": net_benefit(y, probability, 0.50),
            }
        )
    performance = pd.DataFrame(performance_rows)

    references = [
        "base_plus_initial_lactate",
        "base_plus_last_lactate",
        "base_plus_initial_and_last",
        "base_plus_initial_and_clearance",
        "base_plus_observed_18_24h",
        "base_plus_observed_mean",
        "base_plus_time_weighted_mean",
        "base_plus_peak_lactate",
        "base_plus_minimum_lactate",
        "base_plus_lactate_slope",
        "base_plus_persistent_hyperlactatemia",
    ]
    rng = np.random.default_rng(RANDOM_SEED)
    comparison_rows = []
    for reference in references:
        auroc_deltas = []
        auprc_deltas = []
        for _ in range(500):
            index = rng.integers(0, len(y), len(y))
            if np.unique(y[index]).size < 2:
                continue
            auroc_deltas.append(
                roc_auc_score(y[index], predictions["base_plus_trajectory"].to_numpy()[index])
                - roc_auc_score(y[index], predictions[reference].to_numpy()[index])
            )
            auprc_deltas.append(
                average_precision_score(
                    y[index], predictions["base_plus_trajectory"].to_numpy()[index]
                )
                - average_precision_score(y[index], predictions[reference].to_numpy()[index])
            )
        comparison_rows.append(
            {
                "comparison": f"Trajectory group vs {labels[reference].removeprefix('Clinical base + ')}",
                "reference_model": reference,
                "n_bootstrap": len(auroc_deltas),
                "delta_auroc": roc_auc_score(y, predictions["base_plus_trajectory"])
                - roc_auc_score(y, predictions[reference]),
                "delta_auroc_ci95_low": np.percentile(auroc_deltas, 2.5),
                "delta_auroc_ci95_high": np.percentile(auroc_deltas, 97.5),
                "delta_auprc": average_precision_score(y, predictions["base_plus_trajectory"])
                - average_precision_score(y, predictions[reference]),
                "delta_auprc_ci95_low": np.percentile(auprc_deltas, 2.5),
                "delta_auprc_ci95_high": np.percentile(auprc_deltas, 97.5),
                "delta_net_benefit_0_30": net_benefit(
                    y, predictions["base_plus_trajectory"].to_numpy(), 0.30
                )
                - net_benefit(y, predictions[reference].to_numpy(), 0.30),
                "delta_net_benefit_0_40": net_benefit(
                    y, predictions["base_plus_trajectory"].to_numpy(), 0.40
                )
                - net_benefit(y, predictions[reference].to_numpy(), 0.40),
                "delta_net_benefit_0_50": net_benefit(
                    y, predictions["base_plus_trajectory"].to_numpy(), 0.50
                )
                - net_benefit(y, predictions[reference].to_numpy(), 0.50),
            }
        )
    return performance, pd.DataFrame(comparison_rows)


def _sensitivity_rows(
    scenario: str,
    mimic: pd.DataFrame,
    wide: pd.DataFrame,
    mask: pd.Series,
    no_edge_fill: bool,
) -> list[dict]:
    selected = wide.loc[mask].copy()
    if no_edge_fill:
        filled = selected.interpolate(axis=1, limit_area="inside")
        if filled.isna().any().any():
            raise RuntimeError(f"Unexpected edge missingness in {scenario}")
    else:
        filled = fill_lactate_windows(selected)
    _, _, _, labels, _ = fit_ordered_kmeans(filled, 4, seed=RANDOM_SEED)
    analysis = mimic[mimic["stay_id"].isin(selected.index)].copy()
    analysis["trajectory_group"] = analysis["stay_id"].map(labels.to_dict()).astype(int)
    analysis = restrict_to_24h_survivors(
        analysis, OUTPUTS / "mimic_cs_lactate_24h_cohort.csv"
    )
    estimates = {
        row["term"]: row
        for row in fit_logit_terms(analysis, "trajectory_group", BASE_COVARS)
    }
    model_n = next(iter(estimates.values()))["n_complete"]
    rows = []
    for group, subset in analysis.groupby("trajectory_group"):
        estimate = estimates.get(f"traj_{int(group)}")
        rows.append(
            {
                "scenario": scenario,
                "k": 4,
                "min_lactate_count": 2,
                "trajectory_group": int(group),
                "n": len(subset),
                "deaths": int(subset["hospital_expire_flag"].sum()),
                "mortality_pct": 100 * subset["hospital_expire_flag"].mean(),
                "initial_lactate_median": subset["initial_lactate_24h"].median(),
                "last_lactate_median": subset["last_lactate_24h"].median(),
                "peak_lactate_median": subset["peak_lactate_24h"].median(),
                "adjusted_or_vs_group1": estimate["or"] if estimate else "reference",
                "ci95": (
                    f"{estimate['ci95_low']:.2f}-{estimate['ci95_high']:.2f}"
                    if estimate
                    else "reference"
                ),
                "p_value": estimate["p_value"] if estimate else "reference",
                "model_n": model_n,
            }
        )
    return rows


def observation_sensitivity_and_gap_audit() -> None:
    mimic = pd.read_csv(OUTPUTS / "mimic_analysis_dataset_with_trajectory.csv")
    long_raw = pd.read_csv(OUTPUTS / "mimic_cs_lactate_24h_long.csv")
    duplicate_mask = long_raw.duplicated(["stay_id", "lactate_hour"], keep=False)
    duplicate_groups = long_raw.loc[duplicate_mask].groupby(["stay_id", "lactate_hour"])
    long = deduplicate_long(long_raw, "stay_id")
    wide = time_binned_wide(long, "stay_id").loc[mimic["stay_id"]]
    observed_count = wide.notna().sum(axis=1)
    final_observed = wide["lact_18_24h"].notna()
    first_observed = wide["lact_0_6h"].notna()
    rows = []
    rows.extend(
        _sensitivity_rows(
            "k4_round2_at_least3_observed_final_window_observed_landmark",
            mimic,
            wide,
            observed_count.ge(3) & final_observed,
            no_edge_fill=False,
        )
    )
    rows.extend(
        _sensitivity_rows(
            "k4_round2_no_edge_filling_first_and_final_observed_landmark",
            mimic,
            wide,
            observed_count.ge(3) & first_observed & final_observed,
            no_edge_fill=True,
        )
    )
    filled = fill_lactate_windows(wide)
    log_windows = np.log1p(filled[BIN_LABELS])
    centered = log_windows.sub(log_windows["lact_0_6h"], axis=0)[BIN_LABELS[1:]]
    centered_scaled = StandardScaler().fit_transform(centered)
    centered_model = KMeans(
        n_clusters=4, random_state=RANDOM_SEED, n_init=50
    ).fit(centered_scaled)
    centered_labels_raw = pd.Series(centered_model.labels_, index=centered.index)
    centered_final_change = centered.assign(raw_group=centered_labels_raw).groupby(
        "raw_group"
    )["lact_18_24h"].mean()
    centered_order = {
        raw_group: rank + 1
        for rank, raw_group in enumerate(centered_final_change.sort_values().index)
    }
    centered_labels = centered_labels_raw.map(centered_order).astype(int)
    centered_analysis = mimic.copy()
    centered_analysis["shape_group"] = centered_analysis["stay_id"].map(
        centered_labels.to_dict()
    ).astype(int)
    centered_analysis = restrict_to_24h_survivors(
        centered_analysis, OUTPUTS / "mimic_cs_lactate_24h_cohort.csv"
    )
    centered_analysis["log_initial_lactate_24h"] = np.log1p(
        centered_analysis["initial_lactate_24h"]
    )
    centered_estimates = {
        row["term"]: row
        for row in fit_logit_terms(
            centered_analysis,
            "shape_group",
            BASE_COVARS + ["log_initial_lactate_24h"],
        )
    }
    centered_model_n = next(iter(centered_estimates.values()))["n_complete"]
    for group, subset in centered_analysis.groupby("shape_group"):
        group = int(group)
        estimate = centered_estimates.get(f"traj_{group}")
        rows.append(
            {
                "scenario": "k4_initial_centered_log_shape_landmark",
                "k": 4,
                "min_lactate_count": 2,
                "trajectory_group": group,
                "n": len(subset),
                "deaths": int(subset["hospital_expire_flag"].sum()),
                "mortality_pct": 100 * subset["hospital_expire_flag"].mean(),
                "initial_lactate_median": subset["initial_lactate_24h"].median(),
                "last_lactate_median": subset["last_lactate_24h"].median(),
                "peak_lactate_median": subset["peak_lactate_24h"].median(),
                "adjusted_or_vs_group1": estimate["or"] if estimate else "reference",
                "ci95": (
                    f"{estimate['ci95_low']:.2f}-{estimate['ci95_high']:.2f}"
                    if estimate
                    else "reference"
                ),
                "p_value": estimate["p_value"] if estimate else "reference",
                "model_n": centered_model_n,
            }
        )
    centered_output = centered_analysis[
        ["stay_id", "shape_group", "hospital_expire_flag"]
    ].merge(
        centered.reset_index(),
        on="stay_id",
        how="left",
        validate="one_to_one",
    )
    centered_output.to_csv(
        OUTPUTS / "mimic_initial_centered_shape_assignments.csv", index=False
    )
    sensitivity_path = TABLES / "table_supplementary_sensitivity_trajectory.csv"
    sensitivity = pd.read_csv(sensitivity_path)
    sensitivity = sensitivity[
        ~sensitivity["scenario"].str.startswith(
            ("k4_round2_", "k4_initial_centered_"), na=False
        )
    ]
    sensitivity = pd.concat([sensitivity, pd.DataFrame(rows)], ignore_index=True)
    sensitivity.to_csv(sensitivity_path, index=False)

    features = derive_simple_lactate_features(long)
    gap = mimic[["stay_id", "trajectory_group", "hospital_expire_flag"]].merge(
        features[["stay_id", "last_measurement_gap_hours"]],
        on="stay_id",
        how="left",
        validate="one_to_one",
    )
    audit_rows = []
    for label, subset in [("MIMIC-IV overall", gap), *[
        (f"trajectory_group_{group}", gap[gap["trajectory_group"] == group])
        for group in range(1, 5)
    ]]:
        values = subset["last_measurement_gap_hours"]
        for metric, value in [
            ("last_measurement_gap_median_hours", values.median()),
            ("last_measurement_gap_q25_hours", values.quantile(0.25)),
            ("last_measurement_gap_q75_hours", values.quantile(0.75)),
            ("last_measurement_within_3h_pct", 100 * values.le(3).mean()),
            ("last_measurement_within_6h_pct", 100 * values.le(6).mean()),
            ("last_measurement_more_than_12h_before_landmark_pct", 100 * values.gt(12).mean()),
        ]:
            audit_rows.append(
                {
                    "cohort_or_group": label,
                    "time_window": "last_measurement_to_24h_landmark",
                    "fill_status": "timing_audit",
                    "n_cells": len(subset),
                    "pct_of_patients": np.nan,
                    "pct_of_cells": np.nan,
                    "metric": metric,
                    "value": value,
                }
            )
    for interval, subset in gap.groupby(
        pd.cut(
            gap["last_measurement_gap_hours"],
            bins=[-np.inf, 3, 6, 12, np.inf],
            labels=["0-3 h", ">3-6 h", ">6-12 h", ">12 h"],
        ),
        observed=True,
    ):
        audit_rows.append(
            {
                "cohort_or_group": "MIMIC-IV overall",
                "time_window": "last_measurement_gap_stratum",
                "fill_status": str(interval),
                "n_cells": len(subset),
                "pct_of_patients": 100 * len(subset) / len(gap),
                "pct_of_cells": np.nan,
                "metric": "in_hospital_mortality_pct",
                "value": 100 * subset["hospital_expire_flag"].mean(),
            }
        )
    audit_rows.extend(
        [
            {
                "cohort_or_group": "MIMIC-IV overall",
                "time_window": "duplicate_timestamp_audit",
                "fill_status": "same_timestamp_records",
                "n_cells": int(duplicate_mask.sum()),
                "pct_of_patients": np.nan,
                "pct_of_cells": np.nan,
                "metric": "duplicate_timestamps",
                "value": duplicate_groups.ngroups,
            },
            {
                "cohort_or_group": "MIMIC-IV overall",
                "time_window": "duplicate_timestamp_audit",
                "fill_status": "same_timestamp_discordant_values",
                "n_cells": int(duplicate_mask.sum()),
                "pct_of_patients": np.nan,
                "pct_of_cells": np.nan,
                "metric": "discordant_duplicate_timestamps",
                "value": sum(group["lactate"].nunique() > 1 for _, group in duplicate_groups),
            },
        ]
    )
    audit_path = TABLES / "table_supplementary_lactate_imputation_audit.csv"
    audit = pd.read_csv(audit_path)
    if "metric" not in audit:
        audit["metric"] = np.nan
        audit["value"] = np.nan
    audit = audit[audit["fill_status"] != "timing_audit"]
    audit = audit[audit["time_window"] != "last_measurement_gap_stratum"]
    audit = audit[audit["time_window"] != "duplicate_timestamp_audit"]
    audit = pd.concat([audit, pd.DataFrame(audit_rows)], ignore_index=True)
    audit.to_csv(audit_path, index=False)


def extended_cluster_metrics() -> None:
    mimic = pd.read_csv(OUTPUTS / "mimic_analysis_dataset_with_trajectory.csv")
    long = deduplicate_long(
        pd.read_csv(OUTPUTS / "mimic_cs_lactate_24h_long.csv"), "stay_id"
    )
    wide = time_binned_wide(long, "stay_id").loc[mimic["stay_id"]]
    filled = fill_lactate_windows(wide)
    x = StandardScaler().fit_transform(np.log1p(filled[BIN_LABELS]))
    rows = []
    for k in range(2, 6):
        labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=50).fit_predict(x)
        for metric, value in [
            (f"k_{k}_silhouette", silhouette_score(x, labels)),
            (f"k_{k}_calinski_harabasz", calinski_harabasz_score(x, labels)),
            (f"k_{k}_davies_bouldin", davies_bouldin_score(x, labels)),
        ]:
            rows.append(
                {
                    "analysis": "candidate_k_extended_metrics",
                    "metric": metric,
                    "value": value,
                    "n": len(x),
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "p_value": np.nan,
                }
            )
    path = TABLES / "table_supplementary_cluster_stability.csv"
    current = pd.read_csv(path)
    current = current[current["analysis"] != "candidate_k_extended_metrics"]
    pd.concat([current, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def _firth_penalized_loglik(beta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    linear = np.clip(x @ beta, -35, 35)
    probability = 1 / (1 + np.exp(-linear))
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    weight = probability * (1 - probability)
    information = x.T @ (weight[:, None] * x)
    sign, logdet = np.linalg.slogdet(information)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf
    loglik = np.sum(y * np.log(probability) + (1 - y) * np.log(1 - probability))
    return float(loglik + 0.5 * logdet)


def firth_group_estimates(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["hospital_expire_flag", "trajectory_group", *EICU_COVARS]
    model = frame[columns].dropna().copy()
    dummies = pd.get_dummies(
        model["trajectory_group"].astype(int), prefix="traj", drop_first=True, dtype=float
    )
    for group in [2, 3, 4]:
        if f"traj_{group}" not in dummies:
            dummies[f"traj_{group}"] = 0.0
    dummies = dummies[["traj_2", "traj_3", "traj_4"]]
    covariates = model[EICU_COVARS].astype(float).copy()
    covariates = (covariates - covariates.mean()) / covariates.std(ddof=0).replace(0, 1)
    design = pd.concat([dummies, covariates], axis=1)
    design.insert(0, "const", 1.0)
    x = design.to_numpy(dtype=float)
    y = model["hospital_expire_flag"].to_numpy(dtype=float)
    objective = lambda beta: -_firth_penalized_loglik(beta, x, y)
    result = minimize(objective, np.zeros(x.shape[1]), method="BFGS")
    if not result.success and np.linalg.norm(result.jac) > 1e-4:
        raise RuntimeError(f"Firth optimization failed: {result.message}")
    beta = result.x
    full_loglik = -result.fun

    def profile(index: int, fixed: float) -> float:
        keep = np.arange(x.shape[1]) != index
        start = beta[keep]

        def constrained(nuisance: np.ndarray) -> float:
            candidate = np.empty_like(beta)
            candidate[index] = fixed
            candidate[keep] = nuisance
            return -_firth_penalized_loglik(candidate, x, y)

        fit = minimize(constrained, start, method="BFGS")
        return -fit.fun

    rows = []
    cutoff = chi2.ppf(0.95, 1)
    for group in [2, 3, 4]:
        term = f"traj_{group}"
        index = design.columns.get_loc(term)
        estimate = beta[index]
        null_loglik = profile(index, 0.0)
        p_value = chi2.sf(max(0.0, 2 * (full_loglik - null_loglik)), 1)

        def root_function(value: float) -> float:
            return 2 * (full_loglik - profile(index, value)) - cutoff

        lower_bound = estimate - 0.5
        while root_function(lower_bound) < 0 and lower_bound > -12:
            lower_bound -= 0.5
        upper_bound = estimate + 0.5
        while root_function(upper_bound) < 0 and upper_bound < 12:
            upper_bound += 0.5
        lower = brentq(root_function, lower_bound, estimate - 1e-8)
        upper = brentq(root_function, estimate + 1e-8, upper_bound)
        rows.append(
            {
                "trajectory_group": group,
                "firth_or": math.exp(estimate),
                "firth_ci95_low": math.exp(lower),
                "firth_ci95_high": math.exp(upper),
                "firth_p_value": p_value,
                "firth_model_n": len(model),
            }
        )
    return pd.DataFrame(rows)


def external_transportability() -> None:
    fixed = pd.read_csv(OUTPUTS / "eicu_analysis_dataset_mimic_centroid_trajectory.csv")
    reclustered = pd.read_csv(OUTPUTS / "eicu_analysis_dataset_with_trajectory.csv")
    fixed_landmark = fixed[fixed["hospitaldischargeoffset"].gt(1440)].copy()
    reclustered_landmark = reclustered[
        reclustered["hospitaldischargeoffset"].gt(1440)
    ].copy()
    fixed_effects = adjusted_absolute_effects(
        "eICU-CRD fixed MIMIC-IV centroids",
        fixed_landmark,
        EICU_COVARS,
        cluster_col="hospitalid",
    )
    reclustered_effects = adjusted_absolute_effects(
        "eICU-CRD re-clustered sensitivity",
        reclustered_landmark,
        EICU_COVARS,
        cluster_col="hospitalid",
    )
    fixed_imputed, fixed_imputed_covars = median_impute_eicu_covariates(
        fixed_landmark
    )
    fixed_imputed_effects = adjusted_absolute_effects(
        "eICU-CRD fixed MIMIC-IV centroids, median-imputation sensitivity",
        fixed_imputed,
        fixed_imputed_covars,
        cluster_col="hospitalid",
    )
    fixed_firth = firth_group_estimates(fixed_landmark)
    reclustered_firth = firth_group_estimates(reclustered_landmark)

    no_firth = pd.DataFrame(
        {
            "trajectory_group": [2, 3, 4],
            "firth_or": [np.nan] * 3,
            "firth_ci95_low": [np.nan] * 3,
            "firth_ci95_high": [np.nan] * 3,
            "firth_p_value": [np.nan] * 3,
            "firth_model_n": [np.nan] * 3,
        }
    )

    combined = []
    for method, full_frame, effects, firth in [
        ("Fixed MIMIC-IV centroids (primary)", fixed_landmark, fixed_effects, fixed_firth),
        (
            "Fixed MIMIC-IV centroids (median-imputation sensitivity)",
            fixed_landmark,
            fixed_imputed_effects,
            no_firth,
        ),
        ("eICU re-clustering (sensitivity)", reclustered_landmark, reclustered_effects, reclustered_firth),
    ]:
        counts = (
            full_frame.groupby("trajectory_group")["hospital_expire_flag"]
            .agg(n_landmark="size", deaths="sum")
            .reset_index()
        )
        merged = counts.merge(effects, on="trajectory_group", suffixes=("_landmark", "_model"))
        merged = merged.merge(firth, on="trajectory_group", how="left")
        merged.insert(0, "assignment_method", method)
        combined.append(merged)
    associations = pd.concat(combined, ignore_index=True)
    associations.to_csv(
        TABLES
        / "table_eicu_mimic_centroid_24h_landmark_adjusted_associations_raw.csv",
        index=False,
    )
    associations_display = pd.DataFrame(
        {
            "Assignment method": associations["assignment_method"],
            "Group": associations["trajectory_group"].map(
                lambda value: f"Group {int(value)}"
            ),
            "N at landmark": associations["n_landmark"].astype(int),
            "Deaths": associations["deaths_landmark"].astype(int),
            "Model included, n": associations["n"].astype(int),
            "Adjusted mortality, % (95% CI)": associations.apply(
                lambda row: f"{row['adjusted_mortality_pct']:.1f} ({row['adjusted_mortality_ci95_low']:.1f}-{row['adjusted_mortality_ci95_high']:.1f})",
                axis=1,
            ),
            "Risk difference per 100 (95% CI)": associations.apply(
                lambda row: "Reference"
                if int(row["trajectory_group"]) == 1
                else f"{row['adjusted_risk_difference_per_100']:.1f} ({row['risk_difference_ci95_low']:.1f}-{row['risk_difference_ci95_high']:.1f})",
                axis=1,
            ),
            "Adjusted RR (95% CI)": associations.apply(
                lambda row: "Reference"
                if int(row["trajectory_group"]) == 1
                else f"{row['modified_poisson_rr']:.2f} ({row['rr_ci95_low']:.2f}-{row['rr_ci95_high']:.2f})",
                axis=1,
            ),
            "Regular OR (95% CI)": associations.apply(
                lambda row: "Reference"
                if int(row["trajectory_group"]) == 1
                else f"{row['adjusted_or']:.2f} ({row['or_ci95_low']:.2f}-{row['or_ci95_high']:.2f})",
                axis=1,
            ),
            "Firth OR (95% CI)": associations.apply(
                lambda row: (
                    "Reference"
                    if int(row["trajectory_group"]) == 1
                    else (
                        f"{row['firth_or']:.2f} ({row['firth_ci95_low']:.2f}-{row['firth_ci95_high']:.2f})"
                        if pd.notna(row["firth_or"])
                        else "Not estimated"
                    )
                ),
                axis=1,
            ),
        }
    )
    associations_display.to_csv(
        TABLES / "table_eicu_mimic_centroid_24h_landmark_adjusted_associations.csv",
        index=False,
    )

    primary = associations[
        associations["assignment_method"] == "Fixed MIMIC-IV centroids (primary)"
    ].copy()
    formatted = pd.DataFrame(
        {
            "Group": primary["trajectory_group"].map(lambda value: f"Group {int(value)}"),
            "N at landmark": primary["n_landmark"].astype(int),
            "Deaths": primary["deaths_landmark"].astype(int),
            "Model included, n": primary["n"].astype(int),
            "Excluded for missing covariates, n": (
                primary["n_landmark"] - primary["n"]
            ).astype(int),
            "Adjusted mortality, % (95% CI)": primary.apply(
                lambda row: f"{row['adjusted_mortality_pct']:.1f} ({row['adjusted_mortality_ci95_low']:.1f}-{row['adjusted_mortality_ci95_high']:.1f})",
                axis=1,
            ),
            "Risk difference per 100 (95% CI)": primary.apply(
                lambda row: "Reference"
                if int(row["trajectory_group"]) == 1
                else f"{row['adjusted_risk_difference_per_100']:.1f} ({row['risk_difference_ci95_low']:.1f}-{row['risk_difference_ci95_high']:.1f})",
                axis=1,
            ),
            "Adjusted RR (95% CI)": primary.apply(
                lambda row: "Reference"
                if int(row["trajectory_group"]) == 1
                else f"{row['modified_poisson_rr']:.2f} ({row['rr_ci95_low']:.2f}-{row['rr_ci95_high']:.2f})",
                axis=1,
            ),
        }
    )
    formatted.to_csv(
        TABLES / "table5_eicu_fixed_centroid_transportability_journal.csv", index=False
    )

    merged = reclustered[["patientunitstayid", "trajectory_group"]].merge(
        fixed[["patientunitstayid", "trajectory_group"]],
        on="patientunitstayid",
        suffixes=("_reclustered", "_fixed"),
        validate="one_to_one",
    )
    clarity_path = TABLES / "table_supplementary_cluster_assignment_clarity.csv"
    clarity = pd.read_csv(clarity_path)
    clarity = clarity[clarity["section"] != "external_assignment_concordance"]
    concordance_rows = [
        {
            "section": "external_assignment_concordance",
            "cohort": "eICU-CRD",
            "trajectory_group": "all",
            "metric": "adjusted_rand_index_reclustered_vs_fixed",
            "value": adjusted_rand_score(
                merged["trajectory_group_reclustered"], merged["trajectory_group_fixed"]
            ),
            "n": len(merged),
        },
        {
            "section": "external_assignment_concordance",
            "cohort": "eICU-CRD",
            "trajectory_group": "all",
            "metric": "exact_ordered_group_agreement_pct",
            "value": 100
            * (merged["trajectory_group_reclustered"] == merged["trajectory_group_fixed"]).mean(),
            "n": len(merged),
        },
    ]
    cross = pd.crosstab(
        merged["trajectory_group_reclustered"], merged["trajectory_group_fixed"]
    )
    for reclustered_group in cross.index:
        for fixed_group in cross.columns:
            concordance_rows.append(
                {
                    "section": "external_assignment_concordance",
                    "cohort": "eICU-CRD",
                    "trajectory_group": f"reclustered_{reclustered_group}_fixed_{fixed_group}",
                    "metric": "patient_count",
                    "value": int(cross.loc[reclustered_group, fixed_group]),
                    "n": len(merged),
                }
            )
    pd.concat([clarity, pd.DataFrame(concordance_rows)], ignore_index=True).to_csv(
        clarity_path, index=False
    )


def main() -> None:
    performance, comparisons = simple_marker_prediction()
    performance.to_csv(OUTPUTS / "round2_simple_marker_performance_raw.csv", index=False)
    comparisons.to_csv(OUTPUTS / "round2_simple_marker_comparisons_raw.csv", index=False)
    performance_display = pd.DataFrame(
        {
            "Model": performance["label"],
            "AUROC": performance["auroc"].map(lambda value: f"{value:.3f}"),
            "AUPRC": performance["auprc"].map(lambda value: f"{value:.3f}"),
            "Brier score": performance["brier"].map(lambda value: f"{value:.3f}"),
            "Delta AUROC vs base": performance["delta_auroc_vs_clinical_base"].map(
                lambda value: f"{value:.3f}"
            ),
            "Delta AUPRC vs base": performance["delta_auprc_vs_clinical_base"].map(
                lambda value: f"{value:.3f}"
            ),
            "Net benefit at 0.30": performance["net_benefit_threshold_0_30"].map(
                lambda value: f"{value:.3f}"
            ),
            "Net benefit at 0.40": performance["net_benefit_threshold_0_40"].map(
                lambda value: f"{value:.3f}"
            ),
            "Net benefit at 0.50": performance["net_benefit_threshold_0_50"].map(
                lambda value: f"{value:.3f}"
            ),
        }
    )
    comparison_display = pd.DataFrame(
        {
            "Comparison": comparisons["comparison"],
            "Delta AUROC (95% CI)": comparisons.apply(
                lambda row: f"{row['delta_auroc']:.4f} ({row['delta_auroc_ci95_low']:.4f} to {row['delta_auroc_ci95_high']:.4f})",
                axis=1,
            ),
            "Delta AUPRC (95% CI)": comparisons.apply(
                lambda row: f"{row['delta_auprc']:.4f} ({row['delta_auprc_ci95_low']:.4f} to {row['delta_auprc_ci95_high']:.4f})",
                axis=1,
            ),
            "Delta net benefit at 0.30": comparisons["delta_net_benefit_0_30"].map(
                lambda value: f"{value:.4f}"
            ),
            "Delta net benefit at 0.40": comparisons["delta_net_benefit_0_40"].map(
                lambda value: f"{value:.4f}"
            ),
            "Delta net benefit at 0.50": comparisons["delta_net_benefit_0_50"].map(
                lambda value: f"{value:.4f}"
            ),
        }
    )
    performance_display.to_csv(
        TABLES / "table_q2_lactate_marker_model_performance.csv", index=False
    )
    comparison_display.to_csv(
        TABLES / "table_q2_trajectory_increment_vs_simple_markers.csv", index=False
    )
    observation_sensitivity_and_gap_audit()
    extended_cluster_metrics()
    external_transportability()
    print("Round-2 reviewer analyses completed.")


if __name__ == "__main__":
    main()
