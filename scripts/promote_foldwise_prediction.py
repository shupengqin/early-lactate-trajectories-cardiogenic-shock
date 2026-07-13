from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from methodology_sensitivity_analyses import (
    BASE_COVARS,
    assign_nearest,
    fill_lactate_windows,
    fit_ordered_kmeans,
    preprocessor,
    time_binned_wide,
)
from prediction_utils import logistic_calibration_metrics, restrict_to_24h_survivors


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TABLES = ROOT / "manuscript_tables"


def bootstrap_deltas(predictions: pd.DataFrame, n_bootstrap: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    y = predictions["hospital_expire_flag"].to_numpy()
    comparisons = [
        ("base_plus_trajectory", "clinical_base"),
        ("base_plus_full_lactate", "clinical_base"),
        ("base_plus_full_lactate", "base_plus_trajectory"),
    ]
    rows = []
    for model, reference in comparisons:
        auroc_deltas = []
        auprc_deltas = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(y), len(y))
            if np.unique(y[idx]).size < 2:
                continue
            auroc_deltas.append(
                roc_auc_score(y[idx], predictions[model].to_numpy()[idx])
                - roc_auc_score(y[idx], predictions[reference].to_numpy()[idx])
            )
            auprc_deltas.append(
                average_precision_score(y[idx], predictions[model].to_numpy()[idx])
                - average_precision_score(y[idx], predictions[reference].to_numpy()[idx])
            )
        rows.append(
            {
                "comparison": f"{model} vs {reference}",
                "n_bootstrap": len(auroc_deltas),
                "delta_auroc": round(
                    roc_auc_score(y, predictions[model])
                    - roc_auc_score(y, predictions[reference]),
                    4,
                ),
                "delta_auroc_ci95": (
                    f"{np.percentile(auroc_deltas, 2.5):.4f}-"
                    f"{np.percentile(auroc_deltas, 97.5):.4f}"
                ),
                "delta_auprc": round(
                    average_precision_score(y, predictions[model])
                    - average_precision_score(y, predictions[reference]),
                    4,
                ),
                "delta_auprc_ci95": (
                    f"{np.percentile(auprc_deltas, 2.5):.4f}-"
                    f"{np.percentile(auprc_deltas, 97.5):.4f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def calibration_data(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = predictions["hospital_expire_flag"]
    for model in ["clinical_base", "base_plus_trajectory", "base_plus_full_lactate"]:
        frame = pd.DataFrame({"y": y, "p": predictions[model]})
        frame["bin"] = pd.qcut(frame["p"], q=10, duplicates="drop")
        curve = (
            frame.groupby("bin", observed=True)
            .agg(mean_predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
            .reset_index(drop=True)
        )
        curve["model"] = model
        rows.append(curve)
    return pd.concat(rows, ignore_index=True)


def decision_curve_data(predictions: pd.DataFrame) -> pd.DataFrame:
    y = predictions["hospital_expire_flag"].to_numpy()
    n = len(y)
    prevalence = y.mean()
    thresholds = np.arange(0.05, 0.81, 0.01)
    rows = []
    for model in ["clinical_base", "base_plus_trajectory", "base_plus_full_lactate"]:
        probability = predictions[model].to_numpy()
        for threshold in thresholds:
            positive = probability >= threshold
            tp = ((positive == 1) & (y == 1)).sum()
            fp = ((positive == 1) & (y == 0)).sum()
            rows.append(
                {
                    "threshold": threshold,
                    "net_benefit": tp / n - fp / n * threshold / (1 - threshold),
                    "treat_all": prevalence - (1 - prevalence) * threshold / (1 - threshold),
                    "treat_none": 0.0,
                    "model": model,
                }
            )
    return pd.DataFrame(rows)


def primary_predictions() -> pd.DataFrame:
    foldwise = pd.read_csv(OUTPUTS / "mimic_foldwise_trajectory_predictions.csv")
    conventional = pd.read_csv(OUTPUTS / "mimic_prediction_model_predictions.csv")
    conventional = conventional[
        [
            "stay_id",
            "logistic_base_plus_initial_lactate",
            "logistic_base_plus_lactate_clearance",
        ]
    ]
    frame = foldwise.merge(conventional, on="stay_id", how="left", validate="one_to_one")
    return frame.rename(
        columns={
            "base_plus_fold_trajectory": "base_plus_trajectory",
            "base_plus_full_lactate_fold_trajectory": "base_plus_full_lactate",
            "logistic_base_plus_initial_lactate": "base_plus_initial_lactate",
            "logistic_base_plus_lactate_clearance": "base_plus_lactate_clearance",
        }
    )


def performance_table(predictions: pd.DataFrame) -> pd.DataFrame:
    y = predictions["hospital_expire_flag"].to_numpy()
    specs = [
        ("Clinical base model", "clinical_base"),
        ("Clinical base + initial lactate", "base_plus_initial_lactate"),
        ("Clinical base + initial lactate + clearance", "base_plus_lactate_clearance"),
        ("Clinical base + trajectory group", "base_plus_trajectory"),
        ("Clinical base + full lactate dynamics", "base_plus_full_lactate"),
    ]
    base_auroc = roc_auc_score(y, predictions["clinical_base"])
    base_auprc = average_precision_score(y, predictions["clinical_base"])
    rows = []
    for label, model in specs:
        auroc = roc_auc_score(y, predictions[model])
        auprc = average_precision_score(y, predictions[model])
        calibration = logistic_calibration_metrics(y, predictions[model].to_numpy())
        rows.append(
            {
                "Model": label,
                "AUROC": round(auroc, 3),
                "AUPRC": round(auprc, 3),
                "Brier score": round(brier_score_loss(y, predictions[model]), 3),
                "Calibration intercept": round(calibration["calibration_intercept"], 3),
                "Calibration slope": round(calibration["calibration_slope"], 3),
                "Delta AUROC vs base": round(auroc - base_auroc, 3),
                "Delta AUPRC vs base": round(auprc - base_auprc, 3),
            }
        )
    return pd.DataFrame(rows)


def full_cohort_label_sensitivity() -> pd.DataFrame:
    source = pd.read_csv(OUTPUTS / "mimic_prediction_logistic_comparison.csv").set_index("model")
    mapping = [
        ("clinical_base", "logistic_clinical_base"),
        ("base_plus_global_trajectory", "logistic_base_plus_trajectory"),
        ("base_plus_full_lactate_global_trajectory", "logistic_base_plus_full_lactate"),
    ]
    rows = []
    for label, source_name in mapping:
        row = source.loc[source_name]
        rows.append(
            {
                "model": label,
                "n": int(row["n"]),
                "events": int(row["events"]),
                "auroc": row["auroc"],
                "auprc": row["auprc"],
                "brier": row["brier"],
                "delta_auroc_vs_clinical_base": row["delta_auroc_vs_base"],
                "delta_auprc_vs_clinical_base": row["delta_auprc_vs_base"],
            }
        )
    return pd.DataFrame(rows)


def foldwise_algorithm_comparison() -> pd.DataFrame:
    mimic = pd.read_csv(OUTPUTS / "mimic_analysis_dataset_with_trajectory.csv")
    mimic = restrict_to_24h_survivors(mimic, OUTPUTS / "mimic_cs_lactate_24h_cohort.csv")
    mimic = mimic.reset_index(drop=True)
    long = pd.read_csv(OUTPUTS / "mimic_cs_lactate_24h_long.csv")
    wide = time_binned_wide(long, "stay_id").loc[mimic["stay_id"]]
    y = mimic["hospital_expire_flag"].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    estimators = {
        "logistic_l2": LogisticRegression(max_iter=3000, solver="lbfgs"),
        "random_forest": RandomForestClassifier(
            n_estimators=700, min_samples_leaf=15, random_state=2026, n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=700, min_samples_leaf=15, random_state=2026, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.03, max_depth=2, random_state=2026
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.03,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=2026,
        ),
    }
    predictions = {name: np.full(len(mimic), np.nan) for name in estimators}
    features = BASE_COVARS + [
        "initial_lactate_24h",
        "last_lactate_24h",
        "peak_lactate_24h",
        "lactate_slope_24h",
        "lactate_clearance_24h",
        "fold_trajectory_group",
    ]
    for train_idx, test_idx in cv.split(mimic, y):
        train_ids = mimic.iloc[train_idx]["stay_id"].to_numpy()
        test_ids = mimic.iloc[test_idx]["stay_id"].to_numpy()
        train_filled = fill_lactate_windows(wide.loc[train_ids])
        fill_values = train_filled.median()
        test_filled = fill_lactate_windows(wide.loc[test_ids], fill_values)
        model, scaler, mapping, train_labels, _ = fit_ordered_kmeans(
            train_filled, 4, seed=2026
        )
        test_labels = assign_nearest(test_filled, model, scaler, mapping)
        train = mimic.iloc[train_idx].copy()
        test = mimic.iloc[test_idx].copy()
        train["fold_trajectory_group"] = train["stay_id"].map(train_labels.to_dict()).astype(int)
        test["fold_trajectory_group"] = test["stay_id"].map(test_labels.to_dict()).astype(int)
        for name, estimator in estimators.items():
            pipeline = Pipeline(
                [("prep", preprocessor(features)), ("model", clone(estimator))]
            )
            pipeline.fit(train[features], train["hospital_expire_flag"].astype(int))
            predictions[name][test_idx] = pipeline.predict_proba(test[features])[:, 1]

    output = pd.DataFrame(
        {"stay_id": mimic["stay_id"], "hospital_expire_flag": y, **predictions}
    )
    output.to_csv(OUTPUTS / "mimic_foldwise_extended_ml_predictions.csv", index=False)
    rows = []
    for name in estimators:
        row = {
            "model": name,
            "feature_set": "foldwise_full_lactate_dynamic",
            "n": len(y),
            "events": int(y.sum()),
            "auroc": roc_auc_score(y, output[name]),
            "auprc": average_precision_score(y, output[name]),
            "brier": brier_score_loss(y, output[name]),
        }
        row.update(logistic_calibration_metrics(y, output[name].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auroc", ascending=False)


def foldwise_marker_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    mimic = pd.read_csv(OUTPUTS / "mimic_analysis_dataset_with_trajectory.csv")
    mimic = restrict_to_24h_survivors(mimic, OUTPUTS / "mimic_cs_lactate_24h_cohort.csv")
    mimic = mimic.reset_index(drop=True)
    long = pd.read_csv(OUTPUTS / "mimic_cs_lactate_24h_long.csv")
    wide = time_binned_wide(long, "stay_id").loc[mimic["stay_id"]]
    y = mimic["hospital_expire_flag"].astype(int).to_numpy()
    existing = pd.read_csv(OUTPUTS / "mimic_q2_marker_comparison_predictions.csv")
    existing = existing[
        [
            "stay_id",
            "clinical_base",
            "base_plus_initial_lactate",
            "base_plus_initial_and_clearance",
            "base_plus_persistent_hyperlactatemia",
        ]
    ]
    predictions = mimic[["stay_id", "hospital_expire_flag"]].merge(
        existing, on="stay_id", how="left", validate="one_to_one"
    )
    trajectory_specs = {
        "base_plus_trajectory": BASE_COVARS + ["fold_trajectory_group"],
        "base_plus_persistent_and_trajectory": BASE_COVARS
        + ["persistent_high_lactate_24h", "fold_trajectory_group"],
        "base_plus_initial_clearance_and_trajectory": BASE_COVARS
        + ["initial_lactate_24h", "lactate_clearance_24h", "fold_trajectory_group"],
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
    for name in trajectory_specs:
        predictions[name] = np.nan
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    for train_idx, test_idx in cv.split(mimic, y):
        train_ids = mimic.iloc[train_idx]["stay_id"].to_numpy()
        test_ids = mimic.iloc[test_idx]["stay_id"].to_numpy()
        train_filled = fill_lactate_windows(wide.loc[train_ids])
        fill_values = train_filled.median()
        test_filled = fill_lactate_windows(wide.loc[test_ids], fill_values)
        model, scaler, mapping, train_labels, _ = fit_ordered_kmeans(
            train_filled, 4, seed=2026
        )
        test_labels = assign_nearest(test_filled, model, scaler, mapping)
        train = mimic.iloc[train_idx].copy()
        test = mimic.iloc[test_idx].copy()
        train["fold_trajectory_group"] = train["stay_id"].map(train_labels.to_dict()).astype(int)
        test["fold_trajectory_group"] = test["stay_id"].map(test_labels.to_dict()).astype(int)
        for name, features in trajectory_specs.items():
            pipeline = Pipeline(
                [
                    ("prep", preprocessor(features)),
                    ("model", LogisticRegression(max_iter=3000)),
                ]
            )
            pipeline.fit(train[features], train["hospital_expire_flag"].astype(int))
            predictions.loc[test_idx, name] = pipeline.predict_proba(test[features])[:, 1]

    predictions.to_csv(OUTPUTS / "mimic_foldwise_marker_comparison_predictions.csv", index=False)
    base_auroc = roc_auc_score(y, predictions["clinical_base"])
    base_auprc = average_precision_score(y, predictions["clinical_base"])
    performance_rows = []
    for model in [
        "clinical_base",
        "base_plus_initial_lactate",
        "base_plus_initial_and_clearance",
        "base_plus_persistent_hyperlactatemia",
        "base_plus_trajectory",
        "base_plus_persistent_and_trajectory",
        "base_plus_initial_clearance_and_trajectory",
        "base_plus_full_lactate_dynamics",
    ]:
        auroc = roc_auc_score(y, predictions[model])
        auprc = average_precision_score(y, predictions[model])
        performance_rows.append(
            {
                "model": model,
                "auroc": auroc,
                "auprc": auprc,
                "brier": brier_score_loss(y, predictions[model]),
                "delta_auroc_vs_clinical_base": auroc - base_auroc,
                "delta_auprc_vs_clinical_base": auprc - base_auprc,
            }
        )

    comparisons = [
        ("base_plus_trajectory", "base_plus_initial_and_clearance", "Trajectory vs initial lactate + clearance"),
        ("base_plus_trajectory", "base_plus_persistent_hyperlactatemia", "Trajectory vs persistent hyperlactatemia"),
        ("base_plus_persistent_and_trajectory", "base_plus_persistent_hyperlactatemia", "Trajectory added to persistent hyperlactatemia"),
        ("base_plus_initial_clearance_and_trajectory", "base_plus_initial_and_clearance", "Trajectory added to initial lactate + clearance"),
        ("base_plus_full_lactate_dynamics", "base_plus_initial_clearance_and_trajectory", "Full dynamics vs initial lactate + clearance + trajectory"),
    ]
    rng = np.random.default_rng(2026)
    comparison_rows = []
    for model, reference, label in comparisons:
        auroc_delta = []
        auprc_delta = []
        for _ in range(500):
            idx = rng.integers(0, len(y), len(y))
            if np.unique(y[idx]).size < 2:
                continue
            auroc_delta.append(
                roc_auc_score(y[idx], predictions[model].to_numpy()[idx])
                - roc_auc_score(y[idx], predictions[reference].to_numpy()[idx])
            )
            auprc_delta.append(
                average_precision_score(y[idx], predictions[model].to_numpy()[idx])
                - average_precision_score(y[idx], predictions[reference].to_numpy()[idx])
            )
        comparison_rows.append(
            {
                "comparison": label,
                "n_bootstrap": len(auroc_delta),
                "delta_auroc": roc_auc_score(y, predictions[model])
                - roc_auc_score(y, predictions[reference]),
                "delta_auroc_ci95": f"{np.percentile(auroc_delta, 2.5):.4f}-{np.percentile(auroc_delta, 97.5):.4f}",
                "delta_auprc": average_precision_score(y, predictions[model])
                - average_precision_score(y, predictions[reference]),
                "delta_auprc_ci95": f"{np.percentile(auprc_delta, 2.5):.4f}-{np.percentile(auprc_delta, 97.5):.4f}",
            }
        )
    return pd.DataFrame(performance_rows), pd.DataFrame(comparison_rows)


def main() -> None:
    predictions = primary_predictions()
    predictions.to_csv(OUTPUTS / "mimic_primary_foldwise_predictions.csv", index=False)
    performance = performance_table(predictions)
    performance.to_csv(TABLES / "table4_prediction_performance_journal.csv", index=False)
    performance.to_csv(TABLES / "table_q2_prediction_performance.csv", index=False)
    bootstrap_deltas(predictions).to_csv(
        TABLES / "table_q2_bootstrap_prediction_deltas.csv", index=False
    )
    calibration_data(predictions).to_csv(
        TABLES / "table_q2_calibration_curve_data.csv", index=False
    )
    decision_curve_data(predictions).to_csv(
        TABLES / "table_q2_decision_curve_data.csv", index=False
    )
    full_cohort_label_sensitivity().to_csv(
        TABLES / "table_supplementary_global_label_prediction.csv", index=False
    )
    foldwise_algorithm_comparison().to_csv(
        TABLES / "table_extended_ml_performance.csv", index=False
    )
    marker_performance, marker_comparisons = foldwise_marker_comparison()
    marker_performance.to_csv(
        TABLES / "table_q2_lactate_marker_model_performance.csv", index=False
    )
    marker_comparisons.to_csv(
        TABLES / "table_q2_trajectory_increment_vs_simple_markers.csv", index=False
    )
    print("promoted foldwise trajectory pipeline to primary prediction analysis")


if __name__ == "__main__":
    main()
