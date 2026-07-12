from __future__ import annotations

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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prediction_utils import restrict_to_24h_survivors


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TABLES = ROOT / "manuscript_tables"
FIGURES = ROOT / "manuscript_figures"

BIN_LABELS = ["lact_0_6h", "lact_6_12h", "lact_12_18h", "lact_18_24h"]
BIN_MIDPOINTS = np.array([3.0, 9.0, 15.0, 21.0])

BASE_FEATURES = [
    "age",
    "sofa",
    "charlson_comorbidity_index",
    "mbp_mean",
    "creatinine_max",
    "mechvent_24h",
    "vasoactive_24h",
]
FEATURE_SETS = {
    "clinical_base": BASE_FEATURES,
    "base_plus_trajectory": BASE_FEATURES + ["trajectory_group"],
    "base_plus_full_lactate": BASE_FEATURES
    + [
        "initial_lactate_24h",
        "last_lactate_24h",
        "peak_lactate_24h",
        "lactate_slope_24h",
        "lactate_clearance_24h",
        "trajectory_group",
    ],
}


def make_mimic_features(long_df: pd.DataFrame) -> pd.DataFrame:
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
        .groupby(["stay_id", "time_bin"], observed=True)["lactate"]
        .median()
        .unstack()
        .reindex(columns=BIN_LABELS)
    )
    filled = wide.apply(lambda row: row.astype(float).interpolate(limit_direction="both"), axis=1)
    for col in BIN_LABELS:
        filled[col] = filled[col].fillna(filled[col].median())
    return filled


def make_eicu_features(long_df: pd.DataFrame, fill_values: pd.Series) -> pd.DataFrame:
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
        filled[col] = filled[col].fillna(float(fill_values[col]))
    return filled


def fit_mimic_centroid_model(features: pd.DataFrame) -> dict:
    scaler = StandardScaler()
    x = scaler.fit_transform(np.log1p(features[BIN_LABELS]))
    model = KMeans(n_clusters=4, random_state=2026, n_init=50).fit(x)
    tmp = features.copy()
    tmp["raw_cluster"] = model.labels_
    raw_means = tmp.groupby("raw_cluster")[BIN_LABELS].mean()
    raw_means["overall"] = raw_means[BIN_LABELS].mean(axis=1)
    raw_means["last"] = raw_means["lact_18_24h"]
    order = raw_means.sort_values(["overall", "last"]).index.tolist()
    raw_to_ordered = {int(raw): int(i + 1) for i, raw in enumerate(order)}
    ordered_to_raw = {v: k for k, v in raw_to_ordered.items()}
    return {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "centers": model.cluster_centers_.tolist(),
        "raw_to_ordered": raw_to_ordered,
        "ordered_to_raw": ordered_to_raw,
        "fill_values": features[BIN_LABELS].median().to_dict(),
        "raw_means": raw_means.reset_index().to_dict(orient="records"),
    }


def assign_by_mimic_centroids(features: pd.DataFrame, spec: dict, id_col: str) -> pd.DataFrame:
    mean = np.array(spec["scaler_mean"])
    scale = np.array(spec["scaler_scale"])
    centers = np.array(spec["centers"])
    x = (np.log1p(features[BIN_LABELS]).to_numpy() - mean) / scale
    distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    raw = distances.argmin(axis=1)
    groups = pd.Series(raw).map({int(k): int(v) for k, v in spec["raw_to_ordered"].items()})
    out = pd.DataFrame({id_col: features.index.astype(int), "trajectory_group_mimic_centroid": groups.values})
    for col in BIN_LABELS:
        out[col] = features[col].values
    return out


def fit_logit_or(
    df: pd.DataFrame,
    group_col: str,
    covariates: list[str],
    cluster_col: str | None = None,
) -> dict:
    cols = ["hospital_expire_flag", group_col] + covariates
    if cluster_col is not None:
        cols.append(cluster_col)
    model_df = df[cols].dropna().copy()
    dummies = pd.get_dummies(model_df[group_col].astype(int), prefix="traj", drop_first=True, dtype=float)
    x = pd.concat([dummies, model_df[covariates].astype(float)], axis=1)
    x = sm.add_constant(x, has_constant="add")
    y = model_df["hospital_expire_flag"].astype(float)
    fit_kwargs = {"disp": False}
    if cluster_col is not None:
        fit_kwargs.update(
            cov_type="cluster",
            cov_kwds={"groups": model_df[cluster_col]},
        )
    fit = sm.Logit(y, x).fit(**fit_kwargs)
    conf = fit.conf_int()
    terms = {}
    for term in dummies.columns:
        terms[term] = {
            "or": float(math.exp(fit.params[term])),
            "ci95_low": float(math.exp(conf.loc[term, 0])),
            "ci95_high": float(math.exp(conf.loc[term, 1])),
            "p": float(fit.pvalues[term]),
        }
    return {"n_complete_cases": int(len(model_df)), "aic": float(fit.aic), "terms": terms}


def mortality_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col):
        deaths = int(sub["hospital_expire_flag"].sum())
        rows.append(
            {
                "trajectory_group": int(group),
                "n": int(len(sub)),
                "deaths": deaths,
                "mortality_pct": round(deaths / len(sub) * 100, 2),
                "initial_lactate_median": round(float(sub["initial_lactate_24h"].median()), 2),
                "last_lactate_median": round(float(sub["last_lactate_24h"].median()), 2),
                "peak_lactate_median": round(float(sub["peak_lactate_24h"].median()), 2),
            }
        )
    return pd.DataFrame(rows)


def preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [c for c in features if c == "trajectory_group"]
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


def cv_predictions(df: pd.DataFrame) -> pd.DataFrame:
    y = df["hospital_expire_flag"].astype(int).to_numpy()
    preds = pd.DataFrame({"stay_id": df["stay_id"], "hospital_expire_flag": y})
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    for name, features in FEATURE_SETS.items():
        pipe = Pipeline(
            [
                ("prep", preprocessor(features)),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        )
        preds[name] = cross_val_predict(pipe, df[features], y, cv=cv, method="predict_proba")[:, 1]
    return preds


def bootstrap_deltas(preds: pd.DataFrame, n_boot: int = 500, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = preds["hospital_expire_flag"].to_numpy()
    rows = []
    comparisons = [
        ("base_plus_trajectory", "clinical_base"),
        ("base_plus_full_lactate", "clinical_base"),
        ("base_plus_full_lactate", "base_plus_trajectory"),
    ]
    for model, reference in comparisons:
        auroc_deltas = []
        auprc_deltas = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(preds), len(preds))
            if len(np.unique(y[idx])) < 2:
                continue
            auroc_deltas.append(
                roc_auc_score(y[idx], preds[model].to_numpy()[idx])
                - roc_auc_score(y[idx], preds[reference].to_numpy()[idx])
            )
            auprc_deltas.append(
                average_precision_score(y[idx], preds[model].to_numpy()[idx])
                - average_precision_score(y[idx], preds[reference].to_numpy()[idx])
            )
        rows.append(
            {
                "comparison": f"{model} vs {reference}",
                "n_bootstrap": len(auroc_deltas),
                "delta_auroc": round(
                    roc_auc_score(y, preds[model]) - roc_auc_score(y, preds[reference]), 4
                ),
                "delta_auroc_ci95": f"{np.percentile(auroc_deltas, 2.5):.4f}-{np.percentile(auroc_deltas, 97.5):.4f}",
                "delta_auprc": round(
                    average_precision_score(y, preds[model])
                    - average_precision_score(y, preds[reference]),
                    4,
                ),
                "delta_auprc_ci95": f"{np.percentile(auprc_deltas, 2.5):.4f}-{np.percentile(auprc_deltas, 97.5):.4f}",
            }
        )
    return pd.DataFrame(rows)


def calibration_curve_data(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    out = (
        df.groupby("bin", observed=True)
        .agg(mean_predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
    )
    return out


def dca_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    n = len(y)
    prevalence = y.mean()
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        nb = tp / n - fp / n * (t / (1 - t))
        treat_all = prevalence - (1 - prevalence) * (t / (1 - t))
        rows.append({"threshold": t, "net_benefit": nb, "treat_all": treat_all, "treat_none": 0.0})
    return pd.DataFrame(rows)


def plot_calibration(preds: pd.DataFrame, path: Path) -> pd.DataFrame:
    y = preds["hospital_expire_flag"].to_numpy()
    rows = []
    plt.figure(figsize=(5.2, 4.6), dpi=220)
    for model, label in [
        ("clinical_base", "Clinical base"),
        ("base_plus_trajectory", "Base + trajectory"),
        ("base_plus_full_lactate", "Base + full lactate dynamics"),
    ]:
        curve = calibration_curve_data(y, preds[model].to_numpy(), bins=10)
        curve["model"] = model
        rows.append(curve)
        plt.plot(curve["mean_predicted"], curve["observed"], marker="o", linewidth=1.8, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Predicted mortality risk")
    plt.ylabel("Observed mortality")
    plt.title("Calibration of mortality prediction models")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return pd.concat(rows, ignore_index=True)


def plot_dca(preds: pd.DataFrame, path: Path) -> pd.DataFrame:
    y = preds["hospital_expire_flag"].to_numpy()
    thresholds = np.arange(0.05, 0.81, 0.01)
    rows = []
    plt.figure(figsize=(5.6, 4.6), dpi=220)
    for model, label in [
        ("clinical_base", "Clinical base"),
        ("base_plus_trajectory", "Base + trajectory"),
        ("base_plus_full_lactate", "Base + full lactate dynamics"),
    ]:
        curve = dca_curve(y, preds[model].to_numpy(), thresholds)
        curve["model"] = model
        rows.append(curve)
        plt.plot(curve["threshold"], curve["net_benefit"], linewidth=1.8, label=label)
    base_curve = rows[0]
    plt.plot(base_curve["threshold"], base_curve["treat_all"], linestyle="--", color="gray", label="Treat all")
    plt.plot(base_curve["threshold"], base_curve["treat_none"], linestyle=":", color="black", label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision curve analysis")
    plt.ylim(-0.05, 0.35)
    plt.legend(fontsize=7)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return pd.concat(rows, ignore_index=True)


def landmark_sensitivity(mimic: pd.DataFrame) -> dict:
    df = mimic.copy()
    df["icu_los_hours"] = (
        pd.to_datetime(df["outtime"]) - pd.to_datetime(df["intime"])
    ).dt.total_seconds() / 3600
    landmark = df[df["icu_los_hours"] >= 24].copy()
    regression = fit_logit_or(
        landmark,
        "trajectory_group",
        [
            "age",
            "sofa",
            "charlson_comorbidity_index",
            "mbp_mean",
            "creatinine_max",
            "mechvent_24h",
            "vasoactive_24h",
        ],
    )
    mortality = mortality_table(landmark, "trajectory_group")
    return {
        "n_original": int(len(df)),
        "n_landmark_icu_los_ge_24h": int(len(landmark)),
        "excluded_icu_los_lt_24h": int((df["icu_los_hours"] < 24).sum()),
        "deaths_landmark": int(landmark["hospital_expire_flag"].sum()),
        "mortality_pct_landmark": float(landmark["hospital_expire_flag"].mean() * 100),
        "mortality": mortality.to_dict(orient="records"),
        "adjusted_logistic": regression,
    }


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    mimic = pd.read_csv(OUTPUTS / "mimic_analysis_dataset_with_trajectory.csv")
    eicu = pd.read_csv(OUTPUTS / "eicu_cs_analysis_dataset.csv")
    mimic_long = pd.read_csv(OUTPUTS / "mimic_cs_lactate_24h_long.csv")
    eicu_long = pd.read_csv(OUTPUTS / "eicu_cs_lactate_24h_long.csv")

    mimic_features = make_mimic_features(mimic_long)
    centroid_spec = fit_mimic_centroid_model(mimic_features)
    eicu_features = make_eicu_features(eicu_long, pd.Series(centroid_spec["fill_values"]))
    eicu_assign = assign_by_mimic_centroids(eicu_features, centroid_spec, "patientunitstayid")
    eicu_centroid = eicu.merge(eicu_assign, on="patientunitstayid", how="inner")
    eicu_centroid = eicu_centroid.rename(columns={"trajectory_group_mimic_centroid": "trajectory_group"})

    eicu_mortality = mortality_table(eicu_centroid, "trajectory_group")
    eicu_or = fit_logit_or(
        eicu_centroid,
        "trajectory_group",
        ["age", "apachescore", "meanbp", "creatinine", "vent", "vasoactive_24h"],
        "hospitalid",
    )
    eicu_centroid_landmark = eicu_centroid[
        eicu_centroid["hospitaldischargeoffset"].gt(24 * 60)
    ].copy()
    eicu_landmark_mortality = mortality_table(eicu_centroid_landmark, "trajectory_group")
    eicu_landmark_or = fit_logit_or(
        eicu_centroid_landmark,
        "trajectory_group",
        ["age", "apachescore", "meanbp", "creatinine", "vent", "vasoactive_24h"],
        "hospitalid",
    )
    eicu_assign.to_csv(OUTPUTS / "eicu_mimic_centroid_trajectory_assignments.csv", index=False)
    eicu_centroid.to_csv(OUTPUTS / "eicu_analysis_dataset_mimic_centroid_trajectory.csv", index=False)

    prediction_mimic = restrict_to_24h_survivors(
        mimic, OUTPUTS / "mimic_cs_lactate_24h_cohort.csv"
    )
    preds = cv_predictions(prediction_mimic)
    preds.to_csv(OUTPUTS / "mimic_q2_cv_predictions.csv", index=False)
    performance_rows = []
    y = preds["hospital_expire_flag"].to_numpy()
    for model in FEATURE_SETS:
        performance_rows.append(
            {
                "model": model,
                "auroc": round(float(roc_auc_score(y, preds[model])), 4),
                "auprc": round(float(average_precision_score(y, preds[model])), 4),
                "brier": round(float(brier_score_loss(y, preds[model])), 4),
            }
        )
    performance = pd.DataFrame(performance_rows)
    bootstrap = bootstrap_deltas(preds, n_boot=500)
    calibration = plot_calibration(preds, FIGURES / "figure_calibration_q2.png")
    dca = plot_dca(preds, FIGURES / "figure_dca_q2.png")
    landmark = landmark_sensitivity(mimic)

    eicu_mortality.to_csv(TABLES / "table_eicu_mimic_centroid_validation.csv", index=False)
    pd.DataFrame(
        [
            {
                "term": term,
                "or": round(vals["or"], 2),
                "ci95": f"{vals['ci95_low']:.2f}-{vals['ci95_high']:.2f}",
                "p_value": vals["p"],
            }
            for term, vals in eicu_or["terms"].items()
        ]
    ).to_csv(TABLES / "table_eicu_mimic_centroid_adjusted_or.csv", index=False)
    eicu_landmark_mortality.to_csv(
        TABLES / "table_eicu_mimic_centroid_24h_landmark_validation.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "term": term,
                "or": round(vals["or"], 2),
                "ci95": f"{vals['ci95_low']:.2f}-{vals['ci95_high']:.2f}",
                "p_value": vals["p"],
                "model_n": eicu_landmark_or["n_complete_cases"],
            }
            for term, vals in eicu_landmark_or["terms"].items()
        ]
    ).to_csv(TABLES / "table_eicu_mimic_centroid_24h_landmark_adjusted_or.csv", index=False)
    performance.to_csv(TABLES / "table_q2_prediction_performance.csv", index=False)
    bootstrap.to_csv(TABLES / "table_q2_bootstrap_prediction_deltas.csv", index=False)
    calibration.to_csv(TABLES / "table_q2_calibration_curve_data.csv", index=False)
    dca.to_csv(TABLES / "table_q2_decision_curve_data.csv", index=False)
    pd.DataFrame(landmark["mortality"]).to_csv(TABLES / "table_q2_landmark_mortality.csv", index=False)

    results = {
        "mimic_centroid_external_validation": {
            "n": int(len(eicu_centroid)),
            "deaths": int(eicu_centroid["hospital_expire_flag"].sum()),
            "mortality_pct": float(eicu_centroid["hospital_expire_flag"].mean() * 100),
            "mortality": eicu_mortality.to_dict(orient="records"),
            "adjusted_logistic": eicu_or,
        },
        "mimic_centroid_24h_landmark_validation": {
            "definition": "alive and hospitalized 24 hours after ICU admission",
            "n": int(len(eicu_centroid_landmark)),
            "deaths_after_landmark": int(eicu_centroid_landmark["hospital_expire_flag"].sum()),
            "mortality": eicu_landmark_mortality.to_dict(orient="records"),
            "adjusted_logistic": eicu_landmark_or,
        },
        "prediction_performance": performance.to_dict(orient="records"),
        "bootstrap_deltas": bootstrap.to_dict(orient="records"),
        "landmark_icu_los_ge_24h": landmark,
        "outputs": {
            "eicu_centroid_validation": "manuscript_tables/table_eicu_mimic_centroid_validation.csv",
            "eicu_centroid_or": "manuscript_tables/table_eicu_mimic_centroid_adjusted_or.csv",
            "prediction_performance": "manuscript_tables/table_q2_prediction_performance.csv",
            "bootstrap_deltas": "manuscript_tables/table_q2_bootstrap_prediction_deltas.csv",
            "calibration_figure": "manuscript_figures/figure_calibration_q2.png",
            "dca_figure": "manuscript_figures/figure_dca_q2.png",
        },
    }
    with (OUTPUTS / "q2_enhancement_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
