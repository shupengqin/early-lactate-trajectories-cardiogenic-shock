import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT = Path("outputs/mimic_analysis_dataset_with_trajectory.csv")
OUT_DIR = Path("outputs")


BASE_FEATURES = [
    "age",
    "sofa",
    "charlson_comorbidity_index",
    "mbp_mean",
    "creatinine_max",
    "mechvent_24h",
    "vasoactive_24h",
]

INITIAL_LACTATE = ["initial_lactate_24h"]
CLEARANCE_FEATURES = ["initial_lactate_24h", "lactate_clearance_24h"]
TRAJECTORY_FEATURES = ["trajectory_group"]
FULL_LACTATE_FEATURES = [
    "initial_lactate_24h",
    "last_lactate_24h",
    "peak_lactate_24h",
    "lactate_slope_24h",
    "lactate_clearance_24h",
    "trajectory_group",
]


def calibration_metrics(y_true, y_prob):
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LinearRegression().fit(logit, y_true)
    return {
        "calibration_intercept": float(lr.intercept_),
        "calibration_slope": float(lr.coef_[0]),
    }


def make_preprocessor(df, features):
    categorical = [c for c in features if c in ["trajectory_group", "gender"]]
    numeric = [c for c in features if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
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


def evaluate_model(df, features, model_name, estimator):
    data = df[["hospital_expire_flag"] + features].copy()
    y = data["hospital_expire_flag"].astype(int).to_numpy()
    x = data[features]
    preprocessor = make_preprocessor(data, features)
    pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    prob = cross_val_predict(pipe, x, y, cv=cv, method="predict_proba")[:, 1]
    metrics = {
        "model": model_name,
        "features": features,
        "n": int(len(df)),
        "events": int(y.sum()),
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }
    metrics.update(calibration_metrics(y, prob))
    return metrics, prob


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT)

    model_specs = []
    feature_sets = {
        "clinical_base": BASE_FEATURES,
        "base_plus_initial_lactate": BASE_FEATURES + INITIAL_LACTATE,
        "base_plus_lactate_clearance": BASE_FEATURES + CLEARANCE_FEATURES,
        "base_plus_trajectory": BASE_FEATURES + TRAJECTORY_FEATURES,
        "base_plus_full_lactate": BASE_FEATURES + FULL_LACTATE_FEATURES,
    }
    for feature_name, features in feature_sets.items():
        model_specs.append(
            (
                f"logistic_{feature_name}",
                features,
                LogisticRegression(max_iter=2000, class_weight=None),
            )
        )
        model_specs.append(
            (
                f"rf_{feature_name}",
                features,
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=20,
                    random_state=2026,
                    n_jobs=-1,
                    class_weight=None,
                ),
            )
        )

    results = []
    predictions = pd.DataFrame(
        {
            "stay_id": df["stay_id"],
            "hospital_expire_flag": df["hospital_expire_flag"].astype(int),
        }
    )
    for name, features, estimator in model_specs:
        metrics, prob = evaluate_model(df, features, name, estimator)
        results.append(metrics)
        predictions[name] = prob

    results_df = pd.DataFrame(results).sort_values(["model"])
    results_df.to_csv(OUT_DIR / "mimic_prediction_model_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "mimic_prediction_model_predictions.csv", index=False)

    # Compact comparison table for logistic models, easier for manuscript.
    logistic_results = results_df[results_df["model"].str.startswith("logistic_")].copy()
    base = logistic_results.loc[
        logistic_results["model"] == "logistic_clinical_base"
    ].iloc[0]
    logistic_results["delta_auroc_vs_base"] = logistic_results["auroc"] - base["auroc"]
    logistic_results["delta_auprc_vs_base"] = logistic_results["auprc"] - base["auprc"]
    logistic_results.to_csv(
        OUT_DIR / "mimic_prediction_logistic_comparison.csv", index=False
    )

    output = {
        "all_models": results,
        "best_by_auroc": results_df.sort_values("auroc", ascending=False)
        .head(5)
        .to_dict(orient="records"),
        "logistic_comparison": logistic_results.to_dict(orient="records"),
        "outputs": {
            "metrics": "outputs/mimic_prediction_model_metrics.csv",
            "predictions": "outputs/mimic_prediction_model_predictions.csv",
            "logistic_comparison": "outputs/mimic_prediction_logistic_comparison.csv",
        },
    }
    with open(OUT_DIR / "mimic_prediction_model_results.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
