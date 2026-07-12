import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prediction_utils import logistic_calibration_metrics, restrict_to_24h_survivors


INPUT = Path("outputs/mimic_analysis_dataset_with_trajectory.csv")
COHORT = Path("outputs/mimic_cs_lactate_24h_cohort.csv")
OUT_DIR = Path("outputs")
TABLE_DIR = Path("manuscript_tables")


BASE_FEATURES = [
    "age",
    "sofa",
    "charlson_comorbidity_index",
    "mbp_mean",
    "creatinine_max",
    "mechvent_24h",
    "vasoactive_24h",
]

FULL_LACTATE_FEATURES = [
    "initial_lactate_24h",
    "last_lactate_24h",
    "peak_lactate_24h",
    "lactate_slope_24h",
    "lactate_clearance_24h",
    "trajectory_group",
]


def make_preprocessor(features, scale_numeric=True):
    categorical = [c for c in features if c in ["trajectory_group", "gender"]]
    numeric = [c for c in features if c not in categorical]
    transformers = []
    if numeric:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric))
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


def evaluate(df, model_name, estimator, features, scale_numeric):
    y = df["hospital_expire_flag"].astype(int).to_numpy()
    x = df[features].copy()
    pipe = Pipeline(
        [
            ("prep", make_preprocessor(features, scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    prob = cross_val_predict(pipe, x, y, cv=cv, method="predict_proba")[:, 1]
    out = {
        "model": model_name,
        "feature_set": "full_lactate_dynamic",
        "n": int(len(df)),
        "events": int(y.sum()),
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }
    out.update(logistic_calibration_metrics(y, prob))
    return out, prob


def main():
    OUT_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)
    df = restrict_to_24h_survivors(pd.read_csv(INPUT), COHORT)
    features = BASE_FEATURES + FULL_LACTATE_FEATURES

    specs = [
        (
            "logistic_l2",
            LogisticRegression(max_iter=3000, solver="lbfgs"),
            True,
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=700,
                min_samples_leaf=15,
                random_state=2026,
                n_jobs=-1,
            ),
            False,
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=700,
                min_samples_leaf=15,
                random_state=2026,
                n_jobs=-1,
            ),
            False,
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=2,
                random_state=2026,
            ),
            False,
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.03,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=2026,
            ),
            False,
        ),
    ]

    results = []
    predictions = pd.DataFrame(
        {
            "stay_id": df["stay_id"],
            "hospital_expire_flag": df["hospital_expire_flag"].astype(int),
        }
    )
    for name, estimator, scale_numeric in specs:
        metrics, prob = evaluate(df, name, estimator, features, scale_numeric)
        results.append(metrics)
        predictions[name] = prob

    results_df = pd.DataFrame(results).sort_values("auroc", ascending=False)
    results_df.to_csv(OUT_DIR / "mimic_extended_ml_model_metrics.csv", index=False)
    results_df.to_csv(TABLE_DIR / "table_extended_ml_performance.csv", index=False)
    predictions.to_csv(OUT_DIR / "mimic_extended_ml_predictions.csv", index=False)

    output = {
        "models": results_df.to_dict(orient="records"),
        "best_model_by_auroc": results_df.iloc[0].to_dict(),
        "outputs": {
            "metrics": "outputs/mimic_extended_ml_model_metrics.csv",
            "predictions": "outputs/mimic_extended_ml_predictions.csv",
        },
    }
    with open(OUT_DIR / "mimic_extended_ml_results.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
