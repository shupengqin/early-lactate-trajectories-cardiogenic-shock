from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def restrict_to_24h_survivors(
    df: pd.DataFrame,
    cohort_path: Path,
    id_col: str = "stay_id",
) -> pd.DataFrame:
    """Return patients alive and still hospitalized 24 hours after ICU admission."""
    cohort = pd.read_csv(cohort_path, usecols=[id_col, "intime", "dischtime"])
    intime = pd.to_datetime(cohort["intime"], errors="coerce")
    dischtime = pd.to_datetime(cohort["dischtime"], errors="coerce")
    discharge_hours = (dischtime - intime).dt.total_seconds() / 3600
    at_risk_ids = cohort.loc[discharge_hours.gt(24), id_col]
    return df[df[id_col].isin(at_risk_ids)].copy().reset_index(drop=True)


def logistic_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    x = sm.add_constant(logit_p, has_constant="add")
    fit = sm.GLM(np.asarray(y_true, dtype=float), x, family=sm.families.Binomial()).fit()
    return {
        "calibration_intercept": float(fit.params[0]),
        "calibration_slope": float(fit.params[1]),
    }
