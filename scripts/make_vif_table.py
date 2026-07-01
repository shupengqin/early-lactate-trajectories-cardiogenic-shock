from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "mimic_analysis_dataset_with_trajectory.csv"
TABLES = ROOT / "manuscript_tables"

VARS = [
    "age",
    "sofa",
    "charlson_comorbidity_index",
    "mbp_mean",
    "creatinine_max",
    "mechvent_24h",
    "vasoactive_24h",
]


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT)
    x = df[VARS].copy()
    for col in VARS:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        x[col] = x[col].fillna(x[col].median())
    x = sm.add_constant(x, has_constant="add")
    rows = []
    for i, col in enumerate(x.columns):
        if col == "const":
            continue
        rows.append({"variable": col, "vif": round(float(variance_inflation_factor(x.values, i)), 3)})
    out = pd.DataFrame(rows).sort_values("vif", ascending=False)
    out.to_csv(TABLES / "table_supplementary_vif_primary_model.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

