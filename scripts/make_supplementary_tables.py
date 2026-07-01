from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TABLES = ROOT / "manuscript_tables"


MIMIC_VARS = [
    "age",
    "gender",
    "lactate_n_24h",
    "initial_lactate_24h",
    "last_lactate_24h",
    "peak_lactate_24h",
    "lactate_clearance_24h",
    "sofa",
    "sapsii",
    "oasis",
    "charlson_comorbidity_index",
    "mbp_mean",
    "creatinine_max",
    "mechvent_24h",
    "vasoactive_24h",
    "trajectory_group",
]

EICU_VARS = [
    "age",
    "gender",
    "lactate_n_24h",
    "initial_lactate_24h",
    "last_lactate_24h",
    "peak_lactate_24h",
    "lactate_clearance_24h",
    "acutephysiologyscore",
    "apachescore",
    "meanbp",
    "creatinine",
    "vent",
    "vasoactive_24h",
    "trajectory_group",
]


def missingness_table(df: pd.DataFrame, variables: list[str], cohort: str) -> pd.DataFrame:
    rows = []
    n = len(df)
    for var in variables:
        if var not in df.columns:
            rows.append(
                {
                    "cohort": cohort,
                    "variable": var,
                    "n_total": n,
                    "n_missing": "not_available",
                    "missing_pct": "not_available",
                }
            )
            continue
        missing = int(df[var].isna().sum())
        rows.append(
            {
                "cohort": cohort,
                "variable": var,
                "n_total": n,
                "n_missing": missing,
                "missing_pct": round(missing / n * 100, 2) if n else 0,
            }
        )
    return pd.DataFrame(rows)


def build_missingness() -> pd.DataFrame:
    mimic = pd.read_csv(OUTPUTS / "mimic_analysis_dataset_with_trajectory.csv")
    eicu = pd.read_csv(OUTPUTS / "eicu_analysis_dataset_with_trajectory.csv")
    table = pd.concat(
        [
            missingness_table(mimic, MIMIC_VARS, "MIMIC-IV"),
            missingness_table(eicu, EICU_VARS, "eICU-CRD"),
        ],
        ignore_index=True,
    )
    return table


def build_sensitivity_summary() -> pd.DataFrame:
    with (OUTPUTS / "mimic_trajectory_sensitivity_results.json").open(
        "r", encoding="utf-8"
    ) as f:
        results = json.load(f)

    rows = []
    for scenario in results["scenarios"]:
        label = scenario["scenario"]
        k = scenario["k"]
        min_lactate_count = scenario["min_lactate_count"]
        for group in scenario["mortality"]:
            term = f"traj_{int(group['trajectory_group'])}"
            adjusted = scenario["adjusted_logistic"]["trajectory_terms"].get(term)
            rows.append(
                {
                    "scenario": label,
                    "k": k,
                    "min_lactate_count": min_lactate_count,
                    "trajectory_group": int(group["trajectory_group"]),
                    "n": int(group["n"]),
                    "deaths": int(group["deaths"]),
                    "mortality_pct": round(float(group["mortality_pct"]), 2),
                    "initial_lactate_median": group["initial_lactate_median"],
                    "last_lactate_median": group["last_lactate_median"],
                    "peak_lactate_median": group["peak_lactate_median"],
                    "adjusted_or_vs_group1": (
                        round(float(adjusted["or"]), 2) if adjusted else "reference"
                    ),
                    "ci95": (
                        f"{float(adjusted['ci95_low']):.2f}-{float(adjusted['ci95_high']):.2f}"
                        if adjusted
                        else "reference"
                    ),
                    "p_value": adjusted["p"] if adjusted else "reference",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    missingness = build_missingness()
    sensitivity = build_sensitivity_summary()

    missingness.to_csv(
        TABLES / "table_supplementary_missingness_mimic_eicu.csv", index=False
    )
    sensitivity.to_csv(
        TABLES / "table_supplementary_sensitivity_trajectory.csv", index=False
    )

    print("Generated supplementary tables:")
    print(TABLES / "table_supplementary_missingness_mimic_eicu.csv")
    print(TABLES / "table_supplementary_sensitivity_trajectory.csv")


if __name__ == "__main__":
    main()

