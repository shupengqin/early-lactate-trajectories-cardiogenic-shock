import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MIMIC_PATH = Path("outputs/mimic_analysis_dataset_with_trajectory.csv")
EICU_COHORT_PATH = Path("outputs/eicu_cs_lactate_24h_cohort.csv")
OUT_TABLE_DIR = Path("manuscript_tables")
OUT_FIG_DIR = Path("manuscript_figures")
OUT_DIR = Path("outputs")


CONTINUOUS = [
    ("age", "Age, years"),
    ("sofa", "SOFA score"),
    ("sapsii", "SAPS II"),
    ("oasis", "OASIS"),
    ("charlson_comorbidity_index", "Charlson comorbidity index"),
    ("heart_rate_mean", "Heart rate, mean"),
    ("mbp_mean", "Mean arterial pressure, mean"),
    ("resp_rate_mean", "Respiratory rate, mean"),
    ("spo2_mean", "SpO2, mean"),
    ("hemoglobin_min", "Hemoglobin, minimum"),
    ("platelets_min", "Platelets, minimum"),
    ("wbc_max", "White blood cells, maximum"),
    ("bicarbonate_min", "Bicarbonate, minimum"),
    ("bun_max", "BUN, maximum"),
    ("creatinine_max", "Creatinine, maximum"),
    ("sodium_min", "Sodium, minimum"),
    ("potassium_max", "Potassium, maximum"),
    ("initial_lactate_24h", "Initial lactate"),
    ("last_lactate_24h", "Last lactate within 24 h"),
    ("peak_lactate_24h", "Peak lactate within 24 h"),
]

CATEGORICAL = [
    ("gender", "Female sex", "F"),
    ("myocardial_infarct", "Myocardial infarction", 1),
    ("congestive_heart_failure", "Congestive heart failure", 1),
    ("peripheral_vascular_disease", "Peripheral vascular disease", 1),
    ("chronic_pulmonary_disease", "Chronic pulmonary disease", 1),
    ("diabetes_without_cc", "Diabetes without complications", 1),
    ("diabetes_with_cc", "Diabetes with complications", 1),
    ("renal_disease", "Renal disease", 1),
    ("malignant_cancer", "Malignant cancer", 1),
    ("mechvent_24h", "Mechanical ventilation within 24 h", 1),
    ("vasoactive_24h", "Vasoactive agent within 24 h", 1),
    ("hospital_expire_flag", "In-hospital mortality", 1),
]


def fmt_cont(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return ""
    med = s.median()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"


def fmt_cat(series, positive):
    n = len(series)
    if n == 0:
        return ""
    count = (series == positive).sum()
    return f"{int(count)} ({count / n * 100:.1f})"


def make_table1():
    df = pd.read_csv(MIMIC_PATH)
    groups = sorted(df["trajectory_group"].dropna().astype(int).unique())
    rows = []

    rows.append({"variable": "N", "Overall": str(len(df)), **{f"Group {g}": str((df["trajectory_group"] == g).sum()) for g in groups}})

    for col, label in CONTINUOUS:
        row = {"variable": label, "Overall": fmt_cont(df[col])}
        for g in groups:
            row[f"Group {g}"] = fmt_cont(df.loc[df["trajectory_group"] == g, col])
        rows.append(row)

    for col, label, positive in CATEGORICAL:
        row = {"variable": label, "Overall": fmt_cat(df[col], positive)}
        for g in groups:
            row[f"Group {g}"] = fmt_cat(df.loc[df["trajectory_group"] == g, col], positive)
        rows.append(row)

    out = pd.DataFrame(rows)
    OUT_TABLE_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_TABLE_DIR / "table1_mimic_baseline_by_trajectory.csv", index=False)
    return out


def make_flow_counts():
    mimic_initial = pd.read_csv("outputs/mimic_cs_lactate_24h_cohort.csv")
    mimic_main = pd.read_csv("outputs/mimic_analysis_dataset_with_trajectory.csv")
    eicu = pd.read_csv(EICU_COHORT_PATH)
    counts = {
        "mimic": {
            "cs_first_icu": int(len(mimic_initial)),
            "lactate_ge1_24h": int((mimic_initial["lactate_n_24h"].fillna(0) >= 1).sum()),
            "lactate_ge2_24h": int((mimic_initial["lactate_n_24h"].fillna(0) >= 2).sum()),
            "main_analysis": int(len(mimic_main)),
        },
        "eicu": {
            "cs_first_icu": int(len(eicu)),
            "lactate_ge1_24h": int((eicu["lactate_n_24h"].fillna(0) >= 1).sum()),
            "lactate_ge2_24h": int((eicu["lactate_n_24h"].fillna(0) >= 2).sum()),
            "external_validation": int((eicu["eligible_lactate_trajectory_24h"].fillna(0) == 1).sum()),
        },
    }
    with open(OUT_DIR / "cohort_flow_counts.json", "w", encoding="utf-8") as handle:
        json.dump(counts, handle, ensure_ascii=False, indent=2)
    return counts


def draw_flow(counts):
    OUT_FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=180)
    for ax, title, data, final_label in [
        (axes[0], "MIMIC-IV", counts["mimic"], "Main analysis"),
        (axes[1], "eICU-CRD", counts["eicu"], "External validation"),
    ]:
        ax.axis("off")
        labels = [
            f"Cardiogenic shock\nfirst ICU admissions\nn={data['cs_first_icu']}",
            f">=1 lactate within 24 h\nn={data['lactate_ge1_24h']}",
            f">=2 lactates within 24 h\nn={data['lactate_ge2_24h']}",
            f"{final_label}\nn={data.get('main_analysis', data.get('external_validation'))}",
        ]
        y_positions = [0.85, 0.60, 0.35, 0.12]
        for y, text in zip(y_positions, labels):
            ax.text(
                0.5,
                y,
                text,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#f4f6f8", edgecolor="#4b5563"),
                fontsize=9,
            )
        for y1, y2 in zip(y_positions[:-1], y_positions[1:]):
            ax.annotate(
                "",
                xy=(0.5, y2 + 0.08),
                xytext=(0.5, y1 - 0.08),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.4),
            )
        ax.set_title(title, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "figure_cohort_flow.png")
    plt.close(fig)


def main():
    table1 = make_table1()
    counts = make_flow_counts()
    draw_flow(counts)
    manifest = {
        "table1": "manuscript_tables/table1_mimic_baseline_by_trajectory.csv",
        "flow_counts": "outputs/cohort_flow_counts.json",
        "flow_figure": "manuscript_figures/figure_cohort_flow.png",
        "table1_rows": int(len(table1)),
    }
    with open(OUT_DIR / "table1_flow_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
