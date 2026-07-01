import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("outputs")
TABLE_DIR = Path("manuscript_tables")
FIG_DIR = Path("manuscript_figures")


def ensure_dirs():
    TABLE_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def table_trajectory_main():
    mimic_mort = pd.read_csv(OUT_DIR / "mimic_lactate_trajectory_mortality.csv")
    mimic_sum = pd.read_csv(OUT_DIR / "mimic_lactate_trajectory_summary.csv")
    out = mimic_sum.merge(
        mimic_mort[
            [
                "trajectory_group",
                "deaths",
                "mortality_pct",
                "initial_lactate_median",
                "last_lactate_median",
                "peak_lactate_median",
            ]
        ],
        on="trajectory_group",
        how="left",
    )
    out.to_csv(TABLE_DIR / "table_mimic_trajectory_groups.csv", index=False)
    return out


def table_eicu_validation():
    with open(OUT_DIR / "eicu_external_validation_results.json", "r", encoding="utf-8") as handle:
        res = json.load(handle)
    traj = pd.DataFrame(res["trajectory_summary"])
    mort = pd.DataFrame(res["trajectory_mortality"])
    out = traj.merge(
        mort[
            [
                "trajectory_group",
                "deaths",
                "mortality_pct",
                "initial_lactate_median",
                "last_lactate_median",
                "peak_lactate_median",
            ]
        ],
        on="trajectory_group",
        how="left",
    )
    out.to_csv(TABLE_DIR / "table_eicu_trajectory_validation.csv", index=False)
    return out


def table_regression():
    with open(OUT_DIR / "mimic_lactate_trajectory_results.json", "r", encoding="utf-8") as handle:
        mimic = json.load(handle)
    with open(OUT_DIR / "eicu_external_validation_results.json", "r", encoding="utf-8") as handle:
        eicu = json.load(handle)

    rows = []
    for term, label in [
        ("traj_2", "Trajectory group 2 vs 1"),
        ("traj_3", "Trajectory group 3 vs 1"),
        ("traj_4", "Trajectory group 4 vs 1"),
    ]:
        m = mimic["adjusted_logistic_trajectory"]["terms"].get(term)
        e = eicu["adjusted_logistic_trajectory_apache"]["terms"].get(term)
        rows.append(
            {
                "comparison": label,
                "mimic_or": m["or"],
                "mimic_ci95": f"{m['ci95_low']:.2f}-{m['ci95_high']:.2f}",
                "mimic_p": m["p"],
                "eicu_or": e["or"],
                "eicu_ci95": f"{e['ci95_low']:.2f}-{e['ci95_high']:.2f}",
                "eicu_p": e["p"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table_adjusted_trajectory_or_mimic_eicu.csv", index=False)
    return out


def table_prediction():
    pred = pd.read_csv(OUT_DIR / "mimic_prediction_logistic_comparison.csv")
    cols = [
        "model",
        "auroc",
        "auprc",
        "brier",
        "delta_auroc_vs_base",
        "delta_auprc_vs_base",
    ]
    out = pred[cols].copy()
    out.to_csv(TABLE_DIR / "table_prediction_performance.csv", index=False)
    return out


def plot_trajectory_comparison(mimic_table, eicu_table):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=180, sharey=True)
    x = [3, 9, 15, 21]
    for ax, df, title in [
        (axes[0], mimic_table, "MIMIC-IV"),
        (axes[1], eicu_table, "eICU-CRD"),
    ]:
        for _, row in df.iterrows():
            y = [
                row["mean_0_6h"],
                row["mean_6_12h"],
                row["mean_12_18h"],
                row["mean_18_24h"],
            ]
            label = f"G{int(row['trajectory_group'])} ({int(row['n'])})"
            ax.plot(x, y, marker="o", linewidth=2, label=label)
        ax.set_title(title)
        ax.set_xlabel("Hours after ICU admission")
        ax.set_xticks(x, ["0-6", "6-12", "12-18", "18-24"])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean lactate (mmol/L)")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_trajectory_mimic_eicu.png")
    plt.close(fig)


def plot_mortality_gradient(mimic_table, eicu_table):
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=180)
    x = range(1, 5)
    ax.plot(
        x,
        mimic_table.sort_values("trajectory_group")["mortality_pct"],
        marker="o",
        linewidth=2,
        label="MIMIC-IV",
    )
    ax.plot(
        x,
        eicu_table.sort_values("trajectory_group")["mortality_pct"],
        marker="s",
        linewidth=2,
        label="eICU-CRD",
    )
    ax.set_xlabel("Lactate trajectory group")
    ax.set_ylabel("In-hospital mortality (%)")
    ax.set_xticks(list(x))
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_mortality_gradient_mimic_eicu.png")
    plt.close(fig)


def plot_prediction_performance(pred_table):
    order = [
        "logistic_clinical_base",
        "logistic_base_plus_initial_lactate",
        "logistic_base_plus_lactate_clearance",
        "logistic_base_plus_trajectory",
        "logistic_base_plus_full_lactate",
    ]
    labels = [
        "Clinical",
        "+ Initial lactate",
        "+ Clearance",
        "+ Trajectory",
        "+ Full lactate",
    ]
    df = pred_table.set_index("model").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=180)
    ax.plot(labels, df["auroc"], marker="o", linewidth=2, label="AUROC")
    ax.plot(labels, df["auprc"], marker="s", linewidth=2, label="AUPRC")
    ax.set_ylim(0.55, 0.82)
    ax.set_ylabel("Performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_prediction_performance.png")
    plt.close(fig)


def main():
    ensure_dirs()
    mimic_table = table_trajectory_main()
    eicu_table = table_eicu_validation()
    regression_table = table_regression()
    prediction_table = table_prediction()
    plot_trajectory_comparison(mimic_table, eicu_table)
    plot_mortality_gradient(mimic_table, eicu_table)
    plot_prediction_performance(prediction_table)
    summary = {
        "tables": [
            "manuscript_tables/table_mimic_trajectory_groups.csv",
            "manuscript_tables/table_eicu_trajectory_validation.csv",
            "manuscript_tables/table_adjusted_trajectory_or_mimic_eicu.csv",
            "manuscript_tables/table_prediction_performance.csv",
        ],
        "figures": [
            "manuscript_figures/figure_trajectory_mimic_eicu.png",
            "manuscript_figures/figure_mortality_gradient_mimic_eicu.png",
            "manuscript_figures/figure_prediction_performance.png",
        ],
    }
    with open(OUT_DIR / "manuscript_outputs_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
