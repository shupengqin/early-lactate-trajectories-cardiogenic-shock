from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "manuscript_figures"
TABLES = ROOT / "manuscript_tables"
OUTPUTS = ROOT / "outputs"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


COLORS = {
    "g1": "#4c78a8",
    "g2": "#72b7b2",
    "g3": "#f28e2b",
    "g4": "#b07aa1",
    "g5": "#59a14f",
    "mimic": "#4c78a8",
    "eicu": "#f28e2b",
    "centroid": "#59a14f",
    "gray": "#7f8c8d",
    "dark": "#2f3440",
    "light": "#f4f6f8",
}


def save_pub(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    base = FIGURES / stem
    png = base.with_suffix(".png")
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    with Image.open(png) as im:
        im.convert("RGB").save(base.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")


def panel_label(ax: plt.Axes, label: str, x: float = -0.10) -> None:
    ax.text(
        x,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def parse_or_ci(text: str) -> tuple[float, float, float]:
    match = re.search(r"([0-9.]+)\s*\(([0-9.]+)-([0-9.]+)\)", str(text))
    if not match:
        raise ValueError(f"Cannot parse OR and CI from {text!r}")
    return tuple(float(x) for x in match.groups())


def group_color(group: int) -> str:
    return COLORS.get(f"g{group}", COLORS["gray"])


def draw_flow(ax: plt.Axes, title: str, counts: dict[str, int], final_key: str, label: str) -> None:
    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, label)

    steps = [
        ("Cardiogenic shock\nfirst ICU stay", "cs_first_icu"),
        (">=1 lactate within\nfirst 24 h", "lactate_ge1_24h"),
        (">=2 lactates within\nfirst 24 h", "lactate_ge2_24h"),
        ("Analysis cohort", final_key),
    ]
    y_positions = [0.84, 0.60, 0.36, 0.14]
    for (text, key), y in zip(steps, y_positions):
        ax.text(
            0.50,
            y,
            f"{text}\nN = {counts[key]:,}",
            ha="center",
            va="center",
            fontsize=8,
            linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.32", facecolor=COLORS["light"], edgecolor="#596270", linewidth=0.8),
        )
    for y1, y2 in zip(y_positions[:-1], y_positions[1:]):
        ax.annotate(
            "",
            xy=(0.5, y2 + 0.085),
            xytext=(0.5, y1 - 0.085),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#596270"),
        )


def lactate_availability(ax: plt.Axes, counts: dict[str, dict[str, int]]) -> None:
    panel_label(ax, "C")
    cohorts = ["MIMIC-IV", "eICU-CRD"]
    keys = ["mimic", "eicu"]
    left = np.zeros(len(keys))
    categories = [
        ("No lactate in 24 h", lambda c: c["cs_first_icu"] - c["lactate_ge1_24h"], "#d8dee6"),
        ("One lactate only", lambda c: c["lactate_ge1_24h"] - c["lactate_ge2_24h"], "#a9b7c6"),
        (">=2 lactates", lambda c: c["lactate_ge2_24h"], COLORS["mimic"]),
    ]
    totals = np.array([counts[k]["cs_first_icu"] for k in keys])
    for name, fn, color in categories:
        vals = np.array([fn(counts[k]) for k in keys])
        pct = vals / totals * 100
        ax.barh(cohorts, pct, left=left, color=color, edgecolor="white", height=0.48, label=name)
        for i, (lft, width, val) in enumerate(zip(left, pct, vals)):
            if width >= 9:
                ax.text(lft + width / 2, i, f"{val:,}\n{width:.1f}%", ha="center", va="center", fontsize=8)
        left += pct
    ax.set_xlim(0, 100)
    ax.set_xlabel("Proportion of first ICU cardiogenic shock stays (%)")
    ax.set_title("Early lactate availability", fontweight="bold", pad=6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.52), ncol=3, fontsize=8, handlelength=1.4, borderaxespad=0.4)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)


def figure1() -> None:
    counts = json.loads((OUTPUTS / "cohort_flow_counts.json").read_text(encoding="utf-8"))
    traj = pd.read_csv(TABLES / "table2_trajectory_groups_journal.csv")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def rect_box(x, y, w, h, text, fc, ec="#30343b", fs=8.0, weight="normal", wrap=24):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.006",
            linewidth=0.85,
            edgecolor=ec,
            facecolor=fc,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)
        wrapped = "\n".join(
            "\n".join(textwrap.wrap(part, width=wrap, break_long_words=False)) for part in str(text).split("\n")
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            wrapped,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight=weight,
            linespacing=1.08,
            color="#111111",
        )
        return rect

    def header(x, y, w, h, text, color):
        rect_box(x, y, w, h, "", color, fs=9.2, weight="bold", wrap=44)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.2,
            color="white",
            fontweight="bold",
        )

    def down_arrow(x, y_start, y_end):
        ax.annotate(
            "",
            xy=(x, y_end),
            xytext=(x, y_start),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#222222", shrinkA=0, shrinkB=0),
        )

    def side_arrow(x_start, y, x_end):
        ax.annotate(
            "",
            xy=(x_end, y),
            xytext=(x_start, y),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#222222", shrinkA=0, shrinkB=0),
        )

    def group_counts(cohort):
        sub = traj[traj["Cohort"].eq(cohort)].copy()
        out = {}
        for _, row in sub.iterrows():
            g = int(str(row["Group"]).split()[-1])
            out[g] = int(row["N"])
        return out

    mimic_groups = group_counts("MIMIC-IV")
    eicu_groups = group_counts("eICU-CRD")

    cohorts = [
        {
            "header": "Derivation cohort (MIMIC-IV)",
            "color": "#9f2f2f",
            "light": "#f7dede",
            "x0": 0.035,
            "main_x": 0.065,
            "excl_x": 0.320,
            "counts": counts["mimic"],
            "final": "main_analysis",
            "initial": "First ICU admission with\ncardiogenic shock\nn = 2,922",
            "excluded_1": "Excluded: n = 331\nNo lactate in first 24 h",
            "included_1": "At least one lactate\nin first 24 h\nn = 2,591",
            "excluded_2": "Excluded: n = 410\nOnly one lactate in first 24 h",
            "final_text": "Trajectory: n = 2,181\nLandmark: n = 2,015",
            "groups": mimic_groups,
        },
        {
            "header": "External validation cohort (eICU-CRD)",
            "color": "#2f4f9f",
            "light": "#dfe5fb",
            "x0": 0.535,
            "main_x": 0.565,
            "excl_x": 0.820,
            "counts": counts["eicu"],
            "final": "external_validation",
            "initial": "First ICU admission with\ncardiogenic shock\nn = 1,650",
            "excluded_1": "Excluded: n = 854\nNo lactate in first 24 h",
            "included_1": "At least one lactate\nin first 24 h\nn = 796",
            "excluded_2": "Excluded: n = 320\nOnly one lactate in first 24 h",
            "final_text": "Trajectory: n = 476\nLandmark: n = 429",
            "groups": eicu_groups,
        },
    ]

    box_w = 0.20
    excl_w = 0.19
    h_initial = 0.120
    h_mid = 0.090
    h_final = 0.090
    for spec in cohorts:
        header(spec["x0"], 0.925, 0.43, 0.05, spec["header"], spec["color"])
        rect_box(spec["main_x"], 0.745, box_w, h_initial, spec["initial"], spec["light"], fs=8.0, wrap=24)
        rect_box(spec["excl_x"], 0.650, excl_w, 0.090, spec["excluded_1"], "#f4f5f7", fs=8.0, wrap=24)
        side_arrow(spec["main_x"] + box_w + 0.018, 0.695, spec["excl_x"] - 0.015)
        down_arrow(spec["main_x"] + box_w / 2, 0.745, 0.635)
        rect_box(spec["main_x"] + 0.015, 0.545, box_w - 0.030, h_mid, spec["included_1"], spec["light"], fs=8.0, wrap=22)
        rect_box(spec["excl_x"] - 0.010, 0.435, excl_w + 0.035, 0.105, spec["excluded_2"], "#f4f5f7", fs=8.0, wrap=27)
        side_arrow(spec["main_x"] + box_w + 0.018, 0.487, spec["excl_x"] - 0.025)
        down_arrow(spec["main_x"] + box_w / 2, 0.545, 0.425)
        rect_box(spec["main_x"] - 0.010, 0.330, box_w + 0.020, h_final, spec["final_text"], spec["light"], fs=8.0, wrap=27)

        table_x = spec["x0"]
        table_y = 0.050
        table_w = 0.43
        table_h = 0.185
        rect_box(table_x, table_y, table_w, table_h, "", "#ffffff", ec="#30343b", fs=8)
        ax.text(
            table_x + table_w / 2,
            table_y + table_h - 0.024,
            "Trajectory groups in the final cohort",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        ax.plot([table_x, table_x + table_w], [table_y + table_h - 0.048, table_y + table_h - 0.048], color="#30343b", lw=0.75, transform=ax.transAxes)
        cell_w = table_w / 2
        ax.plot([table_x + cell_w, table_x + cell_w], [table_y, table_y + table_h - 0.048], color="#30343b", lw=0.75, transform=ax.transAxes)
        row_mid = table_y + (table_h - 0.048) / 2
        ax.plot([table_x, table_x + table_w], [row_mid, row_mid], color="#30343b", lw=0.75, transform=ax.transAxes)
        down_arrow(spec["main_x"] + box_w / 2, 0.330, table_y + table_h + 0.020)
        for group, j in zip([1, 2, 3, 4], range(4)):
            label = {
                1: "G1 low-stable",
                2: "G2 moderate-decreasing",
                3: "G3 high-decreasing",
                4: "G4 persistent-high",
            }[group]
            col = j % 2
            row = 1 - j // 2
            cx = table_x + cell_w * col + cell_w / 2
            cy = table_y + (row + 0.5) * (table_h - 0.048) / 2
            ax.text(
                cx,
                cy,
                f"{label}\nN = {spec['groups'][group]:,}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                linespacing=1.12,
            )
    save_pub(fig, "figure1_study_flow_composite")


def plot_trajectories(ax: plt.Axes, df: pd.DataFrame, cohort: str, label: str) -> None:
    panel_label(ax, label)
    time_cols = [
        "Mean lactate 0-6 h",
        "Mean lactate 6-12 h",
        "Mean lactate 12-18 h",
        "Mean lactate 18-24 h",
    ]
    x = np.array([3, 9, 15, 21])
    sub = df[df["Cohort"] == cohort].copy()
    for _, row in sub.iterrows():
        group = int(str(row["Group"]).split()[-1])
        y = [row[col] for col in time_cols]
        color = group_color(group)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=3.5,
            linewidth=1.7,
            color=color,
        )
        label_offset = {1: -0.30, 2: 0.30, 3: 0.0, 4: 0.0}.get(group, 0.0)
        ax.text(
            x[-1] + 0.85,
            y[-1] + label_offset,
            f"G{group} (n={int(row['N'])})",
            color=color,
            fontsize=8,
            va="center",
            ha="left",
        )
    ax.set_title(cohort, fontweight="bold", pad=5)
    ax.set_xlabel("Hours after ICU admission")
    ax.set_ylabel("Mean lactate (mmol/L)")
    ax.set_xticks(x, ["0-6", "6-12", "12-18", "18-24"])
    ax.set_xlim(2.2, 30.5)
    ax.set_ylim(0.7, 13.4)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)


def plot_mortality_gradient(ax: plt.Axes, df: pd.DataFrame, label: str) -> None:
    panel_label(ax, label)
    cohorts = ["MIMIC-IV", "eICU-CRD"]
    x = np.arange(1, 5)
    width = 0.36
    for offset, cohort, color in [(-width / 2, cohorts[0], COLORS["mimic"]), (width / 2, cohorts[1], COLORS["eicu"])]:
        sub = df[df["cohort"] == cohort].sort_values("trajectory_group").copy()
        vals = sub["mortality_pct"].to_numpy()
        ax.bar(x + offset, vals, width=width, color=color, label=cohort)
        label_lift = 2.0 if offset < 0 else 6.0
        for xi, yi in zip(x + offset, vals):
            ax.text(xi, yi + label_lift, f"{yi:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [f"G{i}" for i in x])
    ax.set_ylim(0, 100)
    ax.set_ylabel("In-hospital mortality (%)")
    ax.set_title("Landmark mortality", fontweight="bold", pad=5)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, handlelength=1.4)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)


def forest_from_table3(ax: plt.Axes, table: pd.DataFrame, label: str) -> None:
    panel_label(ax, label)
    rows = []
    for _, row in table.iterrows():
        comp = row["Comparison"].replace("Trajectory group ", "G").replace(" vs 1", " vs G1")
        for cohort, color in [("MIMIC-IV", COLORS["mimic"]), ("eICU-CRD", COLORS["eicu"])]:
            odds, low, high = parse_or_ci(row[f"{cohort} adjusted OR (95% CI)"])
            rows.append({"label": comp, "cohort": cohort, "or": odds, "low": low, "high": high, "color": color})
    y_positions = np.arange(len(rows))[::-1]
    for y, row in zip(y_positions, rows):
        ax.errorbar(
            row["or"],
            y,
            xerr=[[row["or"] - row["low"]], [row["high"] - row["or"]]],
            fmt="o",
            color=row["color"],
            ecolor=row["color"],
            elinewidth=1.1,
            capsize=2.5,
            markersize=4,
        )
    ax.axvline(1, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.65, 100)
    ax.set_yticks(y_positions, [f"G{r['label'][1]}, {r['cohort'].replace('-CRD', '')}" for r in rows])
    ax.set_xlabel("Adjusted odds ratio")
    ax.set_title("Adjusted associations", fontweight="bold", pad=5)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6, which="both")


def plot_centroid_mortality(ax: plt.Axes, df: pd.DataFrame, label: str) -> None:
    panel_label(ax, label)
    groups = df["trajectory_group"].astype(int).to_numpy()
    vals = df["mortality_pct"].to_numpy()
    ax.bar(groups, vals, color=[group_color(g) for g in groups], width=0.65)
    for x, y, n in zip(groups, vals, df["n"]):
        ax.text(x, y + 2.2, f"{y:.1f}%\nn={int(n)}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(groups, [f"G{i}" for i in groups])
    ax.set_ylim(0, 100)
    ax.set_ylabel("In-hospital mortality (%)")
    ax.set_title("Fixed-centroid mortality", fontweight="bold", pad=5, x=0.60)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)


def plot_centroid_forest(ax: plt.Axes, df: pd.DataFrame, label: str) -> None:
    panel_label(ax, label)
    d = df.copy()
    d["group"] = d["term"].str.extract(r"traj_(\d+)").astype(int)
    d[["low", "high"]] = d["ci95"].str.split("-", expand=True).astype(float)
    d = d.sort_values("group", ascending=False)
    y = np.arange(len(d))
    for yy, (_, row) in zip(y, d.iterrows()):
        odds = float(row["or"])
        low = float(row["low"])
        high = float(row["high"])
        group = int(row["group"])
        ax.errorbar(
            odds,
            yy,
            xerr=[[odds - low], [high - odds]],
            fmt="o",
            color=group_color(group),
            ecolor=group_color(group),
            elinewidth=1.1,
            capsize=2.5,
            markersize=4,
        )
    ax.axvline(1, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.55, 35)
    ax.set_yticks(y, [f"G{int(g)} vs G1" for g in d["group"]])
    ax.set_xlabel("Adjusted odds ratio")
    ax.set_title("Centroid ORs", fontweight="bold", pad=5)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6, which="both")


def figure2() -> None:
    traj = pd.read_csv(TABLES / "table2_trajectory_groups_journal.csv")
    landmark = pd.read_csv(TABLES / "table_primary_24h_landmark_associations.csv")
    ors = pd.read_csv(TABLES / "table3_adjusted_or_journal.csv")
    centroid = pd.read_csv(TABLES / "table_eicu_mimic_centroid_24h_landmark_validation.csv")
    centroid_or = pd.read_csv(TABLES / "table_eicu_mimic_centroid_24h_landmark_adjusted_or.csv")

    fig = plt.figure(figsize=(7.5, 5.8))
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.10, top=0.91)
    gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.88, width_ratios=[1.12, 1.12, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    plot_trajectories(ax_a, traj, "MIMIC-IV", "A")
    plot_trajectories(ax_b, traj, "eICU-CRD", "B")
    ax_b.set_ylabel("")
    plot_mortality_gradient(ax_c, landmark, "C")
    forest_from_table3(ax_d, ors, "D")
    plot_centroid_mortality(ax_e, centroid, "E")
    plot_centroid_forest(ax_f, centroid_or, "F")
    save_pub(fig, "figure2_trajectory_validation_composite")


def _delta_limits(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    parts = series.str.extract(r"^(-?[0-9.]+)-(-?[0-9.]+)$").astype(float)
    return parts[0].to_numpy(), parts[1].to_numpy()


def figure3() -> None:
    performance = pd.read_csv(TABLES / "table4_prediction_performance_journal.csv")
    algorithms = pd.read_csv(TABLES / "table_extended_ml_performance.csv")
    bootstrap = pd.read_csv(TABLES / "table_q2_bootstrap_prediction_deltas.csv")

    fig = plt.figure(figsize=(7.5, 5.6))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.10, top=0.91)
    gs = fig.add_gridspec(2, 2, hspace=0.72, wspace=0.62)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    ax = axes[0]
    panel_label(ax, "A")
    x = np.arange(len(performance))
    ax.plot(x, performance["AUROC"], marker="o", color=COLORS["mimic"], label="AUROC")
    ax.plot(x, performance["AUPRC"], marker="s", color=COLORS["g2"], label="AUPRC")
    ax.set_xticks(x, ["Clinical", "+ Initial", "+ Clearance", "+ Trajectory", "+ Full"], rotation=18, ha="right")
    ax.set_ylim(0.50, 0.78)
    ax.set_ylabel("Cross-validated performance")
    ax.set_title("Feature-set comparison", fontweight="bold", pad=5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)

    ax = axes[1]
    panel_label(ax, "B")
    algorithms = algorithms.sort_values("auroc")
    y = np.arange(len(algorithms))
    labels = algorithms["model"].replace({
        "logistic_l2": "L2 logistic",
        "random_forest": "Random forest",
        "extra_trees": "ExtraTrees",
        "gradient_boosting": "Gradient boosting",
        "hist_gradient_boosting": "Hist gradient boosting",
    })
    ax.scatter(algorithms["auroc"], y + 0.10, color=COLORS["mimic"], label="AUROC")
    ax.scatter(algorithms["auprc"], y - 0.10, color=COLORS["g2"], marker="s", label="AUPRC")
    ax.set_yticks(y, labels)
    ax.set_xlim(0.54, 0.76)
    ax.set_xlabel("Cross-validated performance")
    ax.set_title("Prespecified algorithms", fontweight="bold", pad=5)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, handlelength=1.2)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)

    short_labels = ["Trajectory vs base", "Full vs base", "Full vs trajectory"]
    for ax, metric, ci_col, title, label in [
        (axes[2], "delta_auroc", "delta_auroc_ci95", "Incremental AUROC", "C"),
        (axes[3], "delta_auprc", "delta_auprc_ci95", "Incremental AUPRC", "D"),
    ]:
        panel_label(ax, label)
        low, high = _delta_limits(bootstrap[ci_col])
        estimates = bootstrap[metric].to_numpy()
        y = np.arange(len(estimates))[::-1]
        ax.errorbar(estimates, y, xerr=[estimates - low, high - estimates], fmt="o", color=COLORS["dark"], capsize=3)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_yticks(y, short_labels)
        ax.set_xlabel("Absolute difference (95% CI)")
        ax.set_title(title, fontweight="bold", pad=5)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)

    save_pub(fig, "figure3_prediction_performance_composite")


def figure4() -> None:
    calibration = pd.read_csv(TABLES / "table_q2_calibration_curve_data.csv")
    dca = pd.read_csv(TABLES / "table_q2_decision_curve_data.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.25))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.88, wspace=0.34)

    ax = axes[0]
    panel_label(ax, "A")
    model_labels = {
        "clinical_base": ("Clinical", COLORS["gray"]),
        "base_plus_trajectory": ("+ Trajectory", COLORS["g2"]),
        "base_plus_full_lactate": ("+ Full lactate", COLORS["g3"]),
    }
    for model, (label, color) in model_labels.items():
        sub = calibration[calibration["model"].eq(model)].sort_values("mean_predicted")
        ax.plot(sub["mean_predicted"], sub["observed"], marker="o", markersize=3, color=color, label=label)
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted mortality")
    ax.set_ylabel("Observed mortality")
    ax.set_title("Calibration", fontweight="bold", pad=5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(color="#e5e7eb", linewidth=0.6)

    ax = axes[1]
    panel_label(ax, "B")
    for model, (label, color) in model_labels.items():
        sub = dca[dca["model"].eq(model)].sort_values("threshold")
        ax.plot(sub["threshold"], sub["net_benefit"], color=color, label=label)
    base = dca[dca["model"].eq("clinical_base")].sort_values("threshold")
    ax.plot(base["threshold"], base["treat_all"], color="#555555", linestyle=":", label="Treat all")
    ax.plot(base["threshold"], base["treat_none"], color="#888888", linestyle="--", label="Treat none")
    ax.set_xlim(0.05, 0.80)
    ax.set_ylim(-0.03, 0.34)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis", fontweight="bold", pad=5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(color="#e5e7eb", linewidth=0.6)

    save_pub(fig, "figure4_calibration_decision_composite")


def mortality_panel(ax: plt.Axes, df: pd.DataFrame, title: str, label: str) -> None:
    panel_label(ax, label, x=-0.12)
    groups = df["trajectory_group"].astype(int).to_numpy()
    vals = df["mortality_pct"].astype(float).to_numpy()
    ax.bar(groups, vals, color=[group_color(g) for g in groups], width=0.65)
    label_heights: list[float] = []
    for y in vals:
        label_y = y + 1.5
        if label_heights and label_y - label_heights[-1] < 5.0:
            label_y += 5.0
        label_heights.append(label_y)
    for x, y, label_y in zip(groups, vals, label_heights):
        ax.text(x, label_y, f"{y:.1f}%", ha="center", va="bottom", fontsize=8, color="#111111")
    ax.set_xticks(groups, [f"G{i}" for i in groups])
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Mortality (%)")
    ax.set_title(title, fontweight="bold", pad=5)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)


def ami_mortality(ax: plt.Axes, df: pd.DataFrame, label: str) -> None:
    panel_label(ax, label, x=-0.12)
    d = df[df["subgroup_definition"].eq("AMI-CS by acute MI ICD")].copy()
    d = d[d["subgroup"].str.contains("trajectory group", case=False, na=False)]
    d["ami"] = d["subgroup"].str.extract(r"^(Yes|No):")
    d["group"] = d["subgroup"].str.extract(r"group (\d+)").astype(int)
    width = 0.36
    for offset, ami, color in [(-width / 2, "Yes", COLORS["mimic"]), (width / 2, "No", COLORS["eicu"])]:
        sub = d[d["ami"].eq(ami)].sort_values("group")
        x = sub["group"].to_numpy()
        y = sub["mortality_pct"].to_numpy()
        ax.bar(x + offset, y, width=width, color=color, label=("AMI-CS" if ami == "Yes" else "Non-AMI-CS"))
        label_offset = 6.5 if ami == "Yes" else 1.5
        for xi, yi in zip(x + offset, y):
            ax.text(xi, yi + label_offset, f"{yi:.1f}", ha="center", va="bottom", fontsize=8, color="#111111")
    ax.set_xticks([1, 2, 3, 4], [f"G{i}" for i in range(1, 5)])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mortality (%)")
    ax.set_title("AMI-CS mortality", fontweight="bold", pad=5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)


def ami_forest(ax: plt.Axes, df: pd.DataFrame, label: str) -> None:
    panel_label(ax, label, x=-0.12)
    d = df[df["subgroup_definition"].eq("AMI-CS by acute MI ICD")].copy()
    d["group"] = d["term"].str.extract(r"traj_(\d+)").astype(int)
    d = d.sort_values(["subgroup", "group"], ascending=[True, False])
    rows = []
    for _, row in d.iterrows():
        rows.append(
            {
                "label": f"{'AMI-CS' if row['subgroup'] == 'Yes' else 'Non-AMI-CS'}\nG{int(row['group'])} vs G1",
                "or": row["or"],
                "low": row["ci95_low"],
                "high": row["ci95_high"],
                "color": COLORS["mimic"] if row["subgroup"] == "Yes" else COLORS["eicu"],
            }
        )
    y = np.arange(len(rows))[::-1]
    for yy, row in zip(y, rows):
        ax.errorbar(
            row["or"],
            yy,
            xerr=[[row["or"] - row["low"]], [row["high"] - row["or"]]],
            fmt="o",
            color=row["color"],
            ecolor=row["color"],
            capsize=2.5,
            elinewidth=1.1,
            markersize=4,
        )
    ax.axvline(1, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.45, 55)
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=7.5)
    ax.tick_params(axis="y", pad=2)
    ax.set_xlabel("Adjusted odds ratio")
    ax.set_title("AMI-CS adjusted ORs", fontweight="bold", pad=5)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6, which="both")


def figure5() -> None:
    sens = pd.read_csv(TABLES / "table_supplementary_sensitivity_trajectory.csv")
    landmark = pd.read_csv(TABLES / "table_q2_landmark_mortality.csv")
    ami = pd.read_csv(TABLES / "table_q2_ami_subgroup_mortality.csv")
    ami_or = pd.read_csv(TABLES / "table_q2_ami_subgroup_adjusted_or.csv")

    fig = plt.figure(figsize=(7.5, 5.5))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.10, top=0.91)
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.92)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    mortality_panel(axes[0], sens[sens["scenario"].eq("k3_min2")], "K = 3 solution", "A")
    mortality_panel(axes[1], sens[sens["scenario"].eq("k5_min2")], "K = 5 solution", "B")
    mortality_panel(axes[2], sens[sens["scenario"].eq("k4_min3")], "At least 3 lactates", "C")
    mortality_panel(axes[3], landmark, "ICU stay at least 24 h", "D")
    ami_mortality(axes[4], ami, "E")
    ami_forest(axes[5], ami_or, "F")
    save_pub(fig, "figure5_robustness_subgroup_composite")


def main() -> None:
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    print("Created PLOS-sized main figures 1-5 in manuscript_figures/.")


if __name__ == "__main__":
    main()




