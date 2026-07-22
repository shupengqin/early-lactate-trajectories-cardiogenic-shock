from pathlib import Path
import math
import pandas as pd
import numpy as np
from scipy import stats

ROOT = Path('.')
OUT = ROOT / 'manuscript_tables'
DATA = ROOT / 'outputs' / 'mimic_analysis_dataset_with_trajectory.csv'

CONTINUOUS = [
    ('age', 'Age, years'),
    ('sofa', 'SOFA score'),
    ('sapsii', 'SAPS II'),
    ('oasis', 'OASIS'),
    ('charlson_comorbidity_index', 'Charlson comorbidity index'),
    ('heart_rate_mean', 'Heart rate, mean'),
    ('mbp_mean', 'Mean arterial pressure, mean'),
    ('resp_rate_mean', 'Respiratory rate, mean'),
    ('spo2_mean', 'SpO2, mean'),
    ('hemoglobin_min', 'Hemoglobin, minimum'),
    ('platelets_min', 'Platelets, minimum'),
    ('wbc_max', 'White blood cells, maximum'),
    ('bicarbonate_min', 'Bicarbonate, minimum'),
    ('bun_max', 'BUN, maximum'),
    ('creatinine_max', 'Creatinine, maximum'),
    ('sodium_min', 'Sodium, minimum'),
    ('potassium_max', 'Potassium, maximum'),
    ('initial_lactate_24h', 'Initial lactate'),
    ('last_lactate_24h', 'Last lactate within 24 h'),
    ('peak_lactate_24h', 'Peak lactate within 24 h'),
]

CATEGORICAL = [
    ('gender', 'Female sex', 'F'),
    ('myocardial_infarct', 'Myocardial infarction', 1),
    ('congestive_heart_failure', 'Congestive heart failure', 1),
    ('peripheral_vascular_disease', 'Peripheral vascular disease', 1),
    ('chronic_pulmonary_disease', 'Chronic pulmonary disease', 1),
    ('diabetes_without_cc', 'Diabetes without complications', 1),
    ('diabetes_with_cc', 'Diabetes with complications', 1),
    ('renal_disease', 'Renal disease', 1),
    ('malignant_cancer', 'Malignant cancer', 1),
    ('mechvent_24h', 'Mechanical ventilation within 24 h', 1),
    ('vasoactive_24h', 'Vasoactive agent within 24 h', 1),
    ('hospital_expire_flag', 'In-hospital mortality', 1),
]


def fmt_p(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'


def fmt_float(x, digits=2):
    if pd.isna(x):
        return ''
    return f'{x:.{digits}f}'


def fmt_cont(series):
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        return ''
    return f'{s.median():.1f} [{s.quantile(0.25):.1f}, {s.quantile(0.75):.1f}]'


def fmt_cat(series, positive):
    valid = series.dropna()
    n = len(valid)
    if n == 0:
        return ''
    count = (valid == positive).sum()
    return f'{int(count)} ({count / n * 100:.1f})'


def max_pairwise_smd_cont(df, col, groups):
    vals = []
    for i, g1 in enumerate(groups):
        x1 = pd.to_numeric(df.loc[df.trajectory_group == g1, col], errors='coerce').dropna()
        for g2 in groups[i+1:]:
            x2 = pd.to_numeric(df.loc[df.trajectory_group == g2, col], errors='coerce').dropna()
            if len(x1) < 2 or len(x2) < 2:
                continue
            pooled = math.sqrt((x1.var(ddof=1) + x2.var(ddof=1)) / 2)
            if pooled > 0:
                vals.append(abs(x1.mean() - x2.mean()) / pooled)
    return max(vals) if vals else np.nan


def max_pairwise_smd_cat(df, col, positive, groups):
    vals = []
    for i, g1 in enumerate(groups):
        s1 = df.loc[df.trajectory_group == g1, col].dropna()
        p1 = (s1 == positive).mean() if len(s1) else np.nan
        for g2 in groups[i+1:]:
            s2 = df.loc[df.trajectory_group == g2, col].dropna()
            p2 = (s2 == positive).mean() if len(s2) else np.nan
            if pd.isna(p1) or pd.isna(p2):
                continue
            p_pool = (p1 + p2) / 2
            denom = math.sqrt(p_pool * (1 - p_pool))
            if denom > 0:
                vals.append(abs(p1 - p2) / denom)
    return max(vals) if vals else np.nan


def markdown_table(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for _, row in df.iterrows():
        vals = [str(row[c]) if not pd.isna(row[c]) else '' for c in cols]
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def make_table1():
    df = pd.read_csv(DATA)
    groups = sorted(df['trajectory_group'].dropna().astype(int).unique())
    rows = []
    rows.append({'Characteristic': 'N', 'Overall': str(len(df)), **{f'Group {g}': str(int((df.trajectory_group == g).sum())) for g in groups}, 'P value': '', 'Max SMD': ''})

    for col, label in CONTINUOUS:
        samples = [pd.to_numeric(df.loc[df.trajectory_group == g, col], errors='coerce').dropna() for g in groups]
        valid_samples = [s for s in samples if len(s) > 0]
        p = stats.kruskal(*valid_samples).pvalue if len(valid_samples) >= 2 else np.nan
        row = {'Characteristic': label, 'Overall': fmt_cont(df[col]), 'P value': fmt_p(p), 'Max SMD': fmt_float(max_pairwise_smd_cont(df, col, groups))}
        for g in groups:
            row[f'Group {g}'] = fmt_cont(df.loc[df.trajectory_group == g, col])
        rows.append(row)

    for col, label, positive in CATEGORICAL:
        contingency = []
        for g in groups:
            s = df.loc[df.trajectory_group == g, col].dropna()
            contingency.append([(s == positive).sum(), (s != positive).sum()])
        try:
            p = stats.chi2_contingency(contingency).pvalue
        except Exception:
            p = np.nan
        row = {'Characteristic': label, 'Overall': fmt_cat(df[col], positive), 'P value': fmt_p(p), 'Max SMD': fmt_float(max_pairwise_smd_cat(df, col, positive, groups))}
        for g in groups:
            row[f'Group {g}'] = fmt_cat(df.loc[df.trajectory_group == g, col], positive)
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out[['Characteristic', 'Overall', 'Group 1', 'Group 2', 'Group 3', 'Group 4', 'P value', 'Max SMD']]
    out.to_csv(OUT / 'table1_mimic_baseline_by_trajectory_journal.csv', index=False)
    return out


def format_trajectory(path, cohort_label):
    df = pd.read_csv(path).copy()
    out = pd.DataFrame({
        'Cohort': cohort_label,
        'Group': df['trajectory_group'].map(lambda x: f'Group {int(x)}'),
        'N': df['n'].astype(int),
        'Mean lactate 0-6 h': df['mean_0_6h'].map(lambda x: f'{x:.2f}'),
        'Mean lactate 6-12 h': df['mean_6_12h'].map(lambda x: f'{x:.2f}'),
        'Mean lactate 12-18 h': df['mean_12_18h'].map(lambda x: f'{x:.2f}'),
        'Mean lactate 18-24 h': df['mean_18_24h'].map(lambda x: f'{x:.2f}'),
        'Deaths': df['deaths'].astype(int),
        'Mortality, %': df['mortality_pct'].map(lambda x: f'{x:.2f}'),
    })
    return out


def make_other_tables():
    traj = pd.concat([
        format_trajectory(OUT / 'table_mimic_trajectory_groups.csv', 'MIMIC-IV'),
        format_trajectory(OUT / 'table_eicu_trajectory_validation.csv', 'eICU-CRD'),
    ], ignore_index=True)
    traj.to_csv(OUT / 'table2_trajectory_groups_journal.csv', index=False)

    landmark = pd.read_csv(OUT / 'table_primary_24h_landmark_associations.csv')
    absolute = pd.read_csv(OUT / 'table_primary_adjusted_absolute_effects.csv')

    def effect_table(cohort):
        counts = landmark[landmark['cohort'].eq(cohort)][
            ['trajectory_group', 'n', 'deaths', 'mortality_pct']
        ]
        effects = absolute[absolute['cohort'].eq(cohort)].copy()
        merged = counts.merge(effects, on='trajectory_group', suffixes=('_landmark', '_model'))
        return pd.DataFrame({
            'Group': merged['trajectory_group'].map(lambda x: f'Group {int(x)}'),
            'N at landmark': merged['n_landmark'].astype(int),
            'Deaths': merged['deaths_landmark'].astype(int),
            'Model included, n': merged['n_model'].astype(int),
            'Excluded for missing covariates, n': (merged['n_landmark'] - merged['n_model']).astype(int),
            'Adjusted mortality, % (95% CI)': merged.apply(
                lambda r: f"{r['adjusted_mortality_pct']:.1f} ({r['adjusted_mortality_ci95_low']:.1f}-{r['adjusted_mortality_ci95_high']:.1f})",
                axis=1,
            ),
            'Risk difference per 100 (95% CI)': merged.apply(
                lambda r: 'Reference' if int(r['trajectory_group']) == 1 else
                f"{r['adjusted_risk_difference_per_100']:.1f} ({r['risk_difference_ci95_low']:.1f}-{r['risk_difference_ci95_high']:.1f})",
                axis=1,
            ),
            'Adjusted RR (95% CI)': merged.apply(
                lambda r: 'Reference' if int(r['trajectory_group']) == 1 else
                f"{r['modified_poisson_rr']:.2f} ({r['rr_ci95_low']:.2f}-{r['rr_ci95_high']:.2f})",
                axis=1,
            ),
            'Adjusted OR (95% CI)': merged.apply(
                lambda r: 'Reference' if int(r['trajectory_group']) == 1 else
                f"{r['adjusted_or']:.2f} ({r['or_ci95_low']:.2f}-{r['or_ci95_high']:.2f})",
                axis=1,
            ),
            'P value': merged.apply(
                lambda r: '' if int(r['trajectory_group']) == 1 else fmt_p(r['or_p_value']),
                axis=1,
            ),
        })

    reg_out = effect_table('MIMIC-IV')
    reg_out.to_csv(OUT / 'table3_adjusted_or_journal.csv', index=False)

    table5 = pd.read_csv(OUT / 'table5_eicu_fixed_centroid_transportability_journal.csv')
    fixed_external = pd.read_csv(
        OUT / 'table_eicu_mimic_centroid_24h_landmark_adjusted_associations_raw.csv'
    )
    fixed_external = fixed_external[
        fixed_external['assignment_method'].eq('Fixed MIMIC-IV centroids (primary)')
    ].copy()
    fixed_supplementary = pd.DataFrame({
        'Group': fixed_external['trajectory_group'].map(lambda x: f'Group {int(x)}'),
        'N at landmark': fixed_external['n_landmark'].astype(int),
        'Deaths': fixed_external['deaths_landmark'].astype(int),
        'Model included, n': fixed_external['n'].astype(int),
        'Excluded for missing covariates, n': (
            fixed_external['n_landmark'] - fixed_external['n']
        ).astype(int),
        'Adjusted mortality, % (95% CI)': fixed_external.apply(
            lambda r: f"{r['adjusted_mortality_pct']:.1f} ({r['adjusted_mortality_ci95_low']:.1f}-{r['adjusted_mortality_ci95_high']:.1f})",
            axis=1,
        ),
        'Risk difference per 100 (95% CI)': fixed_external.apply(
            lambda r: 'Reference' if int(r['trajectory_group']) == 1 else
            f"{r['adjusted_risk_difference_per_100']:.1f} ({r['risk_difference_ci95_low']:.1f}-{r['risk_difference_ci95_high']:.1f})",
            axis=1,
        ),
        'Adjusted RR (95% CI)': fixed_external.apply(
            lambda r: 'Reference' if int(r['trajectory_group']) == 1 else
            f"{r['modified_poisson_rr']:.2f} ({r['rr_ci95_low']:.2f}-{r['rr_ci95_high']:.2f})",
            axis=1,
        ),
        'Adjusted OR (95% CI)': fixed_external.apply(
            lambda r: 'Reference' if int(r['trajectory_group']) == 1 else
            f"{r['adjusted_or']:.2f} ({r['or_ci95_low']:.2f}-{r['or_ci95_high']:.2f})",
            axis=1,
        ),
    })
    supplementary_absolute = pd.concat(
        [
            reg_out.assign(Cohort='MIMIC-IV'),
            fixed_supplementary.assign(Cohort='eICU-CRD fixed MIMIC-IV centroids'),
        ],
        ignore_index=True,
    )
    supplementary_absolute = supplementary_absolute[
        [
            'Cohort',
            'Group',
            'N at landmark',
            'Deaths',
            'Model included, n',
            'Excluded for missing covariates, n',
            'Adjusted mortality, % (95% CI)',
            'Risk difference per 100 (95% CI)',
            'Adjusted RR (95% CI)',
            'Adjusted OR (95% CI)',
        ]
    ]
    supplementary_absolute.to_csv(
        OUT / 'table_supplementary_adjusted_effects_formatted.csv', index=False
    )

    # The foldwise table is the primary prediction analysis. The older
    # table_prediction_performance.csv contains global trajectory labels and
    # is retained only as a sensitivity analysis.
    pred = pd.read_csv(OUT / 'table_q2_prediction_performance.csv')
    order = [
        'Clinical base model',
        'Clinical base + initial lactate',
        'Clinical base + initial lactate + clearance',
        'Clinical base + trajectory group',
        'Clinical base + full lactate dynamics',
    ]
    pred = pred.set_index('Model').loc[order].reset_index()
    metric_columns = [
        'AUROC',
        'AUPRC',
        'Brier score',
        'Calibration intercept',
        'Calibration slope',
        'Delta AUROC vs base',
        'Delta AUPRC vs base',
    ]
    pred_out = pred[['Model', *metric_columns]].copy()
    for column in metric_columns:
        pred_out[column] = pd.to_numeric(pred_out[column], errors='raise').map(
            lambda x: f'{x:.3f}'
        )
    pred_out.to_csv(OUT / 'table4_prediction_performance_journal.csv', index=False)

    boot = pd.read_csv(OUT / 'table_q2_bootstrap_prediction_deltas.csv')
    boot_out = boot.copy()
    boot_out.columns = [c.replace('_', ' ').title() for c in boot_out.columns]
    boot_out.to_csv(OUT / 'table_supp_bootstrap_prediction_deltas_journal.csv', index=False)
    return traj, reg_out, pred_out, boot_out


def main():
    OUT.mkdir(exist_ok=True)
    table1 = make_table1()
    traj, reg, pred, boot = make_other_tables()
    md = []
    md.append('# Journal-ready tables')
    md.append('')
    md.append('Continuous variables are shown as median [interquartile range], and categorical variables as n (%). P values for continuous variables were calculated using Kruskal-Wallis tests; categorical variables used chi-square tests. Max SMD is the maximum pairwise standardized mean difference across trajectory groups.')
    md.append('')
    md.append('## Table 1. Clinical characteristics in MIMIC-IV by lactate trajectory group')
    md.append(markdown_table(table1))
    md.append('')
    md.append('## Table 2. Lactate trajectory groups in MIMIC-IV and eICU-CRD')
    md.append(markdown_table(traj))
    md.append('')
    md.append('## Table 3. Adjusted association between lactate trajectories and in-hospital mortality')
    md.append(markdown_table(reg))
    md.append('')
    md.append('## Table 4. Prediction model performance in MIMIC-IV')
    md.append(markdown_table(pred))
    md.append('')
    md.append('## Supplementary Table. Bootstrap increments in prediction performance')
    md.append(markdown_table(boot))
    md.append('')
    (OUT / 'journal_ready_tables.md').write_text('\n'.join(md), encoding='utf-8')
    print('Generated journal-ready tables:')
    for name in [
        'table1_mimic_baseline_by_trajectory_journal.csv',
        'table2_trajectory_groups_journal.csv',
        'table3_adjusted_or_journal.csv',
        'table4_prediction_performance_journal.csv',
        'table_supp_bootstrap_prediction_deltas_journal.csv',
        'journal_ready_tables.md',
    ]:
        print(OUT / name)

if __name__ == '__main__':
    main()
