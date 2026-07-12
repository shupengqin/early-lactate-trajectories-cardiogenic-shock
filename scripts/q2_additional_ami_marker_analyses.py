from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prediction_utils import restrict_to_24h_survivors

ROOT = Path('.')
OUT = ROOT / 'outputs'
TABLES = ROOT / 'manuscript_tables'

BASE_FEATURES = [
    'age', 'sofa', 'charlson_comorbidity_index', 'mbp_mean',
    'creatinine_max', 'mechvent_24h', 'vasoactive_24h'
]

MODEL_FEATURE_SETS = {
    'clinical_base': BASE_FEATURES,
    'base_plus_initial_lactate': BASE_FEATURES + ['initial_lactate_24h'],
    'base_plus_initial_and_clearance': BASE_FEATURES + ['initial_lactate_24h', 'lactate_clearance_24h'],
    'base_plus_persistent_hyperlactatemia': BASE_FEATURES + ['persistent_high_lactate_24h'],
    'base_plus_trajectory': BASE_FEATURES + ['trajectory_group'],
    'base_plus_persistent_and_trajectory': BASE_FEATURES + ['persistent_high_lactate_24h', 'trajectory_group'],
    'base_plus_initial_clearance_and_trajectory': BASE_FEATURES + ['initial_lactate_24h', 'lactate_clearance_24h', 'trajectory_group'],
    'base_plus_full_lactate_dynamics': BASE_FEATURES + [
        'initial_lactate_24h', 'last_lactate_24h', 'peak_lactate_24h',
        'lactate_slope_24h', 'lactate_clearance_24h', 'persistent_high_lactate_24h', 'trajectory_group'
    ],
}


def fmt_p(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'


def preprocessor(features):
    categorical = [c for c in features if c in ['trajectory_group', 'gender']]
    numeric = [c for c in features if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric))
    if categorical:
        transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical))
    return ColumnTransformer(transformers)


def cv_predict(df, feature_sets):
    y = df['hospital_expire_flag'].astype(int).to_numpy()
    preds = pd.DataFrame({'stay_id': df['stay_id'], 'hospital_expire_flag': y})
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    for name, features in feature_sets.items():
        pipe = Pipeline([
            ('prep', preprocessor(features)),
            ('model', LogisticRegression(max_iter=3000)),
        ])
        preds[name] = cross_val_predict(pipe, df[features], y, cv=cv, method='predict_proba')[:, 1]
    return preds


def performance_table(preds):
    y = preds['hospital_expire_flag'].to_numpy()
    base_auc = roc_auc_score(y, preds['clinical_base'])
    base_pr = average_precision_score(y, preds['clinical_base'])
    rows = []
    for col in [c for c in preds.columns if c not in ['stay_id', 'hospital_expire_flag']]:
        auc = roc_auc_score(y, preds[col])
        pr = average_precision_score(y, preds[col])
        rows.append({
            'model': col,
            'auroc': auc,
            'auprc': pr,
            'brier': brier_score_loss(y, preds[col]),
            'delta_auroc_vs_clinical_base': auc - base_auc,
            'delta_auprc_vs_clinical_base': pr - base_pr,
        })
    return pd.DataFrame(rows)


def bootstrap_model_deltas(preds, comparisons, n_boot=500, seed=2026):
    rng = np.random.default_rng(seed)
    y = preds['hospital_expire_flag'].to_numpy()
    rows = []
    for model, reference, label in comparisons:
        da, dp = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, len(preds), len(preds))
            if len(np.unique(y[idx])) < 2:
                continue
            da.append(roc_auc_score(y[idx], preds[model].to_numpy()[idx]) - roc_auc_score(y[idx], preds[reference].to_numpy()[idx]))
            dp.append(average_precision_score(y[idx], preds[model].to_numpy()[idx]) - average_precision_score(y[idx], preds[reference].to_numpy()[idx]))
        rows.append({
            'comparison': label,
            'n_bootstrap': len(da),
            'delta_auroc': roc_auc_score(y, preds[model]) - roc_auc_score(y, preds[reference]),
            'delta_auroc_ci95': f'{np.percentile(da, 2.5):.4f}-{np.percentile(da, 97.5):.4f}',
            'delta_auprc': average_precision_score(y, preds[model]) - average_precision_score(y, preds[reference]),
            'delta_auprc_ci95': f'{np.percentile(dp, 2.5):.4f}-{np.percentile(dp, 97.5):.4f}',
        })
    return pd.DataFrame(rows)


def fit_trajectory_or(df, covariates):
    model_df = df[['hospital_expire_flag', 'trajectory_group'] + covariates].dropna().copy()
    dummies = pd.get_dummies(model_df['trajectory_group'].astype(int), prefix='traj', drop_first=True, dtype=float)
    x = pd.concat([dummies, model_df[covariates].astype(float)], axis=1)
    x = sm.add_constant(x, has_constant='add')
    y = model_df['hospital_expire_flag'].astype(float)
    try:
        fit = sm.Logit(y, x).fit(disp=False, maxiter=200)
        conf = fit.conf_int()
        terms = []
        for term in dummies.columns:
            terms.append({
                'term': term,
                'or': math.exp(fit.params[term]),
                'ci95_low': math.exp(conf.loc[term, 0]),
                'ci95_high': math.exp(conf.loc[term, 1]),
                'p_value': fit.pvalues[term],
            })
        return terms, int(len(model_df)), float(fit.aic)
    except Exception as exc:
        return [{'term': 'model_failed', 'or': np.nan, 'ci95_low': np.nan, 'ci95_high': np.nan, 'p_value': np.nan, 'error': str(exc)}], int(len(model_df)), np.nan


def subgroup_tables(df):
    covars = BASE_FEATURES
    summary_rows = []
    or_rows = []
    for subgroup_col, subgroup_name in [('acute_mi_cs', 'AMI-CS by acute MI ICD'), ('myocardial_infarct', 'MI history/Charlson marker')]:
        for value, label in [(1, 'Yes'), (0, 'No')]:
            sub = df[df[subgroup_col] == value].copy()
            deaths = int(sub['hospital_expire_flag'].sum())
            summary_rows.append({
                'subgroup_definition': subgroup_name,
                'subgroup': label,
                'n': int(len(sub)),
                'deaths': deaths,
                'mortality_pct': deaths / len(sub) * 100 if len(sub) else np.nan,
                'persistent_high_lactate_n': int(sub['persistent_high_lactate_24h'].sum()),
            })
            for group, gdf in sub.groupby('trajectory_group'):
                gd = int(gdf['hospital_expire_flag'].sum())
                summary_rows.append({
                    'subgroup_definition': subgroup_name,
                    'subgroup': f'{label}: trajectory group {int(group)}',
                    'n': int(len(gdf)),
                    'deaths': gd,
                    'mortality_pct': gd / len(gdf) * 100,
                    'persistent_high_lactate_n': int(gdf['persistent_high_lactate_24h'].sum()),
                })
            terms, n_complete, aic = fit_trajectory_or(sub, covars)
            for t in terms:
                or_rows.append({
                    'subgroup_definition': subgroup_name,
                    'subgroup': label,
                    'n_complete': n_complete,
                    'aic': aic,
                    **t,
                })
    return pd.DataFrame(summary_rows), pd.DataFrame(or_rows)


def persistent_or_table(df):
    covars = BASE_FEATURES
    model_df = df[['hospital_expire_flag', 'persistent_high_lactate_24h'] + covars].dropna().copy()
    x = sm.add_constant(model_df[['persistent_high_lactate_24h'] + covars].astype(float), has_constant='add')
    y = model_df['hospital_expire_flag'].astype(float)
    fit = sm.Logit(y, x).fit(disp=False, maxiter=200)
    conf = fit.conf_int()
    return pd.DataFrame([{
        'comparison': 'Persistent hyperlactatemia vs no persistent hyperlactatemia',
        'n_complete': int(len(model_df)),
        'or': math.exp(fit.params['persistent_high_lactate_24h']),
        'ci95': f"{math.exp(conf.loc['persistent_high_lactate_24h',0]):.2f}-{math.exp(conf.loc['persistent_high_lactate_24h',1]):.2f}",
        'p_value': fit.pvalues['persistent_high_lactate_24h'],
    }])


def markdown_table(df):
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind in 'fc':
            if 'p_value' in col:
                out[col] = out[col].map(fmt_p)
            elif 'mortality_pct' in col:
                out[col] = out[col].map(lambda x: '' if pd.isna(x) else f'{x:.2f}')
            elif col in ['or', 'ci95_low', 'ci95_high', 'delta_auroc', 'delta_auprc', 'auroc', 'auprc', 'brier', 'delta_auroc_vs_clinical_base', 'delta_auprc_vs_clinical_base']:
                out[col] = out[col].map(lambda x: '' if pd.isna(x) else f'{x:.4f}')
            else:
                out[col] = out[col].map(lambda x: '' if pd.isna(x) else f'{x:.2f}')
    lines = ['| ' + ' | '.join(out.columns) + ' |', '| ' + ' | '.join(['---'] * len(out.columns)) + ' |']
    for _, row in out.iterrows():
        lines.append('| ' + ' | '.join(str(row[c]) for c in out.columns) + ' |')
    return '\n'.join(lines)


def main():
    TABLES.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(OUT / 'mimic_analysis_dataset_with_trajectory.csv')
    ami = pd.read_csv(OUT / 'mimic_acute_mi_diagnoses.csv')
    ami['acute_mi_dx'] = 1
    df = df.merge(ami[['subject_id', 'hadm_id', 'acute_mi_dx']].drop_duplicates(), on=['subject_id', 'hadm_id'], how='left')
    df['acute_mi_cs'] = df['acute_mi_dx'].fillna(0).astype(int)

    prediction_df = restrict_to_24h_survivors(
        df, OUT / 'mimic_cs_lactate_24h_cohort.csv'
    )
    subgroup_summary, subgroup_or = subgroup_tables(prediction_df)
    persistent_or = persistent_or_table(prediction_df)
    preds = cv_predict(prediction_df, MODEL_FEATURE_SETS)
    perf = performance_table(preds)
    comparisons = [
        ('base_plus_trajectory', 'base_plus_initial_and_clearance', 'Trajectory vs initial lactate + clearance'),
        ('base_plus_trajectory', 'base_plus_persistent_hyperlactatemia', 'Trajectory vs persistent hyperlactatemia'),
        ('base_plus_persistent_and_trajectory', 'base_plus_persistent_hyperlactatemia', 'Trajectory added to persistent hyperlactatemia'),
        ('base_plus_initial_clearance_and_trajectory', 'base_plus_initial_and_clearance', 'Trajectory added to initial lactate + clearance'),
        ('base_plus_full_lactate_dynamics', 'base_plus_initial_clearance_and_trajectory', 'Full dynamics vs initial lactate + clearance + trajectory'),
    ]
    boot = bootstrap_model_deltas(preds, comparisons, n_boot=500)

    subgroup_summary.to_csv(TABLES / 'table_q2_ami_subgroup_mortality.csv', index=False)
    subgroup_or.to_csv(TABLES / 'table_q2_ami_subgroup_adjusted_or.csv', index=False)
    persistent_or.to_csv(TABLES / 'table_q2_persistent_hyperlactatemia_adjusted_or.csv', index=False)
    perf.to_csv(TABLES / 'table_q2_lactate_marker_model_performance.csv', index=False)
    boot.to_csv(TABLES / 'table_q2_trajectory_increment_vs_simple_markers.csv', index=False)
    preds.to_csv(OUT / 'mimic_q2_marker_comparison_predictions.csv', index=False)

    report = []
    report.append('# Additional Q2 analyses: AMI subgroup and trajectory increment')
    report.append('')
    report.append('## 1. AMI-CS and non-AMI-CS subgroup analysis')
    report.append('')
    report.append('Acute myocardial infarction-related cardiogenic shock (AMI-CS) was defined using acute MI diagnosis codes in the same hospital admission: ICD-9 410.xx, excluding subsequent episode codes ending in 2, or ICD-10 I21/I22. The subgroup analysis used the MIMIC-IV 24-hour landmark risk set. Because eICU-CRD exported analysis data did not contain a reliable MI diagnosis field, this subgroup analysis could not be repeated in eICU-CRD.')
    report.append('')
    report.append(markdown_table(subgroup_summary))
    report.append('')
    report.append('Adjusted trajectory associations within subgroups:')
    report.append('')
    report.append(markdown_table(subgroup_or))
    report.append('')
    report.append('## 2. Persistent hyperlactatemia as a simple marker')
    report.append('')
    report.append(markdown_table(persistent_or))
    report.append('')
    report.append('## 3. Trajectory models versus simpler lactate summaries')
    report.append('')
    report.append(markdown_table(perf))
    report.append('')
    report.append('Bootstrap comparison of prediction increments:')
    report.append('')
    report.append(markdown_table(boot))
    report.append('')
    report.append('## Interpretation for manuscript')
    report.append('')
    report.append('The mortality gradient across trajectory groups remained present in both AMI-CS and non-AMI-CS subgroups in MIMIC-IV. In the 24-hour landmark prediction cohort, trajectory group performed similarly to initial lactate plus clearance and persistent hyperlactatemia. Bootstrap intervals did not show a clear improvement when trajectory group was added to either simpler summary.')
    report.append('')
    (ROOT / 'q2_additional_ami_marker_report.md').write_text('\n'.join(report), encoding='utf-8')

    results = {
        'acute_mi_cs_count': int(prediction_df['acute_mi_cs'].sum()),
        'non_acute_mi_cs_count': int((prediction_df['acute_mi_cs'] == 0).sum()),
        'outputs': {
            'subgroup_mortality': 'manuscript_tables/table_q2_ami_subgroup_mortality.csv',
            'subgroup_adjusted_or': 'manuscript_tables/table_q2_ami_subgroup_adjusted_or.csv',
            'persistent_or': 'manuscript_tables/table_q2_persistent_hyperlactatemia_adjusted_or.csv',
            'marker_performance': 'manuscript_tables/table_q2_lactate_marker_model_performance.csv',
            'trajectory_increment': 'manuscript_tables/table_q2_trajectory_increment_vs_simple_markers.csv',
            'report': 'q2_additional_ami_marker_report.md',
        },
    }
    (OUT / 'q2_additional_ami_marker_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
