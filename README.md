# Early lactate trajectories in cardiogenic shock

This repository contains the SQL and Python analysis code for the manuscript:

**First-day lactate trajectories and subsequent in-hospital mortality in cardiogenic shock: a landmark multicohort study**

## Data access

This repository does not include patient-level data. The analyses require credentialed access to third-party datasets through PhysioNet:

- MIMIC-IV version 3.1: https://physionet.org/content/mimiciv/
- eICU Collaborative Research Database: https://physionet.org/content/eicu-crd/

Users must complete all required training and data use agreements before accessing these data. The authors are not permitted to redistribute MIMIC-IV or eICU-CRD data.

## Repository structure

```text
scripts/   Python scripts for cohort processing, trajectory modelling, prediction, validation, tables, and figures
sql/       SQL scripts for cohort construction and lactate extraction
docs/      Protocol and supplementary analysis reports
```

## Software environment

The analyses were run with Python 3.14.5 and PostgreSQL. Main Python packages:

```text
pandas 3.0.4
statsmodels 0.14.6
scikit-learn 1.9.0
scipy 1.18.0
matplotlib 3.11.0
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Reproducibility workflow

1. Load MIMIC-IV and eICU-CRD locally according to PhysioNet documentation.
2. Update database connection parameters and file paths for the local environment.
3. Run the SQL scripts in `sql/` to create cohort and lactate extraction tables.
4. Run the cohort and trajectory scripts, followed by the association, prediction, external-validation, and marker-comparison scripts.
5. Run `scripts/methodology_sensitivity_analyses.py` to construct the 24-hour landmark analyses, measurement-process audits, stability analyses, and initial foldwise trajectory predictions.
6. Run `scripts/promote_foldwise_prediction.py` to make foldwise trajectory fitting the primary prediction pipeline and regenerate its bootstrap intervals, calibration data, decision curves, simple-marker comparisons, and prespecified algorithm comparisons.
7. Run the table-generation scripts and then `scripts/create_composite_main_figures.py` to reproduce the five final composite figures.
8. Keep generated patient-level intermediate files under the applicable PhysioNet data use agreements; do not commit them to a public repository.

The primary association and prediction analyses use a 24-hour landmark and include patients alive and still hospitalized at that time. For every prediction model containing trajectory group, lactate-window filling, scaling, K-means fitting, trajectory assignment, and model fitting are repeated within each cross-validation training fold. The globally assigned trajectory-label analysis is retained only as a sensitivity analysis. eICU-CRD models explicitly use APACHE IVa and hospital-clustered standard errors.

## Citation

Please cite the associated manuscript when using this code.

## License

The code is released under the MIT License. This license applies only to the analysis code, not to MIMIC-IV, eICU-CRD, or other third-party datasets.
