# Early lactate trajectories in cardiogenic shock

This repository contains the SQL and Python analysis code for the manuscript:

**Early lactate trajectories and in-hospital mortality in ICU patients with cardiogenic shock: a multicohort study using MIMIC-IV and eICU-CRD**

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
4. Run the Python scripts in `scripts/` following the protocol in `docs/mimic_cardiogenic_shock_lactate_protocol.md`.
5. Keep generated patient-level intermediate files under the applicable PhysioNet data use agreements.

## Citation

Repository URL: https://github.com/shupengqin/early-lactate-trajectories-cardiogenic-shock

Please cite the associated manuscript when using this code.

## License

The code is released under the MIT License. This license applies only to the analysis code, not to MIMIC-IV, eICU-CRD, or other third-party datasets.

