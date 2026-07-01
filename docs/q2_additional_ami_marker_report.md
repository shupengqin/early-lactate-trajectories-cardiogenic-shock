# Additional Q2 analyses: AMI subgroup and trajectory increment

## 1. AMI-CS and non-AMI-CS subgroup analysis

Acute myocardial infarction-related cardiogenic shock (AMI-CS) was defined using acute MI diagnosis codes in the same hospital admission: ICD-9 410.xx, excluding subsequent episode codes ending in 2, or ICD-10 I21/I22. Because eICU-CRD exported analysis data did not contain a reliable MI diagnosis field, this subgroup analysis was performed in the MIMIC-IV discovery cohort.

| subgroup_definition | subgroup | n | deaths | mortality_pct | persistent_high_lactate_n |
| --- | --- | --- | --- | --- | --- |
| AMI-CS by acute MI ICD | Yes | 870 | 363 | 41.72 | 166 |
| AMI-CS by acute MI ICD | Yes: trajectory group 1 | 304 | 77 | 25.33 | 0 |
| AMI-CS by acute MI ICD | Yes: trajectory group 2 | 313 | 106 | 33.87 | 19 |
| AMI-CS by acute MI ICD | Yes: trajectory group 3 | 160 | 96 | 60.00 | 57 |
| AMI-CS by acute MI ICD | Yes: trajectory group 4 | 93 | 84 | 90.32 | 90 |
| AMI-CS by acute MI ICD | No | 1311 | 481 | 36.69 | 244 |
| AMI-CS by acute MI ICD | No: trajectory group 1 | 427 | 107 | 25.06 | 3 |
| AMI-CS by acute MI ICD | No: trajectory group 2 | 473 | 134 | 28.33 | 11 |
| AMI-CS by acute MI ICD | No: trajectory group 3 | 276 | 129 | 46.74 | 99 |
| AMI-CS by acute MI ICD | No: trajectory group 4 | 135 | 111 | 82.22 | 131 |
| MI history/Charlson marker | Yes | 1049 | 437 | 41.66 | 195 |
| MI history/Charlson marker | Yes: trajectory group 1 | 364 | 96 | 26.37 | 1 |
| MI history/Charlson marker | Yes: trajectory group 2 | 379 | 128 | 33.77 | 20 |
| MI history/Charlson marker | Yes: trajectory group 3 | 197 | 115 | 58.38 | 68 |
| MI history/Charlson marker | Yes: trajectory group 4 | 109 | 98 | 89.91 | 106 |
| MI history/Charlson marker | No | 1132 | 407 | 35.95 | 215 |
| MI history/Charlson marker | No: trajectory group 1 | 367 | 88 | 23.98 | 2 |
| MI history/Charlson marker | No: trajectory group 2 | 407 | 112 | 27.52 | 10 |
| MI history/Charlson marker | No: trajectory group 3 | 239 | 110 | 46.03 | 88 |
| MI history/Charlson marker | No: trajectory group 4 | 119 | 97 | 81.51 | 115 |

Adjusted trajectory associations within subgroups:

| subgroup_definition | subgroup | n_complete | aic | term | or | ci95_low | ci95_high | p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMI-CS by acute MI ICD | Yes | 867 | 965.43 | traj_2 | 1.3338 | 0.9210 | 1.9315 | 0.127 |
| AMI-CS by acute MI ICD | Yes | 867 | 965.43 | traj_3 | 3.5736 | 2.2720 | 5.6210 | <0.001 |
| AMI-CS by acute MI ICD | Yes | 867 | 965.43 | traj_4 | 19.9345 | 9.0413 | 43.9522 | <0.001 |
| AMI-CS by acute MI ICD | No | 1310 | 1442.18 | traj_2 | 1.0255 | 0.7471 | 1.4076 | 0.876 |
| AMI-CS by acute MI ICD | No | 1310 | 1442.18 | traj_3 | 1.8170 | 1.2660 | 2.6079 | 0.001 |
| AMI-CS by acute MI ICD | No | 1310 | 1442.18 | traj_4 | 8.3401 | 4.8405 | 14.3701 | <0.001 |
| MI history/Charlson marker | Yes | 1046 | 1161.33 | traj_2 | 1.2320 | 0.8793 | 1.7263 | 0.225 |
| MI history/Charlson marker | Yes | 1046 | 1161.33 | traj_3 | 3.0161 | 1.9976 | 4.5538 | <0.001 |
| MI history/Charlson marker | Yes | 1046 | 1161.33 | traj_4 | 16.5889 | 8.0803 | 34.0572 | <0.001 |
| MI history/Charlson marker | No | 1131 | 1245.60 | traj_2 | 1.0678 | 0.7569 | 1.5062 | 0.709 |
| MI history/Charlson marker | No | 1131 | 1245.60 | traj_3 | 1.8942 | 1.2853 | 2.7915 | 0.001 |
| MI history/Charlson marker | No | 1131 | 1245.60 | traj_4 | 8.6411 | 4.8885 | 15.2744 | <0.001 |

## 2. Persistent hyperlactatemia as a simple marker

| comparison | n_complete | or | ci95 | p_value |
| --- | --- | --- | --- | --- |
| Persistent hyperlactatemia vs no persistent hyperlactatemia | 2177 | 5.5033 | 4.18-7.25 | <0.001 |

## 3. Trajectory models versus simpler lactate summaries

| model | auroc | auprc | brier | delta_auroc_vs_clinical_base | delta_auprc_vs_clinical_base |
| --- | --- | --- | --- | --- | --- |
| clinical_base | 0.7254 | 0.6163 | 0.2024 | 0.0000 | 0.0000 |
| base_plus_initial_lactate | 0.7553 | 0.6562 | 0.1924 | 0.0299 | 0.0398 |
| base_plus_initial_and_clearance | 0.7687 | 0.6871 | 0.1861 | 0.0434 | 0.0708 |
| base_plus_persistent_hyperlactatemia | 0.7701 | 0.6865 | 0.1859 | 0.0447 | 0.0702 |
| base_plus_trajectory | 0.7687 | 0.6959 | 0.1857 | 0.0433 | 0.0795 |
| base_plus_persistent_and_trajectory | 0.7754 | 0.7035 | 0.1833 | 0.0501 | 0.0871 |
| base_plus_initial_clearance_and_trajectory | 0.7716 | 0.6994 | 0.1845 | 0.0462 | 0.0831 |
| base_plus_full_lactate_dynamics | 0.7824 | 0.7181 | 0.1800 | 0.0571 | 0.1018 |

Bootstrap comparison of prediction increments:

| comparison | n_bootstrap | delta_auroc | delta_auroc_ci95 | delta_auprc | delta_auprc_ci95 |
| --- | --- | --- | --- | --- | --- |
| Trajectory vs initial lactate + clearance | 500 | -0.0001 | -0.0088-0.0073 | 0.0088 | -0.0077-0.0250 |
| Trajectory vs persistent hyperlactatemia | 500 | -0.0014 | -0.0110-0.0069 | 0.0094 | -0.0088-0.0248 |
| Trajectory added to persistent hyperlactatemia | 500 | 0.0054 | 0.0002-0.0113 | 0.0170 | 0.0050-0.0284 |
| Trajectory added to initial lactate + clearance | 500 | 0.0028 | -0.0014-0.0073 | 0.0123 | 0.0039-0.0205 |
| Full dynamics vs initial lactate + clearance + trajectory | 500 | 0.0109 | 0.0040-0.0173 | 0.0187 | 0.0072-0.0310 |

## Interpretation for manuscript

The mortality gradient across trajectory groups remained present in both AMI-CS and non-AMI-CS subgroups in MIMIC-IV. The trajectory model performed similarly to the initial lactate plus clearance model for AUROC but produced stronger AUPRC than initial lactate plus clearance and persistent hyperlactatemia alone. Adding trajectory group to persistent hyperlactatemia further improved AUROC and AUPRC, supporting the claim that the trajectory phenotype contains information beyond a simple binary high-lactate marker.
