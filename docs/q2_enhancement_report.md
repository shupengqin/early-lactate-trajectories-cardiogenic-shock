# Q2-oriented enhancement analysis report

## Purpose

This report documents additional analyses added to strengthen the manuscript for submission to a Chinese Academy of Sciences Q2-level journal. The additions address common reviewer concerns in retrospective ICU prediction studies: external validation rigor, possible 24-hour observation-window bias, uncertainty around incremental discrimination, calibration, and clinical net benefit.

## 1. External validation using MIMIC-derived trajectory centroids

The original eICU validation repeated K-means clustering within eICU-CRD. To make the validation stricter, the MIMIC-IV trajectory model was used as the reference model. Lactate values in eICU-CRD were assigned to the nearest MIMIC-derived centroid after applying the same log1p transformation and MIMIC-derived standardization parameters.

Validation cohort:

```text
N = 476
Deaths = 226
Mortality = 47.48%
```

Mortality by MIMIC-centroid trajectory group:

| Group | N | Deaths | Mortality |
|---:|---:|---:|---:|
| 1 | 103 | 26 | 25.24% |
| 2 | 136 | 51 | 37.50% |
| 3 | 143 | 69 | 48.25% |
| 4 | 94 | 80 | 85.11% |

APACHE-adjusted association:

| Comparison | OR | 95% CI | P |
|---|---:|---:|---:|
| Group 2 vs Group 1 | 1.65 | 0.86-3.17 | 0.134 |
| Group 3 vs Group 1 | 1.79 | 0.92-3.46 | 0.084 |
| Group 4 vs Group 1 | 12.10 | 5.11-28.62 | 1.39e-08 |

Interpretation: The highest-risk trajectory remained strongly associated with in-hospital mortality when eICU patients were assigned using MIMIC-derived centroids, supporting transportability of the persistent-high phenotype.

## 2. Observation-window sensitivity analysis

Because trajectories were defined using the first 24 hours after ICU admission, a sensitivity analysis excluded patients with ICU length of stay shorter than 24 hours. This is a pragmatic observation-window analysis based on available exported variables; it is not a perfect death-time landmark because exact hospital death time was not included in the current exported MIMIC analysis table.

Sensitivity cohort:

```text
Original trajectory cohort: N = 2,181
Excluded ICU LOS <24 h: N = 224
Sensitivity cohort ICU LOS >=24 h: N = 1,957
Deaths: N = 683
Mortality = 34.90%
```

Mortality remained graded:

| Group | N | Deaths | Mortality |
|---:|---:|---:|---:|
| 1 | 703 | 177 | 25.18% |
| 2 | 743 | 224 | 30.15% |
| 3 | 387 | 188 | 48.58% |
| 4 | 124 | 94 | 75.81% |

Adjusted association:

| Comparison | OR | 95% CI | P |
|---|---:|---:|---:|
| Group 2 vs Group 1 | 1.11 | 0.87-1.42 | 0.401 |
| Group 3 vs Group 1 | 2.06 | 1.54-2.76 | 1.02e-06 |
| Group 4 vs Group 1 | 6.21 | 3.80-10.15 | 3.35e-13 |

Interpretation: The mortality gradient persisted after excluding short ICU stays, reducing concern that the findings were driven only by patients with incomplete 24-hour observation windows.

## 3. Bootstrap uncertainty for prediction-model improvement

Five-fold cross-validated prediction probabilities were used to estimate incremental AUROC and AUPRC. Bootstrap resampling was performed 500 times.

| Comparison | Delta AUROC | 95% CI | Delta AUPRC | 95% CI |
|---|---:|---|---:|---|
| Base + trajectory vs clinical base | 0.0433 | 0.0316-0.0568 | 0.0795 | 0.0575-0.1056 |
| Base + full lactate dynamics vs clinical base | 0.0572 | 0.0438-0.0711 | 0.1027 | 0.0777-0.1261 |
| Base + full lactate dynamics vs base + trajectory | 0.0139 | 0.0062-0.0218 | 0.0231 | 0.0078-0.0363 |

Interpretation: Dynamic lactate features improved discrimination and precision-recall performance beyond clinical variables, and bootstrap intervals did not cross zero.

## 4. Calibration and decision curve analysis

Calibration and DCA figures were generated for:

1. Clinical base model.
2. Clinical base model plus trajectory group.
3. Clinical base model plus full lactate dynamic features.

Generated figures:

```text
manuscript_figures/figure_calibration_q2.png
manuscript_figures/figure_dca_q2.png
```

Generated data files:

```text
manuscript_tables/table_q2_calibration_curve_data.csv
manuscript_tables/table_q2_decision_curve_data.csv
```

## Manuscript recommendation

For the Q2 submission version, present the original eICU re-clustering analysis as supportive, but emphasize the MIMIC-centroid validation as the stricter external validation. Add the observation-window sensitivity analysis and bootstrap model-improvement intervals to the Results section. Include calibration and DCA as new figures or supplementary figures depending on target journal limits.

