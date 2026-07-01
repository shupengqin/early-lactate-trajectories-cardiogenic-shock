-- MIMIC-IV cardiogenic shock analysis dataset with covariates.
-- Requires public.cs_lactate_24h_cohort.

DROP TABLE IF EXISTS public.cs_lactate_analysis_dataset;

CREATE TABLE public.cs_lactate_analysis_dataset AS
WITH base AS (
    SELECT *
    FROM public.cs_lactate_24h_cohort
    WHERE eligible_lactate_trajectory_24h = 1
),
vent_24h AS (
    SELECT
        b.stay_id,
        1 AS mechvent_24h
    FROM base b
    INNER JOIN mimiciv_derived.ventilation v
        ON b.stay_id = v.stay_id
       AND v.starttime < b.intime + INTERVAL '24 hour'
       AND v.endtime > b.intime
    WHERE v.ventilation_status IN ('InvasiveVent', 'NonInvasiveVent')
    GROUP BY b.stay_id
),
vaso_24h AS (
    SELECT
        b.stay_id,
        1 AS vasoactive_24h,
        MAX(GREATEST(
            COALESCE(v.dopamine, 0),
            COALESCE(v.epinephrine, 0),
            COALESCE(v.norepinephrine, 0),
            COALESCE(v.phenylephrine, 0),
            COALESCE(v.vasopressin, 0),
            COALESCE(v.dobutamine, 0),
            COALESCE(v.milrinone, 0)
        )) AS max_raw_vaso_rate_24h
    FROM base b
    INNER JOIN mimiciv_derived.vasoactive_agent v
        ON b.stay_id = v.stay_id
       AND v.starttime < b.intime + INTERVAL '24 hour'
       AND v.endtime > b.intime
    GROUP BY b.stay_id
),
ned_24h AS (
    SELECT
        b.stay_id,
        MAX(n.norepinephrine_equivalent_dose) AS max_norepi_equiv_24h,
        AVG(n.norepinephrine_equivalent_dose) AS mean_norepi_equiv_24h
    FROM base b
    INNER JOIN mimiciv_derived.norepinephrine_equivalent_dose n
        ON b.stay_id = n.stay_id
       AND n.starttime < b.intime + INTERVAL '24 hour'
       AND n.endtime > b.intime
    GROUP BY b.stay_id
),
aki_0_24 AS (
    SELECT
        b.stay_id,
        MAX(k.aki_stage) AS aki_stage_0_24h
    FROM base b
    INNER JOIN mimiciv_derived.kdigo_stages k
        ON b.stay_id = k.stay_id
       AND k.charttime >= b.intime
       AND k.charttime < b.intime + INTERVAL '24 hour'
    GROUP BY b.stay_id
),
aki_24_72 AS (
    SELECT
        b.stay_id,
        MAX(k.aki_stage) AS aki_stage_24_72h
    FROM base b
    INNER JOIN mimiciv_derived.kdigo_stages k
        ON b.stay_id = k.stay_id
       AND k.charttime >= b.intime + INTERVAL '24 hour'
       AND k.charttime < b.intime + INTERVAL '72 hour'
    GROUP BY b.stay_id
)
SELECT
    b.subject_id,
    b.hadm_id,
    b.stay_id,
    b.intime,
    b.outtime,
    b.hospital_expire_flag,
    b.gender,
    b.age,

    -- Lactate features
    b.lactate_n_24h,
    b.initial_lactate_24h,
    b.last_lactate_24h,
    b.peak_lactate_24h,
    b.min_lactate_24h,
    b.lactate_slope_24h,
    b.lactate_clearance_24h,
    b.persistent_high_lactate_24h,
    b.any_high_lactate_24h,

    -- Severity scores
    sofa.sofa,
    sofa.respiration AS sofa_respiration,
    sofa.coagulation AS sofa_coagulation,
    sofa.liver AS sofa_liver,
    sofa.cardiovascular AS sofa_cardiovascular,
    sofa.cns AS sofa_cns,
    sofa.renal AS sofa_renal,
    saps.sapsii,
    saps.sapsii_prob,
    oasis.oasis,
    oasis.oasis_prob,

    -- Comorbidity
    ch.charlson_comorbidity_index,
    ch.myocardial_infarct,
    ch.congestive_heart_failure,
    ch.peripheral_vascular_disease,
    ch.cerebrovascular_disease,
    ch.chronic_pulmonary_disease,
    ch.diabetes_without_cc,
    ch.diabetes_with_cc,
    ch.renal_disease,
    ch.malignant_cancer,
    ch.severe_liver_disease,

    -- First-day vitals
    vs.heart_rate_mean,
    vs.sbp_mean,
    vs.dbp_mean,
    vs.mbp_mean,
    vs.resp_rate_mean,
    vs.temperature_mean,
    vs.spo2_mean,

    -- First-day labs
    lab.hemoglobin_min,
    lab.platelets_min,
    lab.wbc_max,
    lab.bicarbonate_min,
    lab.bun_max,
    lab.creatinine_max,
    lab.sodium_min,
    lab.potassium_max,
    lab.bilirubin_total_max,
    lab.inr_max,

    -- Support and AKI
    COALESCE(vent.mechvent_24h, 0) AS mechvent_24h,
    COALESCE(vaso.vasoactive_24h, 0) AS vasoactive_24h,
    vaso.max_raw_vaso_rate_24h,
    ned.max_norepi_equiv_24h,
    ned.mean_norepi_equiv_24h,
    COALESCE(aki0.aki_stage_0_24h, 0) AS aki_stage_0_24h,
    COALESCE(aki72.aki_stage_24_72h, 0) AS aki_stage_24_72h,
    CASE WHEN COALESCE(aki0.aki_stage_0_24h, 0) = 0
          AND COALESCE(aki72.aki_stage_24_72h, 0) >= 1
         THEN 1 ELSE 0 END AS new_aki_24_72h
FROM base b
LEFT JOIN mimiciv_derived.first_day_sofa sofa
    ON b.stay_id = sofa.stay_id
LEFT JOIN mimiciv_derived.sapsii saps
    ON b.stay_id = saps.stay_id
LEFT JOIN mimiciv_derived.oasis oasis
    ON b.stay_id = oasis.stay_id
LEFT JOIN mimiciv_derived.charlson ch
    ON b.subject_id = ch.subject_id
   AND b.hadm_id = ch.hadm_id
LEFT JOIN mimiciv_derived.first_day_vitalsign vs
    ON b.stay_id = vs.stay_id
LEFT JOIN mimiciv_derived.first_day_lab lab
    ON b.stay_id = lab.stay_id
LEFT JOIN vent_24h vent
    ON b.stay_id = vent.stay_id
LEFT JOIN vaso_24h vaso
    ON b.stay_id = vaso.stay_id
LEFT JOIN ned_24h ned
    ON b.stay_id = ned.stay_id
LEFT JOIN aki_0_24 aki0
    ON b.stay_id = aki0.stay_id
LEFT JOIN aki_24_72 aki72
    ON b.stay_id = aki72.stay_id;

CREATE INDEX IF NOT EXISTS idx_cs_lactate_analysis_stay
    ON public.cs_lactate_analysis_dataset (stay_id);

SELECT
    COUNT(*) AS n,
    SUM(hospital_expire_flag)::int AS deaths,
    ROUND(AVG(hospital_expire_flag)::numeric * 100, 2) AS mortality_pct,
    SUM(persistent_high_lactate_24h)::int AS persistent_high_n,
    SUM(new_aki_24_72h)::int AS new_aki_24_72h_n,
    SUM(mechvent_24h)::int AS mechvent_24h_n,
    SUM(vasoactive_24h)::int AS vasoactive_24h_n
FROM public.cs_lactate_analysis_dataset;

