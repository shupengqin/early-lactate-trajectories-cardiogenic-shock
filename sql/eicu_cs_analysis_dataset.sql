-- eICU external validation analysis dataset.
-- Requires public.cs_lactate_24h_cohort.

DROP TABLE IF EXISTS public.cs_lactate_analysis_dataset;

CREATE TABLE public.cs_lactate_analysis_dataset AS
WITH base AS (
    SELECT *
    FROM public.cs_lactate_24h_cohort
    WHERE eligible_lactate_trajectory_24h = 1
),
vent AS (
    SELECT
        patientunitstayid,
        MAX(COALESCE(vent, 0)) AS vent,
        MAX(COALESCE(intubated, 0)) AS intubated,
        MAX(COALESCE(dialysis, 0)) AS dialysis,
        MAX(urine) AS urine,
        MAX(temperature) AS temperature,
        MAX(respiratoryrate) AS respiratoryrate,
        MAX(heartrate) AS heartrate,
        MAX(meanbp) AS meanbp,
        MAX(creatinine) AS creatinine,
        MAX(bun) AS bun,
        MAX(wbc) AS wbc,
        MAX(sodium) AS sodium,
        MAX(bilirubin) AS bilirubin,
        MAX(ph) AS ph
    FROM apacheapsvar
    GROUP BY patientunitstayid
),
apache AS (
    SELECT
        patientunitstayid,
        acutephysiologyscore,
        apachescore,
        apacheversion,
        predictedhospitalmortality,
        actualhospitalmortality,
        actualicumortality,
        actualventdays
    FROM apachepatientresult
    WHERE apacheversion = 'IVa'
),
vaso AS (
    SELECT
        b.patientunitstayid,
        1 AS vasoactive_24h
    FROM base b
    INNER JOIN infusiondrug i
        ON b.patientunitstayid = i.patientunitstayid
    WHERE i.infusionoffset >= 0
      AND i.infusionoffset < 24 * 60
      AND (
          lower(i.drugname) LIKE '%norepinephrine%'
          OR lower(i.drugname) LIKE '%noradrenaline%'
          OR lower(i.drugname) LIKE '%epinephrine%'
          OR lower(i.drugname) LIKE '%adrenaline%'
          OR lower(i.drugname) LIKE '%dopamine%'
          OR lower(i.drugname) LIKE '%dobutamine%'
          OR lower(i.drugname) LIKE '%phenylephrine%'
          OR lower(i.drugname) LIKE '%vasopressin%'
      )
    GROUP BY b.patientunitstayid
)
SELECT
    b.patientunitstayid,
    b.uniquepid,
    b.hospitalid,
    b.gender,
    b.age,
    b.ethnicity,
    b.hospital_expire_flag,
    b.unit_expire_flag,
    b.unitdischargeoffset,
    b.hospitaldischargeoffset,

    b.lactate_n_24h,
    b.initial_lactate_24h,
    b.last_lactate_24h,
    b.peak_lactate_24h,
    b.min_lactate_24h,
    b.lactate_slope_24h,
    b.lactate_clearance_24h,
    b.persistent_high_lactate_24h,
    b.any_high_lactate_24h,

    a.acutephysiologyscore,
    a.apachescore,
    a.apacheversion,
    a.predictedhospitalmortality,
    a.actualhospitalmortality,
    a.actualicumortality,
    v.vent,
    v.intubated,
    v.dialysis,
    v.urine,
    v.temperature,
    v.respiratoryrate,
    v.heartrate,
    v.meanbp,
    v.creatinine,
    v.bun,
    v.wbc,
    v.sodium,
    v.bilirubin,
    v.ph,
    COALESCE(vaso.vasoactive_24h, 0) AS vasoactive_24h
FROM base b
LEFT JOIN apache a
    ON b.patientunitstayid = a.patientunitstayid
LEFT JOIN vent v
    ON b.patientunitstayid = v.patientunitstayid
LEFT JOIN vaso
    ON b.patientunitstayid = vaso.patientunitstayid;

CREATE INDEX IF NOT EXISTS idx_eicu_cs_lactate_analysis_stay
    ON public.cs_lactate_analysis_dataset (patientunitstayid);

SELECT
    COUNT(*) AS n,
    SUM(hospital_expire_flag)::int AS deaths,
    ROUND(AVG(hospital_expire_flag)::numeric * 100, 2) AS mortality_pct,
    SUM(persistent_high_lactate_24h)::int AS persistent_high_n,
    SUM(COALESCE(vent, 0))::int AS vent_n,
    SUM(vasoactive_24h)::int AS vasoactive_n
FROM public.cs_lactate_analysis_dataset;
