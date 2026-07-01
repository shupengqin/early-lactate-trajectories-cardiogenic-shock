-- eICU-CRD cardiogenic shock lactate validation cohort.
-- Database: eicu

DROP TABLE IF EXISTS public.cs_lactate_24h_cohort;

CREATE TABLE public.cs_lactate_24h_cohort AS
WITH cs_dx AS (
    SELECT DISTINCT patientunitstayid
    FROM diagnosis
    WHERE lower(diagnosisstring) LIKE '%cardiogenic shock%'
       OR icd9code LIKE '%785.51%'
       OR icd9code LIKE '%78551%'
),
first_icu AS (
    SELECT *
    FROM (
        SELECT
            p.*,
            ROW_NUMBER() OVER (
                PARTITION BY p.uniquepid
                ORDER BY p.hospitaldischargeyear, p.hospitaladmitoffset, p.unitvisitnumber, p.patientunitstayid
            ) AS rn
        FROM patient p
        INNER JOIN cs_dx d
            ON p.patientunitstayid = d.patientunitstayid
    ) x
    WHERE rn = 1
),
lactate_24h AS (
    SELECT
        f.patientunitstayid,
        l.labresultoffset / 60.0 AS lactate_hour,
        l.labresult::double precision AS lactate
    FROM first_icu f
    INNER JOIN lab l
        ON f.patientunitstayid = l.patientunitstayid
    WHERE lower(l.labname) = 'lactate'
      AND l.labresult IS NOT NULL
      AND l.labresult > 0
      AND l.labresult <= 50
      AND l.labresultoffset >= 0
      AND l.labresultoffset < 24 * 60
),
lactate_ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY patientunitstayid ORDER BY lactate_hour ASC) AS rn_asc,
        ROW_NUMBER() OVER (PARTITION BY patientunitstayid ORDER BY lactate_hour DESC) AS rn_desc
    FROM lactate_24h
),
lactate_features AS (
    SELECT
        patientunitstayid,
        COUNT(*) AS lactate_n_24h,
        MAX(lactate) AS peak_lactate_24h,
        MIN(lactate) AS min_lactate_24h,
        MAX(lactate) FILTER (WHERE rn_asc = 1) AS initial_lactate_24h,
        MAX(lactate) FILTER (WHERE rn_desc = 1) AS last_lactate_24h,
        REGR_SLOPE(lactate, lactate_hour) AS lactate_slope_24h
    FROM lactate_ranked
    GROUP BY patientunitstayid
),
lactate_final AS (
    SELECT
        *,
        CASE
            WHEN initial_lactate_24h > 0
                THEN (initial_lactate_24h - last_lactate_24h) / initial_lactate_24h * 100.0
            ELSE NULL
        END AS lactate_clearance_24h,
        CASE WHEN last_lactate_24h >= 4 THEN 1 ELSE 0 END AS persistent_high_lactate_24h,
        CASE WHEN peak_lactate_24h >= 4 THEN 1 ELSE 0 END AS any_high_lactate_24h
    FROM lactate_features
)
SELECT
    f.patientunitstayid,
    f.uniquepid,
    f.hospitalid,
    f.gender,
    CASE
        WHEN f.age = '> 89' THEN 90
        WHEN f.age ~ '^[0-9]+$' THEN f.age::int
        ELSE NULL
    END AS age,
    f.ethnicity,
    f.hospitaldischargestatus,
    CASE WHEN lower(f.hospitaldischargestatus) = 'expired' THEN 1 ELSE 0 END AS hospital_expire_flag,
    f.unitdischargestatus,
    CASE WHEN lower(f.unitdischargestatus) = 'expired' THEN 1 ELSE 0 END AS unit_expire_flag,
    f.unitdischargeoffset,
    lf.lactate_n_24h,
    lf.initial_lactate_24h,
    lf.last_lactate_24h,
    lf.peak_lactate_24h,
    lf.min_lactate_24h,
    lf.lactate_slope_24h,
    lf.lactate_clearance_24h,
    lf.persistent_high_lactate_24h,
    lf.any_high_lactate_24h,
    CASE WHEN lf.lactate_n_24h >= 2 THEN 1 ELSE 0 END AS eligible_lactate_trajectory_24h
FROM first_icu f
LEFT JOIN lactate_final lf
    ON f.patientunitstayid = lf.patientunitstayid;

CREATE INDEX IF NOT EXISTS idx_eicu_cs_lactate_24h_stay
    ON public.cs_lactate_24h_cohort (patientunitstayid);

DROP TABLE IF EXISTS public.cs_lactate_24h_long;

CREATE TABLE public.cs_lactate_24h_long AS
SELECT
    c.patientunitstayid,
    l.labresultoffset / 60.0 AS lactate_hour,
    l.labresult::double precision AS lactate
FROM public.cs_lactate_24h_cohort c
INNER JOIN lab l
    ON c.patientunitstayid = l.patientunitstayid
WHERE c.eligible_lactate_trajectory_24h = 1
  AND lower(l.labname) = 'lactate'
  AND l.labresult IS NOT NULL
  AND l.labresult > 0
  AND l.labresult <= 50
  AND l.labresultoffset >= 0
  AND l.labresultoffset < 24 * 60;

CREATE INDEX IF NOT EXISTS idx_eicu_cs_lactate_24h_long_stay
    ON public.cs_lactate_24h_long (patientunitstayid, lactate_hour);

SELECT
    COUNT(*) AS cs_first_icu,
    SUM(hospital_expire_flag)::int AS deaths,
    ROUND(AVG(hospital_expire_flag)::numeric * 100, 2) AS mortality_pct,
    COUNT(*) FILTER (WHERE lactate_n_24h >= 1) AS lact24_ge1,
    COUNT(*) FILTER (WHERE lactate_n_24h >= 2) AS lact24_ge2,
    COUNT(*) FILTER (WHERE lactate_n_24h >= 3) AS lact24_ge3
FROM public.cs_lactate_24h_cohort;

SELECT
    COUNT(*) AS lactate_observations,
    COUNT(DISTINCT patientunitstayid) AS stays,
    ROUND(AVG(lactate)::numeric, 2) AS mean_lactate,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lactate)::numeric, 2) AS median_lactate
FROM public.cs_lactate_24h_long;

