-- MIMIC-IV v3.1 cardiogenic shock lactate cohort
-- Database: mimiciv31
-- Main exposure window: ICU admission to 24 hours

DROP TABLE IF EXISTS public.cs_lactate_24h_cohort;

CREATE TABLE public.cs_lactate_24h_cohort AS
WITH cs_dx AS (
    SELECT DISTINCT
        subject_id,
        hadm_id
    FROM mimiciv_hosp.diagnoses_icd
    WHERE icd_code IN ('78551', 'R570')
),
cs_icu AS (
    SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime,
        i.outtime,
        a.admittime,
        a.dischtime,
        a.deathtime,
        a.hospital_expire_flag,
        p.gender,
        p.anchor_age AS age,
        ROW_NUMBER() OVER (
            PARTITION BY i.subject_id
            ORDER BY i.intime
        ) AS rn
    FROM mimiciv_icu.icustays i
    INNER JOIN cs_dx d
        ON i.subject_id = d.subject_id
       AND i.hadm_id = d.hadm_id
    INNER JOIN mimiciv_hosp.admissions a
        ON i.subject_id = a.subject_id
       AND i.hadm_id = a.hadm_id
    INNER JOIN mimiciv_hosp.patients p
        ON i.subject_id = p.subject_id
),
base AS (
    SELECT *
    FROM cs_icu
    WHERE rn = 1
),
lactate_24h AS (
    SELECT
        b.subject_id,
        b.hadm_id,
        b.stay_id,
        l.charttime,
        EXTRACT(EPOCH FROM (l.charttime - b.intime)) / 3600.0 AS lactate_hour,
        l.valuenum AS lactate
    FROM base b
    INNER JOIN mimiciv_hosp.labevents l
        ON b.subject_id = l.subject_id
       AND b.hadm_id = l.hadm_id
    WHERE l.itemid IN (50813, 52442, 53154)
      AND l.valuenum IS NOT NULL
      AND l.valuenum > 0
      AND l.valuenum <= 50
      AND l.charttime >= b.intime
      AND l.charttime < b.intime + INTERVAL '24 hour'
),
lactate_ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY charttime ASC) AS rn_asc,
        ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY charttime DESC) AS rn_desc
    FROM lactate_24h
),
lactate_features AS (
    SELECT
        stay_id,
        COUNT(*) AS lactate_n_24h,
        MAX(lactate) AS peak_lactate_24h,
        MIN(lactate) AS min_lactate_24h,
        MAX(lactate) FILTER (WHERE rn_asc = 1) AS initial_lactate_24h,
        MAX(lactate) FILTER (WHERE rn_desc = 1) AS last_lactate_24h,
        REGR_SLOPE(lactate, lactate_hour) AS lactate_slope_24h
    FROM lactate_ranked
    GROUP BY stay_id
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
    b.subject_id,
    b.hadm_id,
    b.stay_id,
    b.intime,
    b.outtime,
    b.admittime,
    b.dischtime,
    b.deathtime,
    b.hospital_expire_flag,
    b.gender,
    b.age,
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
FROM base b
LEFT JOIN lactate_final lf
    ON b.stay_id = lf.stay_id;

CREATE INDEX IF NOT EXISTS idx_cs_lactate_24h_stay
    ON public.cs_lactate_24h_cohort (stay_id);

CREATE INDEX IF NOT EXISTS idx_cs_lactate_24h_subject_hadm
    ON public.cs_lactate_24h_cohort (subject_id, hadm_id);

-- Summary
SELECT
    COUNT(*) AS cs_first_icu,
    SUM(hospital_expire_flag)::int AS hospital_deaths,
    ROUND(AVG(hospital_expire_flag)::numeric * 100, 2) AS mortality_pct,
    COUNT(*) FILTER (WHERE lactate_n_24h >= 1) AS lact24_ge1,
    COUNT(*) FILTER (WHERE lactate_n_24h >= 2) AS lact24_ge2,
    COUNT(*) FILTER (WHERE lactate_n_24h >= 3) AS lact24_ge3
FROM public.cs_lactate_24h_cohort;

