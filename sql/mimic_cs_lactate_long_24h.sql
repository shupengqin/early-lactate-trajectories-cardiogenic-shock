-- Long-format lactate observations for trajectory modeling.
-- Requires public.cs_lactate_24h_cohort from mimic_cs_lactate_cohort.sql.

DROP TABLE IF EXISTS public.cs_lactate_24h_long;

CREATE TABLE public.cs_lactate_24h_long AS
SELECT
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    l.charttime,
    EXTRACT(EPOCH FROM (l.charttime - c.intime)) / 3600.0 AS lactate_hour,
    l.valuenum AS lactate
FROM public.cs_lactate_24h_cohort c
INNER JOIN mimiciv_hosp.labevents l
    ON c.subject_id = l.subject_id
   AND c.hadm_id = l.hadm_id
WHERE c.eligible_lactate_trajectory_24h = 1
  AND l.itemid IN (50813, 52442, 53154)
  AND l.valuenum IS NOT NULL
  AND l.valuenum > 0
  AND l.valuenum <= 50
  AND l.charttime >= c.intime
  AND l.charttime < c.intime + INTERVAL '24 hour';

CREATE INDEX IF NOT EXISTS idx_cs_lactate_24h_long_stay
    ON public.cs_lactate_24h_long (stay_id, lactate_hour);

SELECT
    COUNT(*) AS lactate_observations,
    COUNT(DISTINCT stay_id) AS stays,
    ROUND(AVG(lactate)::numeric, 2) AS mean_lactate,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lactate)::numeric, 2) AS median_lactate
FROM public.cs_lactate_24h_long;

