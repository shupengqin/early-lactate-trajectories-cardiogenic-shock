import csv
import json
import math
from statistics import mean


INPUT = "outputs/mimic_cs_analysis_dataset.csv"
OUTPUT = "outputs/mimic_analysis_qc_and_logistic.json"


def as_float(value):
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def summarize_numeric(rows, col):
    vals = [as_float(r.get(col)) for r in rows]
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return {"n": 0}

    def q(p):
        pos = (len(vals) - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return vals[int(pos)]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)

    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "min": vals[0],
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
        "max": vals[-1],
    }


def missingness(rows):
    cols = rows[0].keys()
    out = {}
    n = len(rows)
    for col in cols:
        miss = sum(1 for r in rows if r.get(col) in ("", None))
        out[col] = {"missing": miss, "missing_pct": miss / n * 100}
    return dict(sorted(out.items(), key=lambda kv: kv[1]["missing_pct"], reverse=True))


def two_by_two_or(rows, exposure_col, outcome_col="hospital_expire_flag"):
    a = b = c = d = 0
    for r in rows:
        e = str(r.get(exposure_col)) == "1"
        y = str(r.get(outcome_col)) == "1"
        if e and y:
            a += 1
        elif e and not y:
            b += 1
        elif not e and y:
            c += 1
        else:
            d += 1
    # Haldane-Anscombe correction, robust to zero cells.
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds_ratio = (aa * dd) / (bb * cc)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lo = math.exp(math.log(odds_ratio) - 1.96 * se)
    hi = math.exp(math.log(odds_ratio) + 1.96 * se)
    return {
        "a_exposed_death": a,
        "b_exposed_survive": b,
        "c_unexposed_death": c,
        "d_unexposed_survive": d,
        "or": odds_ratio,
        "ci95_low": lo,
        "ci95_high": hi,
    }


def try_statsmodels_logistic(rows):
    try:
        import pandas as pd
        import statsmodels.api as sm
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    covariates = [
        "persistent_high_lactate_24h",
        "age",
        "sofa",
        "charlson_comorbidity_index",
        "mbp_mean",
        "creatinine_max",
        "mechvent_24h",
        "vasoactive_24h",
    ]
    data = []
    for r in rows:
        item = {"hospital_expire_flag": as_float(r.get("hospital_expire_flag"))}
        ok = item["hospital_expire_flag"] is not None
        for col in covariates:
            item[col] = as_float(r.get(col))
            if item[col] is None:
                ok = False
        if ok:
            data.append(item)
    if len(data) < 50:
        return {"available": True, "error": "too few complete cases", "n": len(data)}

    df = pd.DataFrame(data)
    y = df["hospital_expire_flag"]
    x = sm.add_constant(df[covariates], has_constant="add")
    model = sm.Logit(y, x).fit(disp=False)
    params = model.params
    conf = model.conf_int()
    result = {
        "available": True,
        "n_complete_cases": int(len(df)),
        "aic": float(model.aic),
        "terms": {},
    }
    for term in params.index:
        result["terms"][term] = {
            "coef": float(params[term]),
            "or": float(math.exp(params[term])),
            "ci95_low": float(math.exp(conf.loc[term, 0])),
            "ci95_high": float(math.exp(conf.loc[term, 1])),
            "p": float(model.pvalues[term]),
        }
    return result


def main():
    with open(INPUT, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    deaths = sum(1 for r in rows if r["hospital_expire_flag"] == "1")
    result = {
        "n": len(rows),
        "deaths": deaths,
        "mortality_pct": deaths / len(rows) * 100,
        "missingness_top20": dict(list(missingness(rows).items())[:20]),
        "numeric_summary": {
            col: summarize_numeric(rows, col)
            for col in [
                "age",
                "initial_lactate_24h",
                "last_lactate_24h",
                "peak_lactate_24h",
                "lactate_clearance_24h",
                "sofa",
                "sapsii",
                "oasis",
                "charlson_comorbidity_index",
                "mbp_mean",
                "creatinine_max",
            ]
        },
        "crude_or": {
            "persistent_high_lactate_24h": two_by_two_or(rows, "persistent_high_lactate_24h"),
            "any_high_lactate_24h": two_by_two_or(rows, "any_high_lactate_24h"),
            "new_aki_24_72h": two_by_two_or(rows, "new_aki_24_72h"),
        },
        "adjusted_logistic": try_statsmodels_logistic(rows),
    }

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

