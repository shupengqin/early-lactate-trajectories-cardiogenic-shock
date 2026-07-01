import csv
import json
import math
from collections import Counter


INPUT = "outputs/mimic_cs_lactate_24h_cohort.csv"
OUTPUT = "outputs/mimic_initial_qc.json"


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


def quantiles(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {}

    def q(p):
        pos = (len(values) - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return values[int(pos)]
        return values[lo] * (hi - pos) + values[hi] * (pos - lo)

    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "sd": (
            sum((v - sum(values) / len(values)) ** 2 for v in values) / (len(values) - 1)
        )
        ** 0.5
        if len(values) > 1
        else None,
        "min": values[0],
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
        "max": values[-1],
    }


def mortality(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0, "deaths": 0, "mortality_pct": None}
    deaths = sum(1 for r in rows if r["hospital_expire_flag"] == "1")
    return {"n": n, "deaths": deaths, "mortality_pct": deaths / n * 100}


def main():
    with open(INPUT, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_lact_n = {}
    for threshold in [1, 2, 3, 4, 5]:
        subset = [r for r in rows if (as_float(r.get("lactate_n_24h")) or 0) >= threshold]
        by_lact_n[f"lactate_ge{threshold}"] = mortality(subset)

    eligible = [r for r in rows if (as_float(r.get("lactate_n_24h")) or 0) >= 2]
    high_last = [r for r in eligible if r.get("persistent_high_lactate_24h") == "1"]
    not_high_last = [r for r in eligible if r.get("persistent_high_lactate_24h") == "0"]
    any_high = [r for r in eligible if r.get("any_high_lactate_24h") == "1"]
    no_high = [r for r in eligible if r.get("any_high_lactate_24h") == "0"]

    result = {
        "overall": mortality(rows),
        "by_lactate_count": by_lact_n,
        "eligible_lactate_trajectory_24h": mortality(eligible),
        "persistent_high_lactate_24h": {
            "yes": mortality(high_last),
            "no": mortality(not_high_last),
        },
        "any_high_lactate_24h": {
            "yes": mortality(any_high),
            "no": mortality(no_high),
        },
        "lactate_distributions_eligible": {
            "initial_lactate_24h": quantiles(
                [as_float(r.get("initial_lactate_24h")) for r in eligible]
            ),
            "last_lactate_24h": quantiles(
                [as_float(r.get("last_lactate_24h")) for r in eligible]
            ),
            "peak_lactate_24h": quantiles(
                [as_float(r.get("peak_lactate_24h")) for r in eligible]
            ),
            "lactate_clearance_24h": quantiles(
                [as_float(r.get("lactate_clearance_24h")) for r in eligible]
            ),
            "lactate_slope_24h": quantiles(
                [as_float(r.get("lactate_slope_24h")) for r in eligible]
            ),
        },
        "gender_counts": dict(Counter(r.get("gender", "") for r in rows)),
    }

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
