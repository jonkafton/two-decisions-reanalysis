#!/usr/bin/env python3
"""Prediction 3: apply the registered decision rule to the trigger-only run.

The thresholds are item counts on the 368 matched items, derived here from the
released verdicts rather than typed in: the published judge passed 240 of the
three systems' 1,104 verdicts, a mean of 80 items per system (21.7%).

  supported     published judge passes >= 40 (half that mean)
                and the second judge passes <= 7 (under 2%)
  failed        published judge passes <= 19 (below a quarter of that mean, 20)
                or the second judge passes >= 8 (2%: 8/368 = 2.17%, 7/368 = 1.90%)
  inconclusive  otherwise

  python3 preregistration/p3_score.py [--dir results]
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from common import RESULTS, SYSTEMS, matched_items, pool, published_verdicts, wilson  # noqa: E402

N = 368


def thresholds():
    j1 = published_verdicts()
    items = matched_items()
    assert len(items) == N
    total = sum(j1[(sid, s)] for sid in items for s in SYSTEMS)
    assert total == 240, total
    mean = total / len(SYSTEMS)                 # 80 items
    support_min = math.ceil(mean / 2)           # 40
    fail_max = math.ceil(mean / 4) - 1          # 19: below a quarter (20)
    second_fail_min = math.ceil(0.02 * N)       # 8
    assert (support_min, fail_max, second_fail_min) == (40, 19, 8)
    return mean, support_min, fail_max, second_fail_min


def decide(c1, c2, support_min, fail_max, second_fail_min):
    if c1 <= fail_max or c2 >= second_fail_min:
        return "failed"
    if c1 >= support_min:
        return "supported"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=RESULTS)
    args = ap.parse_args()
    mean, support_min, fail_max, second_fail_min = thresholds()
    rows = [json.loads(l) for l in open(os.path.join(args.dir, "p3_verdicts.jsonl"), encoding="utf-8") if l.strip()]
    items = set(matched_items())
    by_judge = {"published": {}, "second": {}}
    for r in rows:
        by_judge[r["judge"]][r["sample_id"]] = r
    for judge, got in by_judge.items():
        if set(got) != items:
            sys.exit(f"incomplete: {judge} judge has {len(got)} of {N} verdicts; resume the run first")
    c1 = sum(r["label"] == "correct" for r in by_judge["published"].values())
    c2 = sum(r["label"] == "correct" for r in by_judge["second"].values())
    flags = Counter(r.get("flag", "").split(":")[0] for r in rows if r.get("flag"))
    p = pool()
    by_type = {}
    for rel in sorted({p[s]["relation_type"] for s in items}):
        ids = [s for s in items if p[s]["relation_type"] == rel]
        by_type[rel] = {"n": len(ids),
                        "published": sum(by_judge["published"][s]["label"] == "correct" for s in ids),
                        "second": sum(by_judge["second"][s]["label"] == "correct" for s in ids)}
    lo1, hi1 = wilson(c1, N)[1:]  # percentages, one decimal
    lo2, hi2 = wilson(c2, N)[1:]
    result = {
        "prediction": 3,
        "n_items": N,
        "systems_mean_under_published_judge_items": mean,
        "published_judge_correct": c1, "published_judge_pct": round(100 * c1 / N, 2),
        "published_judge_wilson95_pct": [lo1, hi1],
        "second_judge_correct": c2, "second_judge_pct": round(100 * c2 / N, 2),
        "second_judge_wilson95_pct": [lo2, hi2],
        "thresholds": {"supported_if_published_at_least": support_min, "failed_if_published_at_most": fail_max,
                       "failed_if_second_at_least": second_fail_min},
        "decision": decide(c1, c2, support_min, fail_max, second_fail_min),
        "flags": dict(flags),
        "by_relation_type_exploratory": by_type,
    }
    out = os.path.join(args.dir, "p3_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
