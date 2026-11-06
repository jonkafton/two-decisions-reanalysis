#!/usr/bin/env python3
"""Prediction 2 (H2): apply the registered decision rule to a driver run.

H2, as registered (osf.io/xufd9): "Improving query planning raises factual accuracy
more than cognitive accuracy." Same backbone, two judges from different model
families, a verdict counting only when both pass it.

  supported     planning on raises both-pass accuracy on at least 100 LoCoMo
                questions by at least 5 points more than on at least 100
                LoCoMo-Plus items: D = (F_on - F_off) - (C_on - C_off) >= 5
  failed        the cognitive gain equals or exceeds the factual gain: D <= 0
  inconclusive  0 < D < 5
  not testable  fewer than 100 scored items on either track

The statistic is the factual gain minus the cognitive gain, in percentage points,
on the point estimates. A percentile bootstrap interval over items (resampling the
two tracks independently, seed 20261104, 10,000 draws) is reported beside it and
does not decide. An item whose palace lookup failed or timed out in either arm is
excluded from both arms (paired design), never scored as a miss; the number is
reported. A judge verdict recorded wrong with a flag (unparseable output) counts as
wrong, as H3's registered rule counts it, and the number flagged is reported.

The run directory is what tests/eval/locomo/locomo-driver.ts (PresenceAI) writes:
items.json, recalls.jsonl, verdicts.jsonl, meta.json.

  python3 preregistration/h2/p2_score.py --run <run dir> [--min-items 100] [--boot 10000]
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

SEED = 20261104
JUDGES = ("published", "second")
ARMS = ("system-on", "system-off", "none")


def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def both_pass(verdicts, track, item_id, arm, judges):
    labels = [verdicts.get((track, item_id, arm, j)) for j in judges]
    if any(l is None for l in labels):
        return None
    return all(l == "correct" for l in labels)


def accuracy(passes):
    n = len(passes)
    return (sum(passes) / n if n else float("nan")), n


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the driver's run directory")
    ap.add_argument("--min-items", type=int, default=100)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--judges", default=",".join(JUDGES), help="which judges must pass (default: both)")
    args = ap.parse_args()
    judges = tuple(j for j in args.judges.split(",") if j)

    items = json.load(open(os.path.join(args.run, "items.json"), encoding="utf-8"))
    meta = json.load(open(os.path.join(args.run, "meta.json"), encoding="utf-8"))
    recalls = read_jsonl(os.path.join(args.run, "recalls.jsonl"))
    verdict_rows = read_jsonl(os.path.join(args.run, "verdicts.jsonl"))

    ids = {"cognitive": [i["sampleId"] for i in items["cognitive"]], "factual": [q["id"] for q in items["factual"]]}
    relation = {i["sampleId"]: i["relation"] for i in items["cognitive"]}
    category = {q["id"]: q["categoryName"] for q in items["factual"]}

    excluded = defaultdict(list)
    for r in recalls:
        if r.get("excluded"):
            excluded[(r["track"], r["id"])].append({"arm": r["arm"], "reason": r.get("reason")})

    verdicts = {}
    flagged = Counter()
    errors = Counter()
    for v in verdict_rows:
        key = (v["track"], v["id"], v["arm"], v["judge"])
        if v.get("error"):
            errors[v["judge"]] += 1
            continue
        verdicts[key] = v.get("label")
        if v.get("flag"):
            flagged[v["judge"]] += 1

    result = {"prediction": 2, "registration": meta.get("registration"), "run": args.run, "judges_required": list(judges),
              "design": meta.get("design"), "models": meta.get("models"), "switches": meta.get("switches")}
    passes = {}
    for track in ("cognitive", "factual"):
        scored = [i for i in ids[track] if (track, i) not in excluded]
        incomplete = []
        per_arm = {}
        for arm in ARMS:
            ps = {}
            for i in scored:
                p = both_pass(verdicts, track, i, arm, judges)
                if p is None:
                    incomplete.append((i, arm))
                else:
                    ps[i] = p
            per_arm[arm] = ps
        common = [i for i in scored if all(i in per_arm[a] for a in ("system-on", "system-off"))]
        passes[track] = {arm: [per_arm[arm][i] for i in common] for arm in ("system-on", "system-off")}
        passes[track]["none"] = [per_arm["none"][i] for i in common if i in per_arm["none"]]
        acc = {}
        for arm in ARMS:
            a, n = accuracy(passes[track][arm])
            acc[arm] = {"n": n, "correct": int(sum(passes[track][arm])), "pct": round(100 * a, 2) if n else None,
                        "wilson95_pct": list(wilson(int(sum(passes[track][arm])), n))}
        on, off = acc["system-on"]["pct"], acc["system-off"]["pct"]
        gain = round(on - off, 2) if on is not None and off is not None else None
        group = relation if track == "cognitive" else category
        by_group = {}
        for g in sorted(set(group[i] for i in common)):
            gi = [i for i in common if group[i] == g]
            by_group[g] = {"n": len(gi)}
            for arm_ in ARMS:
                by_group[g][arm_] = int(sum(per_arm[arm_][i] for i in gi if i in per_arm[arm_]))
        result[track] = {"items": len(ids[track]), "excluded_failed_lookup": len([i for i in ids[track] if (track, i) in excluded]),
                         "incomplete_verdicts": len(incomplete), "scored": len(common), "accuracy_both_pass": acc,
                         "gain_on_minus_off_points": gain, "by_group_exploratory": by_group}
        result[track]["common_ids"] = common

    nc, nf = result["cognitive"]["scored"], result["factual"]["scored"]
    gc, gf = result["cognitive"]["gain_on_minus_off_points"], result["factual"]["gain_on_minus_off_points"]
    if nc < args.min_items or nf < args.min_items or gc is None or gf is None:
        decision = "not testable as registered"
        d = None
    else:
        d = round(gf - gc, 2)
        decision = "supported" if d >= 5 else "failed" if d <= 0 else "inconclusive"

    boot = None
    if d is not None and args.boot > 0:
        rng = random.Random(SEED)
        c_on, c_off = passes["cognitive"]["system-on"], passes["cognitive"]["system-off"]
        f_on, f_off = passes["factual"]["system-on"], passes["factual"]["system-off"]
        draws = []
        for _ in range(args.boot):
            ci = [rng.randrange(nc) for _ in range(nc)]
            fi = [rng.randrange(nf) for _ in range(nf)]
            gcb = 100 * (sum(c_on[i] for i in ci) - sum(c_off[i] for i in ci)) / nc
            gfb = 100 * (sum(f_on[i] for i in fi) - sum(f_off[i] for i in fi)) / nf
            draws.append(gfb - gcb)
        draws.sort()
        lo, hi = draws[int(0.025 * args.boot)], draws[min(args.boot - 1, int(0.975 * args.boot))]
        boot = {"seed": SEED, "draws": args.boot, "percentile95": [round(lo, 2), round(hi, 2)],
                "method": "items resampled with replacement within each track, independently; interval does not decide"}

    result.update({
        "statistic_factual_gain_minus_cognitive_gain_points": d,
        "bootstrap": boot,
        "decision": decision,
        "thresholds": {"supported_if_at_least": 5, "failed_if_at_most": 0, "min_items_per_track": args.min_items},
        "flagged_verdicts_counted_wrong": dict(flagged),
        "verdict_errors_not_scored": dict(errors),
        "exclusions": [{"track": t, "id": i, "arms": a} for (t, i), a in sorted(excluded.items())],
    })
    for track in ("cognitive", "factual"):
        result[track].pop("common_ids")
    out = os.path.join(args.run, "p2_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(json.dumps(result, indent=1))
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
